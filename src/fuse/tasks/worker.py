"""The background worker: a separate process that takes tasks from the queue and runs the
agent on them, so long jobs never block the chat.

Run it (next to the bot):
    uv run fuse-worker

Each task gets its own conversation (thread `task:<id>`), checkpointed after every step
like your chat is. That is what makes crashes cheap: if the worker dies, its lease runs
out, the next claim continues from the last checkpoint, and only the unfinished step runs
again. Write actions inside a task still go through the approval gate and the actions log,
so a crash can never repeat one.
"""

import asyncio
import contextlib
import logging
import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

import httpx
import openai

from fuse.agent.runner import AgentReply, AgentRunner
from fuse.tasks.queue import Task, TaskStore

log = logging.getLogger("fuse.worker")

# Errors worth retrying later: the provider is rate limiting us or briefly unreachable.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    httpx.TransportError,
)

TASK_PROMPT = """Background task #{id}.

Goal: {goal}

Work through it with your tools. When you're finished, reply with the final result for the
user: complete, well organised, and easy to read on a phone."""


class TaskReporter(Protocol):
    """How the worker tells you what's happening. Telegram in production."""

    async def started(self, task: Task) -> None: ...
    async def needs_approval(self, task: Task, approval: dict) -> None: ...
    async def finished(self, task: Task, result: str) -> None: ...
    async def failed(self, task: Task, error: str) -> None: ...
    async def retrying(self, task: Task, error: str, delay: timedelta) -> None: ...
    async def cancelled(self, task: Task) -> None: ...


class LogReporter:
    """Only writes to the log. Used when there's nowhere else to report."""

    async def started(self, task: Task) -> None:
        log.info("task %s started", task.id)

    async def needs_approval(self, task: Task, approval: dict) -> None:
        log.info("task %s waiting for approval of %s", task.id, approval.get("tool"))

    async def finished(self, task: Task, result: str) -> None:
        log.info("task %s done (%d chars)", task.id, len(result))

    async def failed(self, task: Task, error: str) -> None:
        log.warning("task %s failed: %s", task.id, error)

    async def retrying(self, task: Task, error: str, delay: timedelta) -> None:
        log.info("task %s will retry in %s: %s", task.id, delay, error)

    async def cancelled(self, task: Task) -> None:
        log.info("task %s cancelled", task.id)


@dataclass
class _Flags:
    cancel_requested: bool = False
    lease_lost: bool = False


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


class Worker:
    def __init__(
        self,
        tasks: TaskStore,
        runner: AgentRunner,
        reporter: TaskReporter | None = None,
        worker_id: str | None = None,
        lease: timedelta = timedelta(minutes=2),
        timeout: timedelta = timedelta(minutes=15),
        retry_base: timedelta = timedelta(seconds=30),
        poll_seconds: float = 2.0,
    ) -> None:
        self.tasks, self.runner = tasks, runner
        self.reporter = reporter or LogReporter()
        self.id = worker_id or default_worker_id()
        self.lease, self.timeout = lease, timeout
        self.retry_base, self.poll_seconds = retry_base, poll_seconds

    async def _drive(self, task: Task) -> AgentReply:
        """Start, resume after an approval, or continue after a crash."""
        thread = task.thread_id
        if task.resume_with is not None:
            return await self.runner.resume_with(thread, task.resume_with)
        if await self.runner.has_history(thread):
            return await self.runner.continue_run(thread)
        await self.reporter.started(task)
        return await self.runner.send(thread, TASK_PROMPT.format(id=task.id, goal=task.goal))

    async def _keep_lease(self, task: Task, work: asyncio.Task, flags: _Flags) -> None:
        """Renew the lease while the task runs; stop the work if you cancel it or if this
        worker lost the task to another one."""
        while not work.done():
            await asyncio.sleep(self.lease.total_seconds() / 3)
            renewed = await self.tasks.renew(task.id, self.id, self.lease)
            if renewed is None:
                flags.lease_lost = True
                work.cancel()
            elif renewed:
                flags.cancel_requested = True
                work.cancel()

    async def run_one(self) -> bool:
        """Claim and run one task. False if there was nothing to do."""
        task = await self.tasks.claim(self.id, self.lease)
        if task is None:
            return False
        log.info("task %s claimed (attempt %d/%d)", task.id, task.attempts, task.max_attempts)
        if task.cancel_requested:
            await self.tasks.cancelled(task.id, self.id)
            await self.reporter.cancelled(task)
            return True

        flags = _Flags()
        work = asyncio.create_task(self._drive(task))
        keeper = asyncio.create_task(self._keep_lease(task, work, flags))
        try:
            reply = await asyncio.wait_for(work, self.timeout.total_seconds())
        except asyncio.CancelledError:
            if flags.cancel_requested:
                await self.tasks.cancelled(task.id, self.id)
                await self.reporter.cancelled(task)
                return True
            if flags.lease_lost:
                log.warning("task %s: lease lost to another worker; stopping", task.id)
                return True
            # This worker is shutting down: hand the task back right away instead of
            # making the next worker wait for the lease to expire.
            await self.tasks.retry_later(task.id, self.id, "worker stopped", timedelta(0))
            raise
        except TimeoutError:
            minutes = int(self.timeout.total_seconds() // 60)
            await self._fail(task, f"It took longer than {minutes} minutes, so I stopped it.")
            return True
        except TRANSIENT_ERRORS as exc:
            await self._retry_or_fail(task, f"{type(exc).__name__}: {exc}")
            return True
        except Exception as exc:
            log.exception("task %s crashed", task.id)
            await self._fail(task, f"Something went wrong: {type(exc).__name__}.")
            return True
        finally:
            keeper.cancel()

        await self._settle(task, reply)
        return True

    async def _settle(self, task: Task, reply: AgentReply) -> None:
        if reply.kind == "approval":
            if await self.tasks.wait_for_approval(task.id, self.id, reply.approval):
                await self.reporter.needs_approval(task, reply.approval)
        elif reply.hit_limit:
            await self._fail(task, "It needed more steps than its budget allows.")
        elif await self.tasks.finish(task.id, self.id, reply.text):
            await self.reporter.finished(task, reply.text)

    async def _fail(self, task: Task, error: str) -> None:
        if await self.tasks.fail(task.id, self.id, error):
            await self.reporter.failed(task, error)

    async def _retry_or_fail(self, task: Task, error: str) -> None:
        if task.attempts >= task.max_attempts:
            await self._fail(task, f"The model kept failing ({error}). I gave up.")
            return
        delay = self.retry_base * (2 ** (task.attempts - 1))  # 30 s, 60 s, 120 s, ...
        if await self.tasks.retry_later(task.id, self.id, error, delay):
            await self.reporter.retrying(task, error, delay)

    async def run_forever(self, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        log.info("worker %s ready", self.id)
        idle_rounds = 0
        while not stop.is_set():
            worked = await self.run_one()
            if worked:
                idle_rounds = 0
                continue
            idle_rounds += 1
            # About once a minute when idle, give up on tasks that kill every worker.
            if idle_rounds % 30 == 1 and await self.tasks.fail_abandoned():
                log.warning("marked abandoned tasks as failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), self.poll_seconds)


def build_worker(settings, services, tasks: TaskStore, reporter: TaskReporter | None = None):
    """The worker as it runs in production: the full agent, with task limits."""
    from fuse.agent.factory import build_runner
    from fuse.agent.prompts import TASK_SYSTEM_PROMPT

    runner = build_runner(
        settings,
        services.checkpointer,
        services.actions,
        services.google,
        services.drafts,
        services.memory,
        system_prompt=TASK_SYSTEM_PROMPT,
        max_steps=settings.task_max_steps,
    )
    return Worker(
        tasks,
        runner,
        reporter,
        timeout=timedelta(minutes=settings.task_timeout_minutes),
    )


async def _main() -> None:
    from fuse.config import get_settings
    from fuse.services import open_services
    from fuse.tasks.queue import PostgresTaskStore
    from fuse.tracing import enable_tracing

    settings = get_settings()
    enable_tracing(settings)
    async with open_services(settings) as services:
        tasks = PostgresTaskStore(services.pool)
        await tasks.ensure_schema()
        worker = build_worker(settings, services, tasks)
        await worker.run_forever()


def main() -> None:
    from fuse.compat import use_selector_event_loop_on_windows
    from fuse.config import get_settings
    from fuse.logging_setup import configure_logging

    use_selector_event_loop_on_windows()
    configure_logging(get_settings().log_level)
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("worker stopped")


if __name__ == "__main__":
    main()
