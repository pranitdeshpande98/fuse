"""Builds the agent. Shared by the Telegram bot and the background worker, so both get the
same tools and the same safety rules."""

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from fuse.actions import ActionStore
from fuse.agent.graph import build_graph
from fuse.agent.prompts import SYSTEM_PROMPT
from fuse.agent.runner import AgentRunner
from fuse.config import Settings
from fuse.contacts import ContactStore
from fuse.google.reader import GoogleReader
from fuse.google.tools import make_google_tools
from fuse.jobalerts.client import JobAlertsClient
from fuse.jobalerts.tools import make_job_alerts_tools
from fuse.jobs.store import JobStore
from fuse.jobs.tools import make_job_tools
from fuse.llm import build_llm
from fuse.memory.store import MemoryStore
from fuse.memory.tools import make_memory_tools
from fuse.proactive.schedule import SystemClock
from fuse.reminders.service import make_reminder_tools
from fuse.reminders.store import ReminderStore
from fuse.skills.loader import Skill, skills_prompt
from fuse.skills.tools import make_skill_tools
from fuse.tools.basic import make_basic_tools
from fuse.tools.drafts import DraftSink, LocalDraftSink, make_draft_tools
from fuse.tools.web import make_web_tools


def build_tools(
    settings: Settings,
    actions: ActionStore,
    google: GoogleReader | None = None,
    drafts: DraftSink | None = None,
    memory: MemoryStore | None = None,
    contacts: ContactStore | None = None,
    skills: list[Skill] | None = None,
    jobs: JobStore | None = None,
    job_alerts: JobAlertsClient | None = None,
    reminders: ReminderStore | None = None,
) -> list[BaseTool]:
    return [
        *make_basic_tools(settings.user_timezone),
        *(make_google_tools(google) if google else []),
        *make_draft_tools(actions, drafts or LocalDraftSink(), contacts, google),
        *(make_memory_tools(memory, actions) if memory else []),
        *(make_web_tools() if settings.web_search_enabled else []),
        *(make_skill_tools(skills) if skills else []),
        *(make_job_tools(jobs) if jobs else []),
        *(make_job_alerts_tools(job_alerts) if job_alerts else []),
        *(make_reminder_tools(reminders, SystemClock(settings.user_timezone)) if reminders else []),
    ]


def build_runner(
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    actions: ActionStore,
    google: GoogleReader | None = None,
    drafts: DraftSink | None = None,
    memory: MemoryStore | None = None,
    extra_tools: Sequence[BaseTool] = (),
    system_prompt: str = SYSTEM_PROMPT,
    max_steps: int | None = None,
    contacts: ContactStore | None = None,
    skills: list[Skill] | None = None,
    jobs: JobStore | None = None,
    job_alerts: JobAlertsClient | None = None,
    reminders: ReminderStore | None = None,
) -> AgentRunner:
    tools = [
        *build_tools(
            settings,
            actions,
            google,
            drafts,
            memory,
            contacts,
            skills,
            jobs,
            job_alerts,
            reminders,
        ),
        *extra_tools,
    ]
    system_prompt += skills_prompt(skills or [])
    model = build_llm(settings, tools)
    graph = build_graph(
        model,
        tools,
        checkpointer,
        memory,
        max_history_messages=settings.agent_history_messages,
        max_history_tokens=settings.agent_history_tokens,
        system_prompt=system_prompt,
    )
    return AgentRunner(graph, max_steps=max_steps or settings.agent_max_steps)
