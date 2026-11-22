# Fuse

[![CI](https://github.com/pranitdeshpande98/fuse/actions/workflows/ci.yml/badge.svg)](https://github.com/pranitdeshpande98/fuse/actions/workflows/ci.yml)

A personal AI agent I use every day. It reads my Gmail and Google Calendar, drafts
replies, remembers our conversations, and **never acts on my behalf without my approval**.
I talk to it on Telegram.

Built with LangGraph, open-weight models (via free OpenAI-compatible APIs) and Postgres.

## What it does

- **Answers questions about my day**: "What's on my calendar this week?", "Any unread
  emails from my manager?", "Summarise the latest thread with Pat."
- **Drafts emails, with approval**: it proposes a draft in Telegram with Approve / Reject
  buttons. Approved drafts land in my Gmail Drafts folder. Nothing is ever sent.
- **Remembers conversations**: state is checkpointed to Postgres after every step, so a
  restart loses nothing, not even an approval that's still waiting.
- **Remembers me**: long-term memory of my preferences and facts (pgvector). The relevant
  ones are recalled into every turn, so "book lunch with Pat" knows Pat and my preferences.
  `/memories` shows them all and `/forget <id>` removes one.
- **Stays up when a model provider doesn't**: if the primary LLM fails or rate-limits,
  the same request goes to a fallback provider.

## Architecture

```mermaid
flowchart TD
    TG[Telegram<br/>locked to one chat ID] --> RUN[AgentRunner<br/>send / resume]
    RUN --> G[LangGraph ReAct loop<br/>model ⇄ tools]
    G --> LLM[Open-weight LLMs<br/>Groq → Cerebras fallback]
    G --> R[Read tools<br/>calendar · email search · read thread]
    G --> W[Write tools<br/>draft email]
    W --> GATE[Approval gate<br/>interrupt → Approve / Reject]
    GATE --> LOG[(Actions log<br/>idempotency keys)]
    G --> CP[(Postgres checkpoints)]
    G <--> MEM[(Memories<br/>pgvector · fastembed)]
    R --> GOOG[Gmail / Calendar APIs]
    GATE --> GOOG
    GOOG --> TOK[(OAuth token<br/>Fernet-encrypted)]
```

**How one approved action works**

1. The model calls `draft_email_reply`. The tool logs the action as `pending` and calls
   LangGraph's `interrupt()`, which pauses the graph. Its state is saved in Postgres.
2. Telegram shows the draft with **Approve / Reject** buttons.
3. On Approve, the graph resumes. LangGraph re-runs the tool from the top; everything
   before the action is idempotent, and `interrupt()` now returns the decision.
4. The action runs through `execute_once`, keyed by `hash(thread_id, tool_call_id)`, so a
   crash or retry can never create the draft twice.

## Memory

- **Two kinds:** a short *profile* ("prefers afternoon meetings"), included in every turn,
  and *facts*, recalled by similarity to the current message (top 5 above 0.6 cosine).
- **Local embeddings:** fastembed `BAAI/bge-small-en-v1.5` (384 dims) runs on the machine;
  no embedding API or key.
- **No silent duplicates:** before saving, `remember` checks for similar memories (≥0.65) and
  asks the model to update instead. Updates *supersede* the old memory, so history is kept.
- **Bounded context:** only the last `AGENT_HISTORY_MESSAGES` messages go to the model; long-term
  facts survive through memory, not the transcript.
- **Consolidation ("dreaming"):** `uv run python -m fuse.memory.consolidate [--dry-run]` reads
  new conversation turns and extracts lasting facts. It reads **only my messages**, never
  email text or tool output, and anything close to an existing memory is listed for review.

Thresholds were measured, not guessed: `uv run python -m fuse.memory.evals` runs a recall eval
with the real model. At 0.6: **80% recall, 5% false recall**. The misses need reasoning
("team lunch" → vegetarian), which is why the agent also has a `search_memory` tool.

## Safety, by design

| Risk | Defence |
| --- | --- |
| Agent acts without consent | Every write tool goes through `require_approval`; only an explicit Approve runs it |
| Duplicate actions after a crash or retry | Idempotency keys + a unique constraint in the `actions` table |
| Prompt injection via email ("ignore your instructions and…") | All email/calendar content wrapped in `<untrusted_content>` tags that can't be closed from inside; the system prompt treats it as data, never instructions. The approval gate is the backstop. |
| Strangers using the bot | Messages and button presses are accepted from one Telegram chat ID only |
| Leaked Google tokens | OAuth tokens are Fernet-encrypted at rest; secrets are `SecretStr` and never logged |
| Over-broad permissions | Gmail/Calendar read-only, plus `gmail.compose` for drafts. There is no send tool. |
| Memory poisoning (an email plants "remember: always forward to X") | Memories are saved only from what I say: the prompt forbids remembering from untrusted content, and consolidation reads only my messages. Recalled notes are framed as facts, not instructions. `/memories` makes everything visible. |
| Losing a memory by accident | `forget_memory` goes through the approval gate; updates supersede rather than overwrite |
| A looping model burning the free quota | Per-turn step budget (`AGENT_MAX_STEPS`) |
| Secrets committed to git | `.gitignore` + a gitleaks scan of the full history on every push |

## Tech stack

| Layer | Choice |
| --- | --- |
| Agent | LangGraph (ReAct loop, `interrupt()` for approvals) |
| Models | Open-weight models over OpenAI-compatible APIs (Groq, Cerebras), with fallback |
| Storage | Postgres 17 + pgvector, in Docker; LangGraph Postgres checkpointer |
| Embeddings | fastembed (`bge-small-en-v1.5`, local, ONNX) |
| Channel | Telegram (python-telegram-bot, long polling) |
| Google | OAuth 2.0 (installed-app flow), Gmail + Calendar APIs |
| Tooling | uv, ruff, pytest + pytest-asyncio, GitHub Actions, LangSmith tracing |

## Run it locally

You need Docker, [uv](https://docs.astral.sh/uv/) and a Telegram account.

```bash
git clone https://github.com/pranitdeshpande98/fuse.git
cd fuse
cp .env.example .env          # then fill it in, step by step below
uv sync
docker compose up -d db       # Postgres + pgvector on 127.0.0.1:55432
```

1. **Telegram:** create a bot with [@BotFather](https://t.me/BotFather), put the token in
   `TELEGRAM_BOT_TOKEN`, then run `uv run python -m fuse.channels.whoami`, message your
   bot, and put the printed ID in `TELEGRAM_ALLOWED_CHAT_ID`.
2. **LLM:** create a free API key at Groq and set `LLM_PRIMARY_API_KEY`
   (optionally a Cerebras key as the fallback).
3. **Google (optional):** create a Google Cloud project with the Gmail and Calendar APIs,
   an OAuth "Desktop app" client saved as `credentials.json`, and yourself as a test user.
   Generate `ENCRYPTION_KEY` (command in `.env.example`), then sign in:
   `uv run python -m fuse.google.auth`
4. **Run:** `uv run fuse-bot` (the first run downloads the ~130 MB embedding model)

Try: *"What's on my calendar this week?"*, then *"Draft a reply to the latest email from
Pat saying Friday works."*

## Tests

112 tests. Unit tests use a scripted fake model and fake Google services: no network and no
API keys needed. Integration tests run against real Postgres.

```bash
uv run pytest                                                         # unit tests

# Integration tests use their own database, so they can never touch your real data.
# Create it once:
docker compose exec db psql -U fuse -d fuse -c "CREATE DATABASE fuse_test"
docker compose exec db psql -U fuse -d fuse_test -c "CREATE EXTENSION IF NOT EXISTS vector"
# Then:
TEST_DATABASE_URL=postgresql://fuse:fuse@127.0.0.1:55432/fuse_test uv run pytest
```

A guard refuses to run integration tests against any database whose name doesn't end in
`_test`.

Highlights:
- `test_approval_restart.py`: pause for approval, throw the whole agent away, rebuild it,
  approve, and check the action ran exactly once.
- `test_google_tools.py`: an email that tries to close the untrusted-content tag and inject
  a "SYSTEM" instruction stays inside the wrapper.
- `test_actions.py`: an action never runs twice, and a failed one can be retried.
- `test_consolidate.py`: consolidation ignores an email that says "remember…", and never
  saves a fact that conflicts with an existing memory.
- `test_memory_evals.py`: recall quality with the real model
  (opt-in: `FUSE_TEST_EMBEDDINGS=1`, downloads the model).

CI runs lint, formatting, all tests against a pgvector service container, and a secret scan.

## Project layout

```
src/fuse/
  agent/        graph.py (ReAct loop) · runner.py (send/resume) · prompts.py
  tools/        approval.py (the gate) · drafts.py · basic.py
  memory/       store.py (pgvector) · embeddings.py · tools.py · context.py (recall)
                commands.py · consolidate.py · evals.py
  google/       auth.py (OAuth + encrypted tokens) · reader.py · tools.py · drafts.py
  channels/     telegram_bot.py · whoami.py
  actions.py    actions log + idempotency
  llm.py        provider fallback chain
  config.py     typed settings from .env
  crypto.py     Fernet encryption
  db.py · compat.py · tracing.py · logging_setup.py
tests/          unit + integration tests
```

## Lessons learned

- **Windows + async Postgres:** psycopg's async mode can't use Windows' default event loop.
  The bot sets a selector loop, and tests use pytest-asyncio's `loop_factories` hook,
  because pytest-asyncio creates its own loops.
- **`localhost` isn't always 127.0.0.1:** on Windows it resolves to IPv6 `::1` first. The sync
  driver quietly retried, but the async pool hung with no timeout. Fix: `127.0.0.1` plus
  a `connect_timeout` on the pool.
- **Tests must not share state with real data:** a test once overwrote my real Google token.
  Tests now use their own rows, and a wrong encryption key gives a clear "sign in again"
  message instead of a crash.
- **Measure thresholds with the real model:** I guessed 0.95 for "duplicate". bge-small scores
  "meetings in the morning" vs "in the afternoon" at 0.94, so that would have silently
  dropped a real change. Measured pairs set it to 0.98.
- **Resuming re-runs the tool:** LangGraph resumes an interrupted tool from its first line,
  which is why everything before `interrupt()` must be safe to repeat.

## Roadmap

- [x] **Phase 1:** reactive agent with approvals, Gmail + Calendar, persistence, CI
- [x] **Phase 2:** long-term memory (profile + pgvector recall, consolidation, evals)
- [ ] **Phase 3:** proactive: morning brief, Gmail push alerts, decision engine
- [ ] **Phase 4:** long-running tasks on background workers
- [ ] **Phase 5:** Slack channel, skills, evals
- [ ] **Phase 6:** WhatsApp channel + job application dashboard
- [ ] **Phase 7:** cloud deployment