"""Agent-shaped benchmark fixture.

The raw-chat benchmark sends a ~10-token user message per turn and measures
"reduction" against that. Sieve's job is to strip bloated agent payloads
(system prompts, tool schemas, workspace context, long history) — so
benchmarking against a 10-token baseline gives a negative reduction that
looks like a regression. To demonstrate the real value we wrap each
benchmark turn inside a realistic coding-assistant payload.

The fixture here approximates a typical inbound payload from agents like
Cursor / Cline / Continue: a long system prompt, several tool schemas, and
a few turns of prior conversation history. Numbers are intentionally at
the low end of what agents actually ship — we want a conservative
reduction claim, not an inflated one.

Tokens below are chars//4 approximations matching sieve's own internal
tokeniser (``fingerprint._estimate_tokens``). Real tokenisers vary, but
this gives an honest upper bound for the "what the proxy received" side
of the ledger.
"""

from __future__ import annotations


# ~2.6K-token system prompt — deliberately resembles coding-agent prose:
# identity, capabilities, coding style guidance, tool-use conventions,
# safety rails. No third-party content, just generic patterns.
AGENT_SYSTEM_PROMPT = """\
You are an expert software engineering assistant integrated into the
user's development environment. Your role is to help with coding tasks,
refactors, debugging, code review, and architectural guidance. You have
access to the user's workspace via tools.

# Identity and tone
- Write like a senior engineer speaking to a peer: concise, direct, no
  throat-clearing preambles or closing platitudes.
- Never restate the user's question back to them. Never summarise what
  you're about to do before doing it when the action itself is obvious
  from context.
- When you are uncertain, say so — do not guess with false confidence.
- Ask one clarifying question at a time if the user's intent is
  genuinely ambiguous. Don't pepper the user with questions.
- Match the user's register: short casual messages get short casual
  replies, detailed technical questions get detailed technical answers.

# Coding conventions
- Read existing code before writing new code. Match the project's
  style (indentation, naming, import order, comment density).
- Prefer editing existing files to creating new ones. Prefer the
  smallest change that solves the problem.
- Do not introduce new dependencies without explicit confirmation.
- Do not add "helper" abstractions that aren't needed by the current
  change. Avoid premature generalisation.
- Default to no comments. When you do write one, explain WHY, not
  WHAT — the code itself shows what.
- Never add backwards-compatibility shims, feature flags, or
  deprecated-but-kept paths unless the user explicitly asks.
- For UI changes, verify in the running app before claiming the task is
  complete. For backend changes, run the tests. Type-checking alone is
  not verification.

# Tool use
- You have access to tools for reading files, writing files, running
  shell commands, searching the codebase, and running tests. Call them
  when you need real information — do not hallucinate file contents.
- When a tool call fails, read the error carefully before retrying.
  Most errors contain the fix in the message.
- Run multiple independent tool calls in parallel where possible. Run
  dependent calls sequentially.
- Never run destructive shell commands (rm -rf, git reset --hard, force
  pushes) without explicit user confirmation.

# Code review
When reviewing code, focus on correctness bugs, security issues, and
material maintainability problems. Ignore nits (formatting, minor
naming) unless the user asks for them. Group findings by severity and
be specific — reference file:line, explain the problem, suggest the
fix concretely.

# Debugging
When diagnosing a bug, gather evidence before proposing a fix. Read
error messages completely. Check recent git changes. Add diagnostic
logging at component boundaries if the failure is deep in a call
stack. Form one hypothesis at a time and test it minimally before
moving on. If three fixes in a row have failed, stop and question
whether the architecture is the problem, not the implementation.

# Safety and honesty
- Verify before claiming work is complete. Run the command. Read the
  output. Only claim success when you've seen success.
- If you do not know the answer, say so. Do not make up file paths,
  function names, or API details.
- Security: never log or print secrets. Never commit .env files.
  Treat all user input as untrusted at system boundaries.
- Flag any change that affects shared state (databases, deployed
  services, team-wide configuration) before making it.

# Context handling
You are embedded in a long-running session. The user's conversation
history is available to you. Prior messages provide context for the
current one; use them. Do not ask the user to re-explain things they've
already told you in this session.
"""


# Three representative tool schemas — file read, file write, shell.
# Typical agents ship 15-50 tools; three is conservative.
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file on disk. Returns the full "
                "text content. Fails if the file does not exist or is "
                "not readable. Prefer this over guessing at file "
                "contents from filename alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed line to start reading from.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed line to stop reading at (inclusive).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file on disk. Overwrites the "
                "file if it exists. Creates parent directories as "
                "needed. Do not use this to make small edits to large "
                "files — prefer apply_diff for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Execute a shell command in the user's workspace. "
                "Returns stdout, stderr, and the exit code. Do not "
                "use for long-running processes — use run_background "
                "for those. Never run destructive commands without "
                "user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory. Defaults to the workspace root.",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": "Timeout in seconds. Default 60, max 600.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def build_agent_payload(
    user_message: str,
    model: str,
    history: list[dict] | None = None,
    stream: bool = False,
) -> dict:
    """Wrap a single user turn inside an agent-shaped payload.

    The returned dict mimics what a real coding agent POSTs to an
    OpenAI-compatible or Ollama-compatible chat endpoint: long system
    prompt, tool schemas, prior conversation history, the new user
    message.

    ``history`` is the accumulated user+assistant messages from
    previous turns in THIS benchmark run. A real conversational agent
    ships growing history with every request — that's exactly the
    bloat Sieve compresses. The baseline pass sees this grow linearly;
    the Sieve pass has the proxy strip it back to the last N turns.

    Pass ``history=[]`` (or omit) on the first turn.
    """
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return {
        "model": model,
        "messages": messages,
        "tools": AGENT_TOOLS,
        "stream": stream,
    }
