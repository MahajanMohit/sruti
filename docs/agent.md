# The agent harness

Sruti runs strictly on-device, which means the agent is driven by a 1–2B model
with no cloud fallback. That constraint decides the entire design.

## The honest starting position

A model this size **cannot hold a multi-step plan**. Asked to plan and execute, it
produces something plan-shaped and then loses the thread. Asked to emit JSON, it
produces something JSON-shaped: a trailing comma, an unquoted key, a chatty
preamble before the brace. Prompting harder does not fix either; the capability
is not there.

So the harness does not ask it to. **Intelligence lives in the code; the model
answers one narrow question at a time.**

## Two guarantees, and the gap between them

### Grammars guarantee form

Every constrained call goes through a GBNF grammar generated from the tool's
schema. Tokens that would break the structure are masked during sampling, so a
malformed call is not unlikely — it is unreachable.

Measured on real models via `tools/verify_grammars.sh`:

| Question | Qwen2.5-0.5B | Result |
|---|---|---|
| Which tool? | `"get_time"` | valid, **wrong** |
| Fill `read_file` arguments | `{"path":"/tmp/notes.txt","max_lines":10}` | valid, correct |
| Fill `set_mode` enum | `{"mode":"quality"}` | valid, correct |

That table is the whole design argument. **Selection is unreliable at this size;
slot-filling is not.** So the harness narrows selection with retrieval before the
model ever sees the list, and leans on the model only for the part it is good at.

### Nothing guarantees judgement

A grammar cannot make a choice correct. Three things compensate:

**Retrieval before selection.** The model is shown 3–5 candidates ranked by
lexical overlap, never the whole registry — accuracy collapses as the list grows.
Lexical rather than embeddings: no second model, no extra memory, and predictable
enough to debug. A wrong candidate set is recoverable; a silently mis-ranked
embedding is not.

**Deterministic validation, not a model self-check.** Required arguments present,
types correct, enum values in range, no undeclared keys. The plan originally
called for a constrained self-check pass by the model; that was dropped
deliberately — a model that picked the wrong tool is not the thing to ask whether
it picked the wrong tool, and it would reject good calls as often as bad ones.
Code checks what code can check.

**Confirmation for anything that mutates.** Writes, network fetches and shell
commands are shown verbatim and confirmed. Reads are not gated: requiring approval
for every read trains the user to tap through without reading, which is worse than
not asking.

## The loop

```
retrieve candidates → choose tool (grammar) → extract arguments (grammar)
  → validate → confirm if mutating → execute → observe → answer
```

Capped at a hard step ceiling, with every stage emitted as an `AgentEvent` so the
trace is auditable rather than a black box. Each pass re-derives from the recorded
observations rather than from a plan the model is expected to remember.

A failed tool call ends the run and reports. Retrying the same request against the
same model reliably reproduces the same failure.

## Tools

| Tool | Tier | Confirmed |
|---|---|---|
| `read_file`, `list_files` | in-app | no |
| `write_file` | in-app | yes |
| `fetch_url` | in-app | yes |
| `read_clipboard` | in-app | no |
| `write_clipboard` | in-app | yes |
| `run_shell` | shell | yes |

File access is confined to one directory inside app storage. That is a real
boundary: paths are canonicalised and then checked to be inside it, so `../`
cannot walk out and a sibling directory sharing a name prefix cannot be reached.
Absolute paths are confined rather than honoured. All of this is unit-tested,
because it is the check standing between a hallucinated path and the filesystem.

### The shell

`run_shell` goes through Termux's `RUN_COMMAND` service — the largest capability
jump available without root or an accessibility service, and the reason it earns
its integration cost: **one correct shell command accomplishes what would
otherwise be a ten-step plan the model cannot hold.** Preferring a single command
over a long chain plays directly to the model's strengths.

It is also unbounded. A shell does whatever a shell can do, so it is off until
explicitly enabled and every command is shown before it runs. That confirmation is
the only meaningful control here, not a formality.

Termux additionally requires `allow-external-apps=true` in its own
`~/.termux/termux.properties`, which no Android permission can grant and nothing
here can detect. A failure to run says so rather than blaming the command.

## What this is not

It is not Claude Code. With no cloud escalation the ceiling is **reliable bounded
tasks and shell-composed one-shots**, not open-ended multi-step autonomy. The
harness gets the most out of a 1–2B model; it cannot exceed one.

The measured tool-selection failures above are not a bug to be fixed by tuning.
They are the size of the model, and the design is built around them.
