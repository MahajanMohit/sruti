# Skills

A skill writes down the steps for a task so the model does not have to plan one.

## Why they exist

The load-bearing constraint of this project is that a 1–2B model cannot hold a
multi-step plan. It can answer one narrow question well — *what filename does
this sentence refer to?* — and it cannot reliably decide what to do, in what
order, using which tools.

Measured on Qwen2.5-0.5B (`tools/verify_grammars.sh`):

| Question | Result |
|---|---|
| Which tool applies? | valid but **wrong** (`get_time` for a file read) |
| What are the arguments? | **correct** (`{"path":"/tmp/notes.txt","max_lines":10}`) |
| Which enum value? | **correct** (`{"mode":"quality"}`) |

That split is the whole design. Argument extraction works; tool selection does
not. A skill removes the part that does not work: it names the tools and their
order, and the model is asked only the question it is good at.

## Format

Markdown with a frontmatter block. One file, shareable as-is.

```markdown
---
name: summarise-file
description: Read a file and write a summary beside it
keywords: summarise, digest, condense
parameters:
  - path: The file to summarise
  - style?: How terse to be
steps:
  - read_file(path={{path}})
  - write_file(path={{path}}.summary.md, content={{step1}})
---

Keep facts and numbers; drop repetition.
```

| Field | Meaning |
|---|---|
| `name` | Required. Also the filename on disk. |
| `description` | Required. One line; shown in the list and given to the model. |
| `keywords` | Words that appear in requests this skill should handle. |
| `parameters` | Slots the model fills. A trailing `?` makes one optional. |
| `steps` | Tool calls, in order. Omit for a guidance-only skill. |
| body | Prose passed to the model as extra instruction. |

### Slots

- `{{name}}` — a parameter, filled by the model from the request.
- `{{stepN}}` — the output of step *N*, filled by the runner.

A slot with no value stops the skill before the step runs, rather than passing
the literal text `{{path}}` to a tool.

### Commas

A comma ends an argument only when what follows it starts another one
(`name=`). So this works without quoting:

```
- run_shell(command=git log --pretty=format:%h,%s -n 3)
```

Quote a value if it genuinely contains something shaped like `, name=`.

## The two kinds

**Templated** — has `steps`. Runs deterministically. The model is asked exactly
one constrained question (fill these slots) and everything else is code. This is
the reliable kind.

**Guidance only** — no `steps`. The body is prepended to the ordinary agent
loop, which still asks the model to choose tools. Weaker, but it takes a
sentence to write.

## How one gets chosen

Keyword scoring against the request, not a model call — choosing between skills
is the same question as choosing between tools, which is the one measured to be
unreliable. A skill must score at least 4 to be used; below that the request
falls through to the ordinary agent loop.

Scoring matches the tool registry: name term 5, keyword 3, description word 1.

The threshold matters more than the ranking. Running the wrong skill is worse
than running none, because a skill acts without asking which tool to use.

## Safety

A skill does not bypass anything. Every tool it names still declares its own
tier and confirmation requirement, so a `run_shell` step still shows the exact
command and waits for approval, and a skill that needs the shell will not run at
all while the shell is off — checked before the first step, so a skill never
stops halfway leaving the workspace in a state nobody asked for.

## Importing

- **File** — pick a `.md` file. It is parsed before it is written, so a file on
  disk always loads.
- **URL** — a direct link to the file itself (a gist raw URL, a repository raw
  link).

Skills live in `filesDir/skills/` as plain files.

## Built-ins

`summarise-file`, `save-page`, `note-clipboard`, `git-status`. They are written
out in full in `SkillStore` rather than constructed in code, because they double
as the worked examples of the format.
