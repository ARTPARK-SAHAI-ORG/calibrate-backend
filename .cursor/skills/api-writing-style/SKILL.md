---
name: api-writing-style
description: House style for writing FastAPI endpoint summaries, descriptions, and Pydantic field/param docs in this backend. Use when adding or editing any route in src/routers/, defining request/response models, or reviewing API documentation for consistency and clarity.
---

# API Writing Style

House style for documenting FastAPI endpoints and their models in this backend
(docstrings + `summary=` + Pydantic `Field(description=...)`).

Goal: every operation reads consistently in `/docs`, `/redoc`, and the generated
public SDK — a short imperative title, one crisp sentence of intent, and every
variable described by **what it is and what it's for** (not its wire format or
internal implementation).

## The five rules

1. **Give every route an explicit `summary`.** Short, imperative, sentence-case,
   verb-first, no trailing period. Without it FastAPI derives an ugly title from
   the function name (`create_persona_endpoint` → "Create Persona Endpoint").
2. **Write the description as the docstring.** One sentence for public API routes;
   one to three for internal/JWT-only routes when load-bearing behavior truly
   needs it. Verb-first. Say what the call *does* — not how it's implemented.
3. **Type and describe every variable.** Path params, query params, and every
   Pydantic model field get a purpose-first `Field(description=...)`. Optional
   fields say what omitting them means. Format/length lives in the schema
   (`min_length`/`max_length`, `examples`) on body/response models — not in prose.
4. **Be concise. Prefer clarity over completeness.** No filler ("This endpoint
   allows you to..."). Say the thing. Use markdown (`**bold**`, backticks) only
   when it earns its place.
5. **Stay consistent across the whole surface.** Same verbs, same phrasings, same
   param names for the same concepts everywhere (see the vocabulary below).

## Public API (`tags=["Public API"]`) — highest bar

These routes feed the auto-generated SDK/CLI (`GET /public-api/openapi.json`).
Hold them to the strictest interpretation of every rule below. Do not add or
remove the `Public API` tag as part of a docs-only pass.

**Current public routes** (as of this skill): `GET /agents`, `POST /agents/resolve`,
`POST /agent-tests/agent/{agent_uuid}/run`, `POST /agent-tests/run`,
`GET /agent-tests/run/{task_id}`.

### Endpoint heading (summary + docstring)

One sentence. What the call does — full stop.

| Route | Summary | Description |
|---|---|---|
| `GET /agents` | List agents | List all agents in your workspace. |
| `POST /agents/resolve` | Resolve agent names to IDs | Resolve agent names to their IDs. |
| `POST /agent-tests/agent/{agent_uuid}/run` | Run agent tests | Run tests for an agent as a background job. |
| `POST /agent-tests/run` | Run agent tests in batch | Run agent tests for every agent in your workspace, or for a selected set. |
| `GET /agent-tests/run/{task_id}` | Get test run status | Get the status and results of a test run. |

**Never put in the endpoint heading:**

- Auth ("Accepts JWT or API key", `get_org_jwt_or_api_key`, Authorizations is its own section)
- Response field names (`` `not_found` ``, `` `skipped` ``, `` `task_id` ``)
- HTTP status codes or error behavior ("404 if…", "400 otherwise")
- Request-param semantics already on the field ("omit `agent_names` to…")
- Internal preconditions or workflow ("connection must be verified", "call verify-connection")
- Implementation backstory ("runs the calibrate LLM command", job types, queue behavior)
- `(non-deleted)` or other DB-filter caveats

That detail belongs on **field descriptions** or in **code comments** — not the heading.

```python
@router.post("/resolve", summary="Resolve agent names to IDs", tags=["Public API"])
async def resolve_agent_names(...):
    """Resolve agent names to their IDs."""
    # `not_found` → ResolveAgentNamesResponse.not_found Field description

@router.post("/run", summary="Run agent tests in batch", tags=["Public API"])
async def run_tests_batch(...):
    """Run agent tests for every agent in your workspace, or for a selected set."""
    # omit-vs-select → BatchRunRequest.agent_names Field description
```

### Field & path param docs (public API)

**Purpose first, scoping second, format never in prose.**

```python
# Path param — purpose + example; NO min_length on path (422 before your 404)
agent_uuid: str = PathParam(
    description="The agent to test. Must be in your workspace.",
    examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
)

# Request body field — omit behavior here, not in endpoint docstring
class BatchRunRequest(BaseModel):
    agent_names: Optional[List[str]] = Field(
        None,
        description="Agents to run. Omit to run every agent in your workspace",
    )

# Response field — name the thing, not the error contract
class ResolveAgentNamesResponse(BaseModel):
    resolved: Dict[str, str] = Field(
        description="Map of name to agent ID for each name that matched"
    )
    not_found: List[str] = Field(
        description="Names with no matching agent in your workspace"
    )
```

### Response models — one shape per API

Don't reuse a generic model when it drags irrelevant fields into the public spec.
Example: agent-test run returns `AgentTestRunCreateResponse` (`task_id` + `status`
only) — not `TaskCreateResponse` (which carries `dataset_id`/`dataset_name` for
STT/TTS eval jobs).

### IDs — read the code before documenting

All entity IDs are `str(uuid.uuid4())` in `db.py` — standard **UUID v4**, 36
characters with hyphens (e.g. `f47ac10b-58cc-4372-a567-0e02b2c3d479`). There is
no 8-char short ID. Examples and `min_length=36`/`max_length=36` on **body/response
models** must match. Never invent `a1b2c3d4`-style placeholders.

### Public API checklist

- [ ] `summary=` — imperative, no period, says **ID** not UUID
- [ ] Docstring — **one sentence**, no fields/errors/auth/implementation
- [ ] Path params — purpose-first `description` + real UUID `examples`; no length constraints
- [ ] Request fields — what it is + what omitting it does
- [ ] Response fields — what each value means; error/shape detail here, not in heading
- [ ] Dedicated `response_model` — no cross-domain fields leaking in
- [ ] Second person ("your workspace"), workspace not org, ID not UUID, API key not sk_/secret
- [ ] Load-bearing context in `# code comment`, not docstring

## Summaries — verb vocabulary

Use one canonical verb per operation shape. Object is singular; use plural only
for list endpoints.

| Shape | Summary | Example |
|---|---|---|
| `GET` collection | `List <plural>` | `List agents` |
| `GET` one | `Get <singular>` | `Get agent` |
| `POST` create | `Create <singular>` | `Create persona` |
| `PUT`/`PATCH` | `Update <singular>` | `Update evaluator` |
| `DELETE` | `Delete <singular>` | `Delete test` |
| link/attach | `Link <x> to <y>` | `Link tool to agent` |
| unlink | `Unlink <x> from <y>` | `Unlink tool from agent` |
| run/launch a job | `Run <thing>` / `Launch <thing>` | `Run agent tests` |
| poll status | `Get <thing> status` | `Get run status` |
| duplicate | `Duplicate <singular>` | `Duplicate agent` |
| bulk write | `Bulk <verb> <plural>` | `Bulk create test cases` |
| reorder | `Reorder <plural>` | `Reorder evaluators` |

Verb choices to keep uniform: use **Get** (not "Fetch"/"Retrieve" in the title),
**List** for collections, **Delete**, **Create** (not "Add"/"New").

## Descriptions (all routes)

Public API: see **Public API** section above — one sentence, strict.

Internal/JWT-only routes may use two or three sentences when the operation has
genuinely non-obvious behavior (one-time return values, irreversibility, soft
delete vs hard delete). Still follow the bans below.

**Banned everywhere in user-facing descriptions:**

- Authentication boilerplate (Authorizations section covers it)
- Internal symbols (`get_org_jwt_or_api_key`, calibrate command names, job types)
- Response field names or nested shapes in the endpoint blurb
- HTTP error narration in the endpoint blurb
- Internal preconditions ("connection must be verified", verify-connection workflow)
- Third-person indirection ("the caller's workspace" → "your workspace")

```python
@router.delete("/{api_key_id}", summary="Delete API key")
async def delete_api_key(...):
    """Permanently delete an API key. This action cannot be undone."""

@router.get("", response_model=list[AgentResponse], summary="List agents", tags=["Public API"])
async def list_agents(...):
    """List all agents in your workspace."""
```

## Path & query params

```python
from fastapi import Path, Query

agent_uuid: str = Path(
    description="The agent to test. Must be in your workspace.",
    examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
)
limit: int = Query(50, ge=1, le=1_000_000, description="Maximum number of results to return")
q: str | None = Query(None, description="Case-insensitive substring filter on name")
```

Standard reuse:
- `limit` → "Maximum number of results to return"
- `offset` → "Number of results to skip"
- `q` → "Case-insensitive substring filter on `<field>`"
- Resource IDs in path → purpose-first + `examples`; **no** `min_length`/`max_length` on path params

## Pydantic model fields

Every field gets `Field(description=...)`. Required fields have no default;
optional fields default to `None` and their description says what omission means.

```python
class AgentCreate(BaseModel):
    name: str = Field(description="Human-readable agent name, unique within the workspace")
    type: Literal["agent", "connection"] = Field(
        "agent",
        description="`agent` applies managed defaults; `connection` stores the config you supply as-is",
    )
    config: dict[str, Any] | None = Field(
        None, description="Behavioral config. Deep-merged over defaults for `type=agent`; omit to use defaults"
    )
```

Field conventions:
- **Lead with what the thing is and what it's for** — not its format
- Ownership/scoping in a second sentence when relevant (`Must be in your workspace.`)
- `min_length=36`/`max_length=36` + `examples` on **body/response models only**
- Mark conditional requirements in **bold**: `**Required for type=connection.**`
- **Never re-list an enum's own values in its description** (see next section)
- Response fields (`task_id`, `status`, `uuid`) get purpose-first descriptions — they surface in the SDK

## Enums / `Literal` — never re-list the values

The docs renderer (Mintlify) already renders every allowed value of a `Literal`
or enum field from the schema — it shows an `enum<string>` type badge **plus** an
auto-generated **"Available options: …"** line listing them all. Restating those
values in the `description` is pure duplication: it desyncs the moment the enum
changes, and a *partial* re-list (e.g. "status: `queued` or `in_progress`" above a
five-value options line) actively misleads.

**Rule:** the description states the field's **purpose**, never the value set.

```python
# BAD — re-lists what the renderer already shows
status: TaskStatus = Field(description="Current status: `queued`, `in_progress`, `done`, or `failed`")
kind: EvaluatorKindLiteral = Field(description="`single` or `side_by_side`")

# GOOD — purpose only; the renderer lists the values
status: TaskStatus = Field(description="Current status of the test run")
kind: EvaluatorKindLiteral = Field(description="Scoring mode: single output vs. side-by-side comparison")
```

**The one exception — explaining what each value *means*.** If a value needs a
gloss the renderer can't derive, write the gloss (real words per value), never a
bare restatement. This reads as prose, not a value list:

```python
# GOOD — each value carries meaning the schema can't
type: Literal["agent", "connection"] = Field(
    description="`agent` applies managed defaults; `connection` stores the config you supply as-is",
)
```

**Narrow the type instead of narrowing in prose.** When a field can only hold a
subset at a given point in the lifecycle, encode that in the type so the renderer
lists exactly the reachable values — don't type the full enum and then caveat it
in prose. Create-response `status` can only be `queued`/`in_progress`, so it uses
`InitialTaskStatus` (a two-value `Literal` in [utils.py](../../../src/utils.py)),
not the full `TaskStatus`.

This is mechanically enforced: `check_api_docs_style.py` flags any field/param
whose annotation is an enum/`Literal` (inline, a shared alias, or an enum class)
and whose description contains a run of ≥2 of that type's own backticked values
separated only by connectors.

## Terminology (user-facing docs)

| Say | Never (in prose) | Exempt (code identifiers) |
|---|---|---|
| **ID** | UUID | `agent_uuid`, `test_uuids`, `{agent_uuid}` |
| **workspace** | org, organization | `org_uuid`, `get_current_org`, `/org-limits` |
| **API key** | sk_…, secret | — |
| **your** / **you** | the caller, the caller's workspace | "caller" in code comments = calling function |

## Non-negotiables specific to this repo

- **Docs-only edits** must not change path, method, function name, `response_model`,
  or `tags` unless deliberately shipping a new public surface (which needs overlay
  updates — see CLAUDE.md). Touch only `summary=`, docstrings, and `Field`/`Path`/
  `Query` descriptions.
- **Public API** routes: strictest bar (see dedicated section). Internal routes
  (e.g. `verify-connection`, `create_agent`) may keep multi-sentence docstrings
  for load-bearing behavior — but still ban auth/internal-symbol leakage.
- When a public route's `response_model` would inherit irrelevant fields from a
  shared model, create a dedicated response type for that route.

## Checklist per endpoint

- [ ] `summary=` present, imperative, sentence-case, no period
- [ ] Docstring: verb-first; one sentence if `tags=["Public API"]`
- [ ] Every path/query param has a purpose-first description (+ `examples` for IDs)
- [ ] Every request/response model field has `Field(description=...)`
- [ ] No enum/`Literal` field re-lists its own values in the description (state purpose; narrow the type for lifecycle subsets)
- [ ] Optional fields explain what omission does
- [ ] No UUID/sk_/org/caller in prose; second person throughout
- [ ] No path/method/name/tags change (response_model change OK when dedicating a shape)
