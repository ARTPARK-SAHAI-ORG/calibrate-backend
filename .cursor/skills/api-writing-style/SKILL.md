---
name: api-writing-style
description: House style for writing FastAPI endpoint summaries, descriptions, and Pydantic field/param docs in this backend. Use when adding or editing any route in src/routers/, defining request/response models, or reviewing API documentation for consistency and clarity.
---

# API Writing Style

House style for documenting FastAPI endpoints and their models in this backend
(docstrings + `summary=` + Pydantic `Field(description=...)`).

Goal: every operation reads consistently in `/docs`, `/redoc`, and the generated
public SDK — a short imperative title, one crisp sentence of intent, non-obvious
behavior called out, and every variable typed and described concisely.

## The five rules

1. **Give every route an explicit `summary`.** Short, imperative, sentence-case,
   verb-first, no trailing period. Without it FastAPI derives an ugly title from
   the function name (`create_persona_endpoint` → "Create Persona Endpoint").
2. **Write the description as the docstring.** One or two sentences. Start with a
   verb. State what it does, then call out anything non-obvious (one-time values,
   irreversible actions, defaults, ordering, side effects, auth mode).
3. **Type and describe every variable.** Path params, query params, and every
   Pydantic model field get a concise description. Include format/shape hints
   (`E.164`, `ULID`, `8-char UUID`, `ISO 8601 UTC`) and mark conditional
   requirements. Optional fields say what omitting them means.
4. **Be concise. Prefer clarity over completeness.** No filler ("This endpoint
   allows you to..."). Say the thing. Use markdown (`**bold**`, backticks, short
   lists) only when it earns its place.
5. **Stay consistent across the whole surface.** Same verbs, same phrasings, same
   param names for the same concepts everywhere (see the vocabulary below).

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

Verb choices to keep uniform: use **Get** (not "Fetch"/"Retrieve" in the title —
those belong in the description), **List** for collections, **Delete** (add
"permanently" in the description if the row is hard-deleted; "Soft-delete ..." in
the description when it's a soft delete), **Create** (not "Add"/"New").

## Descriptions

- First sentence: verb-first statement of intent — `Retrieve a ...`,
  `Create a new ...`, `Soft-delete a ...`, `List the ... for the caller's workspace.`
- Then, only if non-obvious, add short sentences for: one-time-only return values,
  irreversibility, default filters/ordering (`newest first`, `active by default`),
  scoping, required conditions, or auth mode (JWT vs API key).
- Keep it to ~1–3 sentences for CRUD. Complex/job endpoints may use a short list.
- Don't restate the params — describe *behavior*, not the signature.
- **Don't document authentication in the description.** The security scheme
  already renders as an "Authorizations" section, so drop "Accepts a JWT or an
  API key"-style boilerplate.
- **Never leak internal symbols or backstory.** No dependency/function names
  (e.g. `get_org_jwt_or_api_key`), no CI/use-case narrative, no cross-endpoint
  asides. That context belongs in a code comment or a guide — not the public
  description. Keep the blurb about *this* resource and what the call does.

```python
@router.delete("/{api_key_id}", summary="Delete API key")
async def delete_api_key(...):
    """Permanently delete an API key. This action cannot be undone."""
```

```python
@router.get("", response_model=list[AgentResponse], summary="List agents")
async def list_agents(...):
    """List agents for the caller's current workspace. Accepts a JWT or an API key."""
```

## Path & query params

Describe with `fastapi.Path` / `fastapi.Query` (or `Field` on a params model).
Include the shape. Keep it to a phrase.

```python
from fastapi import Path, Query

agent_uuid: str = Path(description="Agent UUID (8-char identifier)")
limit: int = Query(50, ge=1, le=1_000_000, description="Maximum number of results to return")
q: str | None = Query(None, description="Case-insensitive substring filter on name")
```

Standard param descriptions (reuse verbatim where they apply):
- pagination: `limit` → "Maximum number of results to return"; `offset` →
  "Number of results to skip"; `q` → "Case-insensitive substring filter on `<field>`".
- resource ids in path: `"<Entity> UUID (8-char identifier)"`.

## Pydantic model fields

Every field gets `Field(description=...)`. Required fields have no default;
optional fields default to `None` and their description says what omission means.

```python
from pydantic import BaseModel, Field

class AgentCreate(BaseModel):
    name: str = Field(description="Human-readable agent name, unique within the workspace")
    type: Literal["agent", "connection"] = Field(
        "agent",
        description="`agent` applies managed defaults; `connection` stores the caller config as-is",
    )
    config: dict[str, Any] | None = Field(
        None, description="Behavioral config. Deep-merged over defaults for `type=agent`; omit to use defaults"
    )
```

Field description conventions:
- Lead with the meaning, not the type (the type column already shows `string`).
- Add format hints in parentheses or with `e.g.`: `E.164 (e.g., +12345678901)`,
  `ISO 8601 UTC`, `ULID (26 chars)`, `8-char UUID`.
- Mark conditional requirements in **bold**: `**Required for type=connection.**`
- For enums/`Literal`, describe each value briefly with backticks.
- Response-only fields (`uuid`, `created_at`, `masked_key`, `message`) still get a
  short description — they show up in the SDK and docs too.

## Terminology (user-facing docs)

- Say **"workspace"**, never "org" / "organization", in any doc text (summaries,
  docstrings, field/param descriptions). Code identifiers are exempt and stay as-is
  (`org_uuid`, `get_current_org`, `OrgContext`, the `X-Org-UUID` header, the
  `/org-limits` route) — only the prose changes.
- Say **"API key"**, never `sk_…` or "secret". Don't print the key's literal
  prefix; describe headers as `Authorization: Bearer <api-key>` / `X-API-Key`.

## Non-negotiables specific to this repo

- **Never change a route's path, method, function name, `response_model`, or
  `tags`** while restyling docs — docs-only edits must not alter behavior or the
  generated SDK surface. Only touch `summary=`, docstrings, and `Field`/`Path`/
  `Query` descriptions.
- **Public API routes** (`tags=["Public API"]`) feed the auto-generated SDK/CLI.
  Their summaries/descriptions/field docs surface to external users — hold them to
  the highest bar. Do not add or remove the `Public API` tag as part of a docs pass.
- Keep the existing multi-line docstrings that carry load-bearing behavioral notes
  (e.g. agents `verify-connection`, `create_agent`) — tighten wording to this
  style, but don't drop the substance.

## Checklist per endpoint

- [ ] `summary=` present, imperative, sentence-case, no period
- [ ] Docstring: verb-first, states intent + any non-obvious behavior
- [ ] Every path/query param has a description with a shape hint
- [ ] Every request-model field has a `Field(description=...)`
- [ ] Every response-model field has a `Field(description=...)`
- [ ] Optional fields explain what omission does
- [ ] No path/method/name/response_model/tags change
