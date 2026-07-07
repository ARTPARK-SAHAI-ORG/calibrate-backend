"""Shared list-endpoint primitives: pagination, sort, search.

All three are FastAPI dependencies that handlers depend on; they don't touch
SQL directly — they take an already-loaded list of dicts and return a
filtered/sorted/sliced view. That's a deliberate trade: rebuilding queries
across the many entity types in this app would mean a parser per shape, and
the entity tables currently fit in memory. If a particular endpoint outgrows
that, push the predicate down into SQL inside the handler — these deps still
own the request-parameter parsing.

Each endpoint customizes sort/search via a factory that bakes in the
per-endpoint allowlist of columns:

    SummarySort = make_sort_params(
        sortable=["created_at", "updated_at"], default="created_at",
    )
    SummarySearch = make_search_params(searchable=["payload.name"])

    @router.get(...)
    async def handler(
        ...,
        search: SummarySearch = Depends(),
        sort: SummarySort = Depends(),
        pagination: PaginationParams = Depends(),
    ):
        items = search.apply(items)
        items = sort.apply(items)
        page = items[pagination.offset : pagination.offset + pagination.limit]

The factories return a fresh class per call so FastAPI's dependency cache
treats each endpoint's sort/search as a distinct type. `sortable` is enforced
as a strict allowlist — anything else gets a 422 from FastAPI (no
SQL-injection surface even though sort runs post-fetch in Python).
"""

from typing import Any, Dict, List, Literal, Optional, Type

from fastapi import HTTPException, Query

DEFAULT_LIMIT = 50
# Cap is intentionally very high (1M) so "give me everything" use cases like
# CSV export can pass `limit=<total>` without a multi-request loop on the FE.
# The cap exists only as a guard against pathological values (e.g. integer
# overflow attempts); it is not a per-request payload budget — handlers
# remain responsible for their own size/perf characteristics.
MAX_LIMIT = 1_000_000


class PaginationParams:
    """FastAPI dependency for `?limit=&offset=` query params."""

    def __init__(
        self,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
    ):
        self.limit = limit
        self.offset = offset


def make_sort_params(
    *,
    sortable: List[str],
    default: str,
    default_order: Literal["asc", "desc"] = "desc",
) -> Type:
    """Build a FastAPI `Depends`-compatible sort class for one endpoint.

    `sortable` is the allowlist of column names the endpoint will accept for
    `?sort_by=`. `default` must be a member. The returned class has an
    `apply(items, *, secondary_key="uuid")` method that returns a sorted copy
    using the standard `(sort_value, secondary_key)` key — secondary key
    breaks ties so paging is stable when timestamps collide (sqlite
    `CURRENT_TIMESTAMP` is second-resolution).

    Empty/missing sort values coerce to `""` so they sort to one end
    deterministically rather than raising `TypeError` on `None < str`.
    """
    if default not in sortable:
        raise ValueError(
            f"default sort_by={default!r} must be one of sortable={sortable!r}"
        )
    allowed = list(sortable)  # frozen copy
    description = f"Sort key. One of: {', '.join(allowed)}."

    class SortParams:
        def __init__(
            self,
            sort_by: str = Query(default, description=description),
            order: Literal["asc", "desc"] = Query(
                default_order, description="Sort direction."
            ),
        ):
            if sort_by not in allowed:
                # FastAPI auto-422s on Literal mismatch — we can't put a
                # dynamic Literal on the annotation, so validate manually.
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"sort_by={sort_by!r} not allowed; expected one of "
                        f"{allowed!r}"
                    ),
                )
            self.sort_by = sort_by
            self.order = order

        def apply(
            self,
            items: List[Dict[str, Any]],
            *,
            secondary_key: str = "uuid",
        ) -> List[Dict[str, Any]]:
            reverse = self.order == "desc"
            return sorted(
                items,
                key=lambda it: (
                    it.get(self.sort_by) or "",
                    it.get(secondary_key) or "",
                ),
                reverse=reverse,
            )

    SortParams.__name__ = f"SortParams[{'|'.join(allowed)}]"
    return SortParams


def make_search_params(*, searchable: List[str]) -> Type:
    """Build a FastAPI `Depends`-compatible search class for one endpoint.

    `searchable` is the list of dotted paths the search will match against
    (e.g. `"payload.name"` for a nested JSON field, `"name"` for a top-level
    column). The query parameter is always `?q=` — standardize across the
    codebase rather than letting each endpoint pick its own name.

    `q` is case-insensitive substring; empty/whitespace-only `q` is a no-op
    so a FE search-input binding doesn't have to special-case the cleared
    state. Returns `apply(items) -> filtered list`.
    """
    if not searchable:
        raise ValueError("searchable must be non-empty")
    paths = [p.split(".") for p in searchable]
    description = (
        f"Case-insensitive substring search over: {', '.join(searchable)}. "
        "Empty/blank value is a no-op."
    )

    class SearchParams:
        def __init__(
            self,
            q: Optional[str] = Query(None, description=description),
        ):
            self.q: Optional[str] = (
                q.strip().lower() if isinstance(q, str) and q.strip() else None
            )

        def apply(
            self, items: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            if self.q is None:
                return items
            needle = self.q
            return [it for it in items if _matches(it, paths, needle)]

    SearchParams.__name__ = f"SearchParams[{'|'.join(searchable)}]"
    return SearchParams


def _matches(item: Dict[str, Any], paths: List[List[str]], needle: str) -> bool:
    for path in paths:
        value = _get_path(item, path)
        if isinstance(value, str) and needle in value.lower():
            return True
    return False


def _get_path(item: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = item
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "PaginationParams",
    "make_sort_params",
    "make_search_params",
]
