"""Minimal in-memory fake of the supabase-py query builder — enough to
exercise real tool logic (filtering, insert, update) against an in-memory
table dict, instead of mocking the tool function itself away. Only
supports the subset of the query builder this codebase actually uses."""

import uuid
from typing import Any


class FakeUniqueViolation(Exception):
    def __str__(self):
        return "duplicate key value violates unique constraint (23505)"


class _FakeResult:
    def __init__(self, data: list[dict[str, Any]]):
        self.data = data


class _FakeQuery:
    def __init__(self, store: dict[str, list[dict[str, Any]]], table_name: str):
        self._store = store
        self._table_name = table_name
        self._filters: list[tuple[str, str, Any]] = []
        self._op: str | None = None
        self._payload: Any = None
        self._limit: int | None = None
        self._select_args: tuple = ()

    def select(self, *args, **_kwargs):
        self._op = self._op or "select"
        self._select_args = args
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, col, val):
        self._filters.append((col, "eq", val))
        return self

    def neq(self, col, val):
        self._filters.append((col, "neq", val))
        return self

    def gte(self, col, val):
        self._filters.append((col, "gte", val))
        return self

    def lte(self, col, val):
        self._filters.append((col, "lte", val))
        return self

    def in_(self, col, vals):
        self._filters.append((col, "in", vals))
        return self

    def ilike(self, col, pattern):
        self._filters.append((col, "ilike", pattern))
        return self

    def or_(self, expr):
        # expr like "full_name.ilike.%x%,phone_number.ilike.%x%" -- real
        # postgrest OR-filter syntax. Parsed into a list of (col, op, val)
        # tuples matched with OR semantics (any one matching is enough).
        conditions = []
        for clause in expr.split(","):
            col, op, val = clause.split(".", 2)
            conditions.append((col, op, val))
        self._filters.append((None, "or", conditions))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def _matches(self, row: dict[str, Any]) -> bool:
        for col, op, val in self._filters:
            row_val = row.get(col)
            if op == "eq" and row_val != val:
                return False
            if op == "neq" and row_val == val:
                return False
            if op == "gte" and (row_val is None or row_val < val):
                return False
            if op == "lte" and (row_val is None or row_val > val):
                return False
            if op == "in" and row_val not in val:
                return False
            if op == "ilike":
                needle = val.strip("%").lower()
                if not row_val or needle not in str(row_val).lower():
                    return False
            if op == "or":
                if not any(self._matches_one(row, c, o, v) for c, o, v in val):
                    return False
        return True

    def _matches_one(self, row: dict[str, Any], col: str, op: str, val: str) -> bool:
        row_val = row.get(col)
        if op == "ilike":
            needle = val.strip("%").lower()
            return bool(row_val) and needle in str(row_val).lower()
        if op == "eq":
            return row_val == val
        return False

    def execute(self):
        table = self._store.setdefault(self._table_name, [])

        if self._op == "insert":
            if self._table_name in self._store.get("__force_conflict__", set()):
                raise FakeUniqueViolation()
            payloads = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted = []
            for payload in payloads:
                row = {"id": str(uuid.uuid4()), **payload}
                table.append(row)
                inserted.append(row)
            return _FakeResult(inserted)

        if self._op == "update":
            updated = []
            for row in table:
                if self._matches(row):
                    row.update(self._payload)
                    updated.append(row)
            return _FakeResult(updated)

        if self._op == "delete":
            deleted = [row for row in table if self._matches(row)]
            remaining = [row for row in table if not self._matches(row)]
            table[:] = remaining
            return _FakeResult(deleted)

        # select
        matched = [row for row in table if self._matches(row)]
        if self._limit is not None:
            matched = matched[: self._limit]

        select_str = " ".join(str(a) for a in self._select_args)
        if "profiles" in select_str and self._table_name != "profiles":
            profiles_table = self._store.setdefault("profiles", [])
            embedded = []
            for row in matched:
                copy = dict(row)
                match = next((p for p in profiles_table if p.get("id") == row.get("profile_id")), None)
                copy["profiles"] = dict(match) if match else None
                embedded.append(copy)
            matched = embedded

        return _FakeResult(matched)


class FakeSupabaseClient:
    def __init__(self, initial: dict[str, list[dict[str, Any]]] | None = None):
        self._store: dict[str, list[dict[str, Any]]] = {k: list(v) for k, v in (initial or {}).items()}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self._store, name)

    def rows(self, table_name: str) -> list[dict[str, Any]]:
        return self._store.setdefault(table_name, [])

    def force_conflict_on_insert(self, table_name: str) -> None:
        """Simulates a DB trigger/constraint that makes any insert into this
        table raise a unique-violation — e.g. the real pet_members trigger
        that auto-creates the owner row before our own code's insert runs."""
        self._store.setdefault("__force_conflict__", set()).add(table_name)
