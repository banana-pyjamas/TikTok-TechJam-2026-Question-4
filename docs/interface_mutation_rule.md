# CP 0.3 — Interface Mutation Rule

## Frozen surface

The dataclasses in `starter/contracts.py` are **frozen contracts**:

- `SessionState`
- `Context`
- `Strategy`
- `Candidate`
- `RankingResult`

"Frozen" means the **field set and field types** are fixed. It does **not**
mean instances are immutable — `SessionState` and `Candidate` are mutated
during a turn by design.

## Rule

Do not add, remove, rename, or retype a field on any frozen type, and do not
add or remove a frozen type, except through this process:

1. **STOP** current checkpoint work.
2. File an **INTERFACE CHANGE REQUEST** (template below) to the project owner.
3. Wait for **explicit approval**. No approval → no change.
4. Apply the approved change in its **own commit**, updating
   `starter/contracts.py`, this document's field list, and the freeze-guard
   test in `tests/test_contracts.py` together.
5. Notify B / C / D that the contract moved.

Adding a brand-new production module that does **not** touch these types is
normal checkpoint work and needs no request.

## INTERFACE CHANGE REQUEST template

```
INTERFACE CHANGE REQUEST

Type:
<SessionState | Context | Strategy | Candidate | RankingResult>

Old:
<current field(s) / signature>

Proposed:
<new field(s) / signature>

Reason:
<why the current checkpoint cannot proceed without it>

Affected modules:
<production files + reviewer workflows that must adapt>
```

## Enforcement

`tests/test_contracts.py::FreezeGuardTest` asserts the exact field-name set,
field order, **and field type string** of each frozen type. Any unapproved
edit to `starter/contracts.py` (rename, reorder, add, remove, or retype a
field) fails that test. Updating the `FROZEN_FIELDS` mapping is only
legitimate as step 4 of an approved request.

## Current frozen fields and types (CP 0.2, revised after B review)

| Type | Field | Type |
| --- | --- | --- |
| `SessionState` | `session_id` | `str` |
| `SessionState` | `user_profile` | `dict[str, Any]` |
| `SessionState` | `turn` | `int` |
| `SessionState` | `slots` | `dict[str, Any]` |
| `SessionState` | `evidence` | `list[Any]` |
| `SessionState` | `provenance` | `list[dict[str, Any]]` |
| `Context` | `session_id` | `str` |
| `Context` | `turn` | `int` |
| `Context` | `user_message` | `str` |
| `Context` | `state` | `SessionState` |
| `Context` | `derived` | `dict[str, Any]` |
| `Strategy` | `mode` | `str` |
| `Strategy` | `routes` | `list[str]` |
| `Strategy` | `route_weights` | `dict[str, float]` |
| `Strategy` | `params` | `dict[str, Any]` |
| `Candidate` | `parent_asin` | `str` |
| `Candidate` | `route_scores` | `dict[str, float]` |
| `Candidate` | `metadata` | `dict[str, Any]` |
| `RankingResult` | `ranked` | `list[Candidate]` |
| `RankingResult` | `diagnostics` | `dict[str, dict[str, Any]]` |

`Candidate.route_sources` is a **derived read-only property** (routes present
in `route_scores`, in insertion order), not a frozen field.

### Extensibility rule (added after B review)

Per-route scores and strategy parameters live in generic containers so
future routes/knobs do NOT need an interface change:

- a new retrieval route adds a **key** to `Candidate.route_scores` and
  `Strategy.route_weights` / `Strategy.routes` — never a field;
- a new strategy parameter goes in `Strategy.params`;
- a new per-turn computed input goes in `Context.derived`;
- new slot-level signals (EC, MR, ...) go inside `SessionState.slots`
  entries — never a top-level field.

## Frozen None rule (CP 0.4)

At **construction time**:

- **Container-typed fields** (`dict`, `list`, and the nested
  `Context.state`): an explicit `None` is normalized to a fresh empty
  container in `__post_init__`.
- **Scalar-typed fields** (`str`, `int`): `None` is **not** coerced.
  Passing `None` is a caller error; the contract neither raises nor masks
  it. Construction still succeeds so a malformed caller cannot crash the
  agent at build time.

Post-construction assignment to `None` is out of scope for this rule.
Enforced by `tests/test_contracts.py::NoneHandlingTest`.
