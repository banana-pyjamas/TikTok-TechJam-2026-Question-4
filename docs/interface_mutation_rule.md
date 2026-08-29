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

`tests/test_contracts.py` asserts the exact field-name set of each frozen
type. Any unapproved edit to `starter/contracts.py` fails that test.
Updating the assertion is only legitimate as step 4 of an approved request.

## Current frozen field sets (CP 0.2)

| Type | Fields |
| --- | --- |
| `SessionState` | `session_id`, `user_profile`, `turn`, `slots`, `evidence`, `provenance` |
| `Context` | `session_id`, `turn`, `user_message`, `state` |
| `Strategy` | `mode`, `routes` |
| `Candidate` | `parent_asin`, `route_sources`, `bm25_score`, `category_score`, `attribute_score`, `metadata` |
| `RankingResult` | `ranked`, `diagnostics` |
