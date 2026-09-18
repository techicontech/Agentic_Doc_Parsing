# Retrieval and grounding

## Design principles

1. **Evidence first** — the model only sees retrieved **element rows** (paragraph / table / figure), not token-window chunks.
2. **Dynamic ranking** — boosts use query intent and section metadata, never hard-coded question→page maps.
3. **Abstain when unsure** — verification can block an answer if scope, equipment, units, or conflicts fail.
4. **Prefer the right section kind** — numeric specs → `data`; how-to → `procedure`; diagrams when the question asks for a plate/drawing.
5. **Name the family, not the sibling** — contrast clauses (“do not use…”, “not journal-bearing…”) are stripped before identifier pinning so a mentioned *wrong* plate is not treated as the target.

## Query intent

From the user question the retriever derives:

- Content tokens (stopwords removed).
- Phrases (bigrams/trigrams) for component matching (`main bearing`, `exhaust valve`).
- Intent flags — data-like (`clearance`, `torque`), procedure-like (`remove`, `how`), figure-like (`diagram`, `plate`).
- **Named identifiers** from the *positive* clause only: procedure numbers, plate codes, drawing codes, table-row codes, plus procedure implied by drawings such as `KN905-1.1` → `905-1.1`.
- **Negated / contrast tokens** from “do not use”, “not the”, “even if”, “is that a …”.

## Paths

| Path | Mechanism |
|------|-----------|
| Lexical | Postgres FTS + phrase ILIKE + citation-key match |
| Structural | Agent picks sections from the catalog; fetch by `doc_code` / `procedure_no` / title |
| Figures | Panel-level rows from `figures`; suppressed for pure data lookups |

Paths run in parallel. A failing path is logged and skipped.

## Ranking (`ranking.apply_priority`)

Deterministic tool, same rules on every query. Implemented in `src/marine_docs/ranking.py`.

1. **Exact identifier wins.** A procedure, plate, drawing, or `citation_key` named in the *positive* clause is fetched even if no path surfaced it, and is pinned.
2. **Component phrases beat a similar number in another chapter.** Title overlap on distinctive tokens pins the named assembly; weak words (`bearing`, `pressure`, `plate`, …) do not.
3. **Kind pairing.** A data sheet is paired with the checking/mounting procedure for the **same component** (and the reverse). Tables that actually hold the asked values (hydraulic pressure, torque, clearances) are preferred over long tool/safety tables.
4. **Section mates and list siblings** fill out the same sheet once one row scored.
5. **Step text pulls its panel.** `linked_step_number` brings the illustrating figure, including its `figure_image_path`.
6. **One cross-reference hop** along `references_procedure` / `references_data` / `references_section`.
7. **Table-row code hop** (`D13-01` style) stays inside the same component family unless the question is an explicit comparison (“quote both”, “versus”, “same rule”).
8. **Same-chapter sibling demotion.** If the question names `909-6`, `909-4` in the same chapter is unpinned. Complementary *data* sheets in another chapter (e.g. `109-6`) are kept.
9. **Contrast / negation.** Tokens and identifiers in “do not use the cylinder-cover sheet” are not pinned and are dropped from the citation and diagram windows.
10. **Rerank** breaks remaining ties. Pinned hits cannot be demoted (`finalize_order`).
11. **Diversify** so one procedure code cannot fill the whole window; when the query is visual, rows with images are preferred.

RRF (`retrieval.rrf_fuse`) merges the lexical copy and the figure copy of the same `element_id` so the surviving row keeps `figure_image_path`.

## Citation and diagram windows

After ranking, the API does not dump the raw top-N hits into the UI:

- **Citations** (`preferred_evidence_hits`) keep the component stems the question actually names. Comparison questions may keep two families. Same-chapter siblings of a named procedure are dropped.
- **Diagrams** use identifiers from the positive clause only, skip negated plates, and prefer the named drawing/plate. Hyphenated query tokens (`air-cooler`) match titles (`air cooler`).

## Ranking signals (before pins)

- Phrase hit in section path / title / component title.
- `section_kind == data` on spec questions; `procedure` on how-to.
- Dense value tables over tool-list pages.
- Demote generic plates / intro when asking for numbers.

## Verification gate (`verify.verify`)

Six checks; if any fail, the answer model is **not** called:

| Check | Meaning |
|-------|---------|
| equipment | Query model vs fleet registry (compatible family allowed) |
| revision | Named edition vs stored edition at the convention's granularity |
| condition | Named operating condition appears in evidence |
| units | Requested units appear in evidence |
| scope | Query tokens/phrases appear in evidence |
| conflicts | Conflicting numeric claims in the window |

Revision uses `citation_convention.revision_granularity` (`per_section` or `per_document`).

## Known failure modes

| Symptom | Likely cause | Fix direction |
|---------|--------------|---------------|
| Abstain but value exists in PDF | Wrong section ranked | Phrase + kind ranking; not a re-ingest |
| Spec OK, procedure rule missing | Data sheet retrieved, checking procedure not | Kind pairing / procedure path on wear language |
| Noisy diagrams on numeric Q | Related plates attached too eagerly | Visual path only for how-to / explicit diagram asks |
| Sibling chapter in citations | Contrast clause treated as a named identifier | Positive-clause identifier extract (already in ranking) |
| Wrong numbers | Bad OCR on that page | Re-OCR / `OCR_BACKEND`, then selective re-ingest |
| Panel shown without its step | Step label sat outside the pairing window | `panels.py` slack; many panels pair automatically |
| Abstain citing a named edition mismatch | Query asked for Edition X, evidence is Y | Expected |

## Citations

Answers lead with the manual's own quote form when present (`When referring to this page, please quote Procedure … Edition …`). That string is `citation_key` plus revision. Printed page and PDF page are supplementary.

## Re-ingest vs restart

- **Restart API/UI** after code or `.env` changes. Data stays.
- **Re-ingest** only if parse/OCR output or the PDF itself changed.
- Retrieval/ranking fixes do **not** require a full re-ingest.
