# Retrieval & grounding

## Design principles

1. **Evidence first** — the model only sees retrieved chunks; it must not invent values.
2. **Dynamic ranking** — boosts use query intent and section metadata, not hard-coded Q→page maps.
3. **Abstain when unsure** — verification can block an answer if scope/equipment/conflicts fail.
4. **Prefer the right section kind** — numeric specs → `data`; how-to → `procedure`; diagrams on demand.

## Query intent (dynamic)

From the user question the retriever derives:

- **Content tokens** (stopwords removed, light singularization).
- **Phrases** (bigrams/trigrams) for component matching (`main bearing`, `exhaust valve`).
- **Intent flags** — data-like (`clearance`, `torque`, `diameter`, …), procedure-like (`remove`, `how`, …), figure-like (`diagram`, `plate`, …).

## Paths

| Path | Mechanism |
|------|-----------|
| Lexical | Postgres FTS (`to_tsquery` OR + short `plainto_tsquery`) + phrase ILIKE |
| Structural | Title / `doc_code` match; data vs procedure ordering by intent |
| Figures | Optional; suppressed for pure data lookups to cut plate noise |

## Ranking signals (examples)

- Phrase hit in section path/title (strong).
- `section_kind == data` on spec questions; `procedure` on how-to.
- Dense value-table text over tool-list pages.
- Demote generic `plate` / intro `other` when asking for numbers.
- Exact `D#####` / `M#####` in the query when present.

## Known failure modes

| Symptom | Likely cause | Fix direction |
|---------|--------------|---------------|
| Abstain but value exists in PDF | Wrong section ranked (e.g. “Main Engine” intro) | Phrase + kind ranking (already targeted) |
| Spec OK, procedure rule missing | Data sheet retrieved, checking procedure not | Ensure procedure path fires on wear/criteria language |
| Noisy diagrams on numeric Q | Related plates attached too eagerly | Figure attach only for how-to / explicit diagram asks |
| Wrong numbers | Bad OCR on that page | Re-OCR that page / switch `OCR_BACKEND` — then selective re-ingest |

## Re-ingest vs restart

- **Restart API/UI** after code or `.env` changes.
- **Re-ingest** only if parse/OCR output or the PDF itself changed.
- Retrieval/ranking fixes do **not** require a full re-ingest.
