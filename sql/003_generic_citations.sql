-- Generic citation keys (Master Spec v2 §4–§5).
-- Existing procedure_no / plate_no / edition stay as citation_fields copies so
-- an already-ingested corpus does not need a full re-ingest.

ALTER TABLE manuals ADD COLUMN IF NOT EXISTS manufacturer TEXT;
ALTER TABLE manuals ADD COLUMN IF NOT EXISTS citation_convention JSONB;

ALTER TABLE sections ADD COLUMN IF NOT EXISTS citation_key TEXT;
ALTER TABLE sections ADD COLUMN IF NOT EXISTS citation_fields JSONB;

ALTER TABLE elements ADD COLUMN IF NOT EXISTS citation_key TEXT;
ALTER TABLE elements ADD COLUMN IF NOT EXISTS citation_fields JSONB;

CREATE INDEX IF NOT EXISTS sections_citation_key_idx ON sections(citation_key);
CREATE INDEX IF NOT EXISTS elements_citation_key_idx ON elements(citation_key);

ALTER TABLE element_relationships DROP CONSTRAINT IF EXISTS element_relationships_relationship_type_check;
ALTER TABLE element_relationships ADD CONSTRAINT element_relationships_relationship_type_check
    CHECK (relationship_type IN (
        'belongs_to', 'illustrates', 'describes', 'applies_to', 'superseded_by',
        'cross_ref', 'references_procedure', 'references_section', 'references_data',
        'illustrates_step'
    ));

-- Backfill from whatever this corpus already stored.
UPDATE sections
SET citation_key = COALESCE(NULLIF(plate_no, ''), NULLIF(procedure_no, ''), NULLIF(doc_code, ''), citation_key)
WHERE citation_key IS NULL;

UPDATE sections
SET citation_fields = jsonb_strip_nulls(jsonb_build_object(
        'procedure_no', procedure_no,
        'plate_no', plate_no,
        'edition', edition,
        'doc_code', doc_code,
        'component_title', component_title,
        'action_title', action_title,
        'section_kind', section_kind
    ))
WHERE citation_fields IS NULL;

UPDATE elements e
SET citation_key = COALESCE(
    NULLIF(e.plate_no, ''),
    NULLIF(e.procedure_no, ''),
    NULLIF(s.doc_code, ''),
    CASE WHEN e.page_number_printed IS NOT NULL THEN 'p.' || e.page_number_printed END,
    'p.' || e.page::text
)
FROM sections s
WHERE e.section_id = s.id AND e.citation_key IS NULL;

UPDATE elements
SET citation_fields = jsonb_strip_nulls(jsonb_build_object(
        'procedure_no', procedure_no,
        'plate_no', plate_no,
        'edition', edition,
        'drawing_code', drawing_code,
        'panel_index', panel_index,
        'linked_step_number', linked_step_number
    ))
WHERE citation_fields IS NULL;
