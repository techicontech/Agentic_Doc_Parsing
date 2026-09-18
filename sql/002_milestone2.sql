-- Milestone 2 schema: Master Specification Section 5 fields.
-- Idempotent — safe to re-run against an existing Milestone 1 database.

-- Sections: procedure/plate citation keys + header titles (Section 6.1)
ALTER TABLE sections ADD COLUMN IF NOT EXISTS procedure_no    TEXT;
ALTER TABLE sections ADD COLUMN IF NOT EXISTS plate_no        TEXT;
ALTER TABLE sections ADD COLUMN IF NOT EXISTS component_title TEXT;
ALTER TABLE sections ADD COLUMN IF NOT EXISTS action_title    TEXT;

-- Elements: page-level citation keys are finer-grained than sections, because a
-- single procedure section (e.g. M90201) spans several procedure numbers
-- (902-1.2, 902-1.4). Section 7.1 ranks on the element's own procedure_no.
ALTER TABLE elements ADD COLUMN IF NOT EXISTS procedure_no        TEXT;
ALTER TABLE elements ADD COLUMN IF NOT EXISTS plate_no            TEXT;
ALTER TABLE elements ADD COLUMN IF NOT EXISTS edition             TEXT;
ALTER TABLE elements ADD COLUMN IF NOT EXISTS page_number_printed TEXT;
ALTER TABLE elements ADD COLUMN IF NOT EXISTS panel_index         INTEGER;
ALTER TABLE elements ADD COLUMN IF NOT EXISTS drawing_code        TEXT;
ALTER TABLE elements ADD COLUMN IF NOT EXISTS linked_step_number  INTEGER;

-- 'plate' is a distinct element type from 'figure' (Section 6.5)
ALTER TABLE elements DROP CONSTRAINT IF EXISTS elements_type_check;
ALTER TABLE elements ADD CONSTRAINT elements_type_check CHECK (type IN (
    'heading', 'paragraph', 'table', 'figure', 'plate', 'list', 'caption', 'other'
));

CREATE INDEX IF NOT EXISTS elements_procedure_no_idx ON elements(procedure_no);
CREATE INDEX IF NOT EXISTS elements_plate_no_idx     ON elements(plate_no);
CREATE INDEX IF NOT EXISTS elements_drawing_code_idx ON elements(drawing_code);
CREATE INDEX IF NOT EXISTS elements_panel_idx        ON elements(page, panel_index);
CREATE INDEX IF NOT EXISTS elements_step_idx         ON elements(page, linked_step_number);
CREATE INDEX IF NOT EXISTS sections_procedure_no_idx ON sections(procedure_no);
CREATE INDEX IF NOT EXISTS sections_plate_no_idx     ON sections(plate_no);

-- Panels are independently coded figures within one page (Section 6.2)
ALTER TABLE figures ADD COLUMN IF NOT EXISTS panel_index        INTEGER;
ALTER TABLE figures ADD COLUMN IF NOT EXISTS drawing_code       TEXT;
ALTER TABLE figures ADD COLUMN IF NOT EXISTS linked_step_number INTEGER;

-- Cross-reference and panel-to-step relationship types (Sections 6.4, 7.2)
ALTER TABLE element_relationships DROP CONSTRAINT IF EXISTS element_relationships_relationship_type_check;
ALTER TABLE element_relationships ADD CONSTRAINT element_relationships_relationship_type_check
    CHECK (relationship_type IN (
        'belongs_to', 'illustrates', 'describes', 'applies_to', 'superseded_by',
        'cross_ref', 'references_procedure', 'references_data', 'illustrates_step'
    ));
