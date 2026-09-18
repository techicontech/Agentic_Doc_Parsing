-- Milestone 1 schema: knowledge model + fleet registry + provenance
-- Matches Architecture Spec Section 7

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE manuals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    equipment_type  TEXT NOT NULL,
    vessel_class    TEXT,
    revision        TEXT NOT NULL,
    effective_date  DATE,
    source_pdf_path TEXT NOT NULL,
    manufacturer    TEXT,
    citation_convention JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE manual_revisions (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    manual_id                   UUID NOT NULL REFERENCES manuals(id) ON DELETE CASCADE,
    revision_label              TEXT NOT NULL,
    effective_date              DATE,
    superseded_by_revision_id   UUID REFERENCES manual_revisions(id),
    change_summary              TEXT,
    UNIQUE (manual_id, revision_label)
);

CREATE TABLE sections (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    manual_id           UUID NOT NULL REFERENCES manuals(id) ON DELETE CASCADE,
    parent_section_id   UUID REFERENCES sections(id) ON DELETE SET NULL,
    path                TEXT[] NOT NULL,
    title               TEXT NOT NULL,
    page_start          INTEGER,
    page_end            INTEGER,
    doc_code            TEXT,          -- e.g. M90101, D10101, P90151
    edition             TEXT,          -- e.g. 0249
    section_kind        TEXT,          -- data | procedure | plate | schedule | other
    procedure_no        TEXT,          -- e.g. 902-1.3 (primary citation key)
    plate_no            TEXT,          -- e.g. P90001 (primary citation key for plates)
    component_title     TEXT,          -- page header component, e.g. Piston
    action_title        TEXT,          -- page header action, e.g. Dismantling
    citation_key        TEXT,          -- manufacturer-agnostic quote string
    citation_fields     JSONB,         -- flexible bag; do not add maker-specific columns
    UNIQUE (manual_id, path)
);

CREATE INDEX sections_manual_id_idx ON sections(manual_id);
CREATE INDEX sections_doc_code_idx ON sections(doc_code);
CREATE INDEX sections_procedure_no_idx ON sections(procedure_no);
CREATE INDEX sections_plate_no_idx ON sections(plate_no);
CREATE INDEX sections_page_range_idx ON sections(manual_id, page_start, page_end);

CREATE TABLE elements (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    section_id          UUID REFERENCES sections(id) ON DELETE SET NULL,
    type                TEXT NOT NULL CHECK (type IN (
                            'heading', 'paragraph', 'table', 'figure', 'plate', 'list', 'caption', 'other'
                        )),
    page                INTEGER NOT NULL,
    bbox                JSONB,         -- [x0, y0, x1, y1] in page coords
    text                TEXT,
    table_json          JSONB,
    figure_id           TEXT,
    tsv                 TSVECTOR,
    extractor_name      TEXT NOT NULL,
    extractor_version   TEXT NOT NULL,
    confidence          REAL,
    -- Page-level citation keys: one procedure section spans several procedure_no
    -- values, so these live on the element and not only on the section.
    procedure_no        TEXT,
    plate_no            TEXT,
    edition             TEXT,
    page_number_printed TEXT,          -- supplementary only, never the primary key
    panel_index         INTEGER,       -- position of this figure panel within the page
    drawing_code        TEXT,          -- e.g. M90201-0285D06 (printed on panel border)
    linked_step_number  INTEGER,       -- step/label this panel illustrates, when shared
    citation_key        TEXT,          -- always populated (fallback p.{page})
    citation_fields     JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX elements_section_id_idx ON elements(section_id);
CREATE INDEX elements_page_idx ON elements(page);
CREATE INDEX elements_type_idx ON elements(type);
CREATE INDEX elements_procedure_no_idx ON elements(procedure_no);
CREATE INDEX elements_plate_no_idx ON elements(plate_no);
CREATE INDEX elements_drawing_code_idx ON elements(drawing_code);
CREATE INDEX elements_panel_idx ON elements(page, panel_index);
CREATE INDEX elements_step_idx ON elements(page, linked_step_number);
CREATE INDEX sections_citation_key_idx ON sections(citation_key);
CREATE INDEX elements_citation_key_idx ON elements(citation_key);
CREATE INDEX elements_tsv_idx ON elements USING GIN (tsv);
CREATE INDEX elements_text_trgm_ready_idx ON elements (lower(text));

CREATE OR REPLACE FUNCTION elements_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', coalesce(NEW.text, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER elements_tsv_update
    BEFORE INSERT OR UPDATE OF text ON elements
    FOR EACH ROW EXECUTE FUNCTION elements_tsv_trigger();

CREATE TABLE figures (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    element_id      UUID REFERENCES elements(id) ON DELETE SET NULL,
    manual_id       UUID NOT NULL REFERENCES manuals(id) ON DELETE CASCADE,
    page            INTEGER NOT NULL,
    section_id      UUID REFERENCES sections(id) ON DELETE SET NULL,
    image_path      TEXT NOT NULL,     -- MinIO object key
    caption         TEXT,
    figure_label    TEXT,              -- e.g. Fig-4.7 / P90151
    panel_index     INTEGER,           -- 1..N panels per page (Section 6.2)
    drawing_code    TEXT,
    linked_step_number INTEGER,
    UNIQUE (manual_id, page, image_path)
);

CREATE INDEX figures_section_id_idx ON figures(section_id);
CREATE INDEX figures_page_idx ON figures(manual_id, page);

CREATE TABLE element_relationships (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_element_id     UUID NOT NULL REFERENCES elements(id) ON DELETE CASCADE,
    to_element_id       UUID NOT NULL REFERENCES elements(id) ON DELETE CASCADE,
    relationship_type   TEXT NOT NULL CHECK (relationship_type IN (
                            'belongs_to', 'illustrates', 'describes', 'applies_to', 'superseded_by',
                            'cross_ref', 'references_procedure', 'references_section', 'references_data', 'illustrates_step'
                        )),
    UNIQUE (from_element_id, to_element_id, relationship_type)
);

CREATE TABLE element_provenance (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    element_id          UUID NOT NULL REFERENCES elements(id) ON DELETE CASCADE,
    source_pdf_path     TEXT NOT NULL,
    revision_id         UUID REFERENCES manual_revisions(id),
    page                INTEGER NOT NULL,
    bbox                JSONB,
    extractor_name      TEXT NOT NULL,
    extractor_version   TEXT NOT NULL,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE fleet_registry (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vessel_imo      TEXT,
    vessel_class    TEXT,
    equipment_id    TEXT NOT NULL,
    manufacturer    TEXT,
    model           TEXT NOT NULL,
    serial_number   TEXT,
    manual_id       UUID NOT NULL REFERENCES manuals(id),
    revision_status TEXT NOT NULL DEFAULT 'current',
    UNIQUE (equipment_id, manual_id)
);

-- Reserved for Milestone 2; created now so schema is complete.
CREATE TABLE query_log (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text              TEXT NOT NULL,
    router_labels           TEXT[],
    retrieved_element_ids   UUID[],
    fusion_method           TEXT,
    verification_result     JSONB,
    failed_checks           TEXT[],
    final_answer            TEXT,
    abstained               BOOLEAN NOT NULL DEFAULT false,
    timestamp               TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Ingest run tracking for Milestone 1 ops
CREATE TABLE ingest_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    manual_id       UUID REFERENCES manuals(id) ON DELETE SET NULL,
    source_pdf_path TEXT NOT NULL,
    status          TEXT NOT NULL,
    page_count      INTEGER,
    docling_pages   INTEGER,
    mistral_pages   INTEGER,
    skipped_pages   INTEGER,
    notes           JSONB,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ
);