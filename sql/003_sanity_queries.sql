-- Milestone 1 SQL sanity suite
-- Run after full ingest. Expects one Vol II manual + one fleet_registry stub.
-- Usage: psql $DATABASE_URL -f sql/003_sanity_queries.sql

\echo '=== 0. Corpus counts ==='
SELECT
  (SELECT count(*) FROM manuals) AS manuals,
  (SELECT count(*) FROM sections) AS sections,
  (SELECT count(*) FROM elements) AS elements,
  (SELECT count(*) FROM figures) AS figures,
  (SELECT count(*) FROM element_relationships) AS relationships,
  (SELECT count(*) FROM fleet_registry) AS fleet_rows;

\echo '=== 0b. Fleet registry stub (must be exactly one current S50MC-C row) ==='
SELECT equipment_id, manufacturer, model, revision_status, manual_id
FROM fleet_registry;

\echo '=== Q1 Exact identifier: fuel valve tightening torque D09-41 ==='
SELECT e.page, s.doc_code, s.path, left(coalesce(e.text, e.table_json::text), 240) AS evidence
FROM elements e
JOIN sections s ON s.id = e.section_id
WHERE e.text ILIKE '%D09-41%'
   OR e.text ILIKE '%Fuel valve tightening torque%'
   OR e.table_json::text ILIKE '%D09-41%'
   OR e.table_json::text ILIKE '%Fuel valve tightening torque%'
ORDER BY e.page
LIMIT 10;

\echo '=== Q2 Exact identifier: exhaust valve stud screwing-in torque D01-01 ==='
SELECT e.page, s.doc_code, s.path, left(coalesce(e.text, e.table_json::text), 240) AS evidence
FROM elements e
JOIN sections s ON s.id = e.section_id
WHERE e.text ILIKE '%D01-01%'
   OR (e.text ILIKE '%Exhaust valve stud%' AND e.text ILIKE '%torque%')
   OR e.table_json::text ILIKE '%D01-01%'
   OR e.table_json::text ILIKE '%Exhaust valve stud%'
ORDER BY e.page
LIMIT 10;

\echo '=== Q3 Procedure: Cylinder Cover dismantling (M90101) ==='
SELECT s.doc_code, s.edition, s.page_start, s.page_end, s.path,
       left(string_agg(e.text, E'\n' ORDER BY e.page, e.id), 800) AS evidence_preview
FROM sections s
LEFT JOIN elements e ON e.section_id = s.id
WHERE s.doc_code = 'M90101' OR s.title ILIKE '%Cylinder Cover%'
GROUP BY s.id
ORDER BY s.page_start
LIMIT 5;

\echo '=== Q4 Procedure: Fuel Valve checking (M90911) ==='
SELECT s.doc_code, s.edition, s.page_start, s.page_end, s.path,
       left(string_agg(e.text, E'\n' ORDER BY e.page, e.id), 800) AS evidence_preview
FROM sections s
LEFT JOIN elements e ON e.section_id = s.id
WHERE s.doc_code = 'M90911' OR (s.title ILIKE '%Fuel Valve%' AND s.section_kind = 'procedure')
GROUP BY s.id
ORDER BY s.page_start
LIMIT 5;

\echo '=== Q5 Diagram/figure: Cylinder Cover Panel plate (P90151) ==='
SELECT f.page, f.figure_label, f.caption, f.image_path, s.doc_code, s.path
FROM figures f
JOIN sections s ON s.id = f.section_id
WHERE s.doc_code = 'P90151'
   OR s.title ILIKE '%Cylinder Cover Panel%'
ORDER BY f.page;

\echo '=== Q6 Diagram/figure: Fuel Valve / Fuel Pump panel (P90951) ==='
SELECT f.page, f.figure_label, f.caption, f.image_path, s.doc_code, s.path
FROM figures f
JOIN sections s ON s.id = f.section_id
WHERE s.doc_code = 'P90951'
   OR s.title ILIKE '%Fuel Valve and Fuel Pump Panel%'
ORDER BY f.page;

\echo '=== Q7 Cross-reference: elements pointing at Procedure 909-5 / 909-11 ==='
SELECT er.relationship_type,
       fe.page AS from_page, left(fe.text, 160) AS from_text,
       te.page AS to_page, ts.doc_code AS to_doc_code
FROM element_relationships er
JOIN elements fe ON fe.id = er.from_element_id
JOIN elements te ON te.id = er.to_element_id
LEFT JOIN sections ts ON ts.id = te.section_id
WHERE er.relationship_type = 'cross_ref'
  AND (
    fe.text ILIKE '%909-5%'
    OR fe.text ILIKE '%909-11%'
    OR ts.doc_code ILIKE 'M909%'
  )
LIMIT 20;

\echo '=== Q8 Cross-reference: Data sheet describes linked procedure ==='
SELECT er.relationship_type,
       fs.doc_code AS from_code, fs.section_kind AS from_kind,
       ts.doc_code AS to_code, ts.section_kind AS to_kind
FROM element_relationships er
JOIN elements fe ON fe.id = er.from_element_id
JOIN elements te ON te.id = er.to_element_id
JOIN sections fs ON fs.id = fe.section_id
JOIN sections ts ON ts.id = te.section_id
WHERE er.relationship_type IN ('describes', 'illustrates')
ORDER BY from_code
LIMIT 30;

\echo '=== Q9 Wrong-equipment trap support: only S50MC-C is registered ==='
-- Later agent should abstain if query equipment != fleet model.
SELECT fr.model AS registered_model,
       m.title,
       m.revision,
       CASE WHEN fr.model = 'S50MC-C' THEN 'PASS: can detect non-S50MC-C queries as wrong-equipment'
            ELSE 'FAIL'
       END AS trap_support
FROM fleet_registry fr
JOIN manuals m ON m.id = fr.manual_id;

\echo '=== Q10 Unanswerable support: no Vol I Operation manual present ==='
-- Query about Vol I operation content should find zero matching manuals/sections.
-- Note: use a pattern that does NOT match "Volume II".
SELECT count(*) AS vol1_manuals
FROM manuals
WHERE title ~* 'Volume[[:space:]]*I([^IV]|$)' OR title ILIKE '%Vol. I%';

SELECT count(*) AS scavenge_air_operation_hits
FROM elements e
WHERE e.text ILIKE '%scavenge air receiver operation procedure in Volume I%';

\echo '=== Spot-check helper: 10 random populated pages ==='
SELECT e.page,
       count(*) AS element_count,
       array_agg(DISTINCT e.type) AS types,
       max(s.doc_code) AS doc_code
FROM elements e
LEFT JOIN sections s ON s.id = e.section_id
GROUP BY e.page
ORDER BY random()
LIMIT 10;