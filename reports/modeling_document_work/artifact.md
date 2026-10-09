# Modeling report template contract

Reference original: C:/Users/id116/Downloads/모델링결과서 (2).doc.
Retained binary copy: reference_original.doc, SHA256 recorded in template_evidence.json.
Word converted reference: reference.docx, unchanged after conversion. Three rendered pages, one section, one outer table with ten rows. Every reference page inspected.

Page system: A4 11906 by 16838 twips. Margins top 1701, bottom 1440, left/right 1440; header/footer 850. One column, zero column spacing. Existing section and header are preserve-only. Outer table width 9100 twips, black single border 3 eighth-points, existing cover gray shading and merged rows are authoritative.

Typography: source 바탕글 paragraph style a3 with 돋움 direct run font. Cover title 15pt bold centered; report heading 12pt bold. Cover lines use source 1.3 line spacing. Source body subsection patterns are reused at 11pt, headings at 12pt bold, black text. New numeric tables use 10pt and source-aligned gray header fills and light borders. Captions 10pt.

Slot map, word/document.xml outer table rows indexed from zero:
- row0 title: preserve unchanged.
- row1 spacer: preserve unchanged.
- row2 second cell: blank receipt number until assigned.
- row3 second cell: user-provided team 김형수없는김형수.
- row4 second cell: no awards, user-provided 없음.
- row5 second cell: verified Python analysis/training libraries, two lines maximum.
- row6 merged summary heading: preserve.
- row7 merged cell: replace template instructions and empty response space with idea background and impact, each within five lines. Preserve template rhythm and sufficient blank space to retain a cover page.
- row8 merged report heading: preserve reference wording and source formatting.
- row9 merged body: replace instructional examples and blanks with substantive report. Required ordered sections: ① 분석(EDA) 개요; 1-1 분석 핵심 내용; 1-2 분석 배경; 1-3 분석 방법 및 분석 내용; ② 데이터 전처리; 2-1 전처리 방법; 2-2 전처리 절차; ③ 모델링; 3-1 배경/알고리즘 소개; 3-2 모델링 방법; 3-3 성능. Add conclusion and reproducibility under modeling. Clone source paragraph roles and allow body row to split across pages. Body limit ten pages; target entire document at most ten pages.

Data contract: train70-only baseline's fixed validation20 result is distinct from final alloy-separated CV90. Never label CV90 or its 20-image subset as a train70/valid20 holdout experiment. Unknown seed0 full-CV values must remain pending in the internal layout draft and cannot be delivered as final. Seed1 is pilot-only folds0/1; no full-seed-stability claim. Current submitted candidate remains sub6 unless explicitly changed.

Package contract: reference_parts.json inventories all existing package parts and SHA256. Only word/document.xml and word/_rels/document.xml.rels are editable; [Content_Types].xml may add image declarations only. Existing styles, numbering, settings, headers, footnotes/endnotes, theme, font table, metadata and all original relationships remain byte-for-byte preserved. New images have new unique relationship IDs and media names. No content controls or populated TOC require mutation.

Fidelity gate: original hash unchanged; section XML unchanged; preserve-only parts unchanged; cover labels, gray source rows, borders, fonts and mandatory headings recognizably retained; every final page rendered through packaged render_docx.py and inspected. This runtime has no bundled LibreOffice, so native Word exports the PDF and the packaged renderer rasterizes it via bundled Poppler. Internal draft is not a final deliverable. Administrative number stays blank, awards and team supplied by user are filled.

Internal layout QA: final draft iteration v5 is six pages, all six PNGs inspected. Removed the empty form's body-row minimum height and set its text to top alignment so the filled report flows naturally. The source's flow without keep-with-next chains is retained; Word otherwise deferred a large body row and left a near-empty title page. Cover remains page1, original report heading and EDA start page2. All preserve-only parts and section XML match. Final numerical tables will add content, so final render and visual inspection must be repeated before delivery. The packaged renderer was retained locally as packaged_render_docx.py after the plugin cache updated during this task.

Final delivery QA (2026-10-10): the completed report through round17 is eight pages. All final pages were visually inspected; original package parts and section geometry are preserved. The delivered DOCX SHA256 is 42dcb8d60451963febae10e155b5a59d2606a17a9d630eb40868eade351d597e. Final evidence is in final_build_evidence.json and layout_qa.json. Rendering caches are local-only; retained template inputs and the final example figure are versioned. Round18 training started afterward and its results are not yet included in this document.
