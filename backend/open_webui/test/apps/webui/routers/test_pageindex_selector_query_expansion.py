from open_webui.retrieval.pageindex_selector.query_expansion import (
    build_selector_query_variants,
    extract_rare_tokens,
)


def test_selector_query_expansion_keeps_regulatory_signals():
    query = (
        "Does the Agency agree that there is sufficient clinical data from "
        "the OPNT003-PK-001, OPNT003-PK-002 and OPNT003-OOD-001 clinical "
        "studies to support submission of an NDA?"
    )

    variants = build_selector_query_variants(query)
    text = "\n".join(variants)

    assert len(variants) >= 4
    assert "OPNT003-PK-001" in text
    assert "OPNT003-PK-002" in text
    assert "OPNT003-OOD-001" in text
    assert "NDA" in text


def test_extract_rare_tokens_finds_study_ids_and_nda():
    query = "OPNT003-PK-001 OPNT003-PK-002 OPNT003-OOD-001 NDA"
    rare = extract_rare_tokens(query)
    assert "OPNT003-PK-001" in rare
    assert "OPNT003-PK-002" in rare
    assert "OPNT003-OOD-001" in rare
    assert "NDA" in rare
