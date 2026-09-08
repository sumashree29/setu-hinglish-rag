from setu.operators.caep import extract_entity_list, entity_frequencies, build_entity_features, fit_caep_gate

def test_entity_extraction_strips_leading_stopwords():
    corpus = ["The RBI regulates all banks."]
    entities = extract_entity_list(corpus)
    assert "RBI" in entities

def test_exact_match_scores_perfectly():
    freq = {"BSBDA": 2}
    feat = build_entity_features("BSBDA", "BSBDA", freq)
    assert feat[0] == 100
    assert feat[1] > 0.99
