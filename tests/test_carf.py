import numpy as np
from setu.fusion.carf import rrf_baseline, carf_v1, fit_carf_v2_weights, carf_v2

def test_carf_boundaries():
    raw_ranking = ["r1", "r2", "r3", "r4"]
    corrected_ranking = ["c1", "c2", "c3", "c4"]

    fused_zero = carf_v1(raw_ranking, corrected_ranking, cmi_score=0.0, cmi_max=50.0)
    assert fused_zero[:4] == raw_ranking

    fused_full = carf_v1(raw_ranking, corrected_ranking, cmi_score=50.0, cmi_max=50.0)
    assert fused_full[:4] == corrected_ranking

def test_rrf_baseline():
    list1 = ["doc_A", "doc_B", "doc_C"]
    list2 = ["doc_A", "doc_C", "doc_B"]
    fused = rrf_baseline([list1, list2])
    assert fused[0] == "doc_A"
