import os
import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

FIXTURE_PATH = "agent_dev/sample_fixture.csv"


def setup_loaded_session():
    """Helper to initialize SessionState and load sample_fixture.csv."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": FIXTURE_PATH,
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    
    return session, mine_patterns_tool


def test_mine_patterns_fpgrowth_tfidf():
    """Known Answer #17: test tfidf filtering with fpgrowth boundary min_sup values."""
    session, mine_patterns_tool = setup_loaded_session()
    
    # Mine catA at min_sup=3 (upper boundary) -> finds alpha (support 3)
    res_a_3 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    
    assert res_a_3["status"] == "success"
    assert len(res_a_3["patterns"]) == 1
    assert res_a_3["patterns"][0]["pattern"] == ["alpha"]
    assert res_a_3["patterns"][0]["support"] == 3

    # Mine catA at min_sup=4 (above boundary) -> 0 patterns
    res_a_4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    
    assert res_a_4["status"] == "success"
    assert res_a_4["patterns"] == []

    # Mine catB at min_sup=4 (upper boundary) -> finds gamma (support 4)
    res_b_4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    
    assert res_b_4["status"] == "success"
    assert len(res_b_4["patterns"]) == 1
    assert res_b_4["patterns"][0]["pattern"] == ["gamma"]
    assert res_b_4["patterns"][0]["support"] == 4


def test_mine_patterns_zero_terms_survived():
    """Known Answer #17: variance/term_frequency filtering keeps 0 terms for both categories on this fixture."""
    session, mine_patterns_tool = setup_loaded_session()
    
    for category in ["catA", "catB"]:
        for method in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": category,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            
            assert res["status"] == "success"
            assert res["patterns"] == []
            assert "no terms survived filtering" in res.get("note", "").lower()


def test_mine_patterns_topk_and_maxfpgrowth():
    """Known Answer #20: test topk and maxfpgrowth algorithms."""
    session, mine_patterns_tool = setup_loaded_session()
    
    # topk algorithm
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 2
    })
    assert res_topk["status"] == "success"
    assert len(res_topk["patterns"]) == 1
    assert res_topk["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk["patterns"][0]["support"] == 3

    # maxfpgrowth algorithm
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max["status"] == "success"
    assert res_max["patterns"] == []
