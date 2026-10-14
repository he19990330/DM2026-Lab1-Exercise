"""Tests for tools_filtering.py (variance_filter_tool, etc.)"""

import json
import pytest
import numpy as np
import pandas as pd

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    """
    Test variance_filter_tool using agent_dev/sample_fixture.csv against
    TEST_FIXTURE.md known answer #5.
    """
    # 1. Initialize single shared session state
    session = SessionState()

    # 2. Load premade dataset tool and execute
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    # 3. Build global DTM (8x5 matrix: alpha, always, beta, delta, gamma)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_res = build_dtm_tool.invoke({})
    assert dtm_res["n_terms"] == 5

    # 4. Instantiate tools_filtering
    filtering_tools = make_filtering_tools(session)
    variance_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 5. Check tool docstring & schema expectations
    assert variance_tool.name == "variance_filter_tool"
    assert variance_tool.description is not None

    # 6. Run variance_filter_tool with threshold=0.15
    res = variance_tool.invoke({"threshold": 0.15})

    # Assert return summary structure
    assert "result_id" in res
    assert "threshold" in res
    assert res["threshold"] == 0.15
    assert "n_kept" in res
    assert "n_removed" in res
    assert "kept_terms" in res
    assert "removed_terms" in res

    # Check known answer #5 specifics:
    # Retained (>= 0.15): alpha, beta, delta, gamma (4 terms)
    # Removed (< 0.15): always (1 term, variance = 0.109375)
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1
    assert "always" in res["removed_terms"]
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}

    # 7. Check stored full_report in session for schema requirements
    # full_report must be a DataFrame where 1st col is term name, 2nd col is variance
    stored_entry = session.results_store[res["result_id"]]
    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert full_report.shape[1] >= 2

    # Check delta's variance in full_report matches known answer #5 (0.25)
    col_term = full_report.columns[0]
    col_var = full_report.columns[1]
    
    delta_row = full_report[full_report[col_term] == "delta"]
    assert not delta_row.empty
    delta_var = delta_row[col_var].values[0]
    assert np.isclose(delta_var, 0.25, atol=1e-3)


def test_pearson_filter_tool():
    # 1. Initialize session state and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load the test fixture dataset
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_res = build_dtm_tool.invoke({})
    assert session.feature_matrix is not None

    # Get filtering tools
    filtering_tools = make_filtering_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # 3 & 4. Assert input schema / arguments structure
    assert pearson_tool.args_schema is not None
    schema_props = pearson_tool.args_schema.schema()["properties"]
    assert "target_class" in schema_props

    # Test invalid target_class check
    invalid_res = pearson_tool.invoke({"target_class": "catC"})
    assert isinstance(invalid_res, dict)
    assert invalid_res.get("status") == "error" or "error" in invalid_res

    # 5. Execute with target_class="catB" and verify output with known answers
    res = pearson_tool.invoke({"target_class": "catB"})

    # Verify JSON serializability of summary result
    assert json.dumps(res) is not None

    # 6. Assert output schema
    assert isinstance(res, dict)
    assert "result_id" in res
    assert "target_class" in res
    assert res["target_class"] == "catB"

    # Verify full report in session results_store
    result_id = res["result_id"]
    stored_result = session.results_store[result_id]
    full_report = stored_result["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    # Contract check: first column is term, second is pearson_r
    assert full_report.shape[1] >= 2
    assert full_report.columns[1] == "pearson_r"

    # Map terms to correlation values for testing
    terms = full_report.iloc[:, 0].tolist()
    scores = full_report.iloc[:, 1].tolist()
    r_dict = dict(zip(terms, scores))

    # Known answer #6 expectations (TEST_FIXTURE.md)
    expected = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for term, exp_score in expected.items():
        assert term in r_dict, f"Term '{term}' missing from report"
        assert r_dict[term] == pytest.approx(exp_score, abs=1e-4)
