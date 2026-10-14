"""Tests for tools_dtm.py using agent_dev/sample_fixture.csv."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_fixture():
    # 1. Initialize shared session state
    session = SessionState()

    # 2. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 3. Load sample fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 4. Instantiate tools under test
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 5. Execute build_dtm_tool
    result = build_dtm_tool.invoke({})

    # Assert returned dict is structured properly (JSON-serializable summary dict)
    assert isinstance(result, dict)
    assert "result_id" in result

    # Vocabulary terms: alpha, always, beta, delta, gamma (5 terms)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert len(session.feature_names) == 5

    # Matrix dimensions
    assert session.feature_matrix.shape == (8, 5)

    # Sparsity calculation: 21 non-zero entries out of 40 total
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert np.isclose(result["sparsity_pct"], 47.5)

    # Check session artifacts key
    assert "count_vectorizer" in session.artifacts
