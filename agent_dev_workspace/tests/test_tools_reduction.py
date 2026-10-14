import json
import matplotlib.figure
import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_fixture_dtm():
    """Helper to set up a clean session loaded with sample_fixture.csv and DTM built."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session


def _verify_plot_and_store(session, res_id):
    """Verify plot structure (total offsets == 8, distinct colors >= 2) and result storage."""
    assert res_id in session.results_store

    fig = session.pending_figure
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]

    total_offsets = sum(len(coll.get_offsets()) for coll in ax.collections)
    assert total_offsets == 8

    colors = []
    for coll in ax.collections:
        fc = coll.get_facecolors()
        if len(fc) > 0:
            colors.extend(fc)
    unique_colors = np.unique(np.array(colors), axis=0)
    assert len(unique_colors) >= 2


@pytest.fixture(autouse=True)
def prevent_savefig_and_show(monkeypatch):
    """Enforce that no tool calls savefig() or show() during test execution."""
    def _forbidden_call(*args, **kwargs):
        raise AssertionError(
            "Tool called savefig() or show() directly! Tools must set session.pending_figure instead."
        )

    monkeypatch.setattr("matplotlib.pyplot.savefig", _forbidden_call)
    monkeypatch.setattr("matplotlib.pyplot.show", _forbidden_call)
    monkeypatch.setattr("matplotlib.figure.Figure.savefig", _forbidden_call)
    monkeypatch.setattr("matplotlib.figure.Figure.show", _forbidden_call)


def test_input_schema():
    session = SessionState()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    schema = reduce_tool.args_schema.model_json_schema()
    properties = schema["properties"]

    assert "method" in properties
    assert "n_components" in properties
    assert "random_state" in properties
    assert "perplexity" in properties
    assert "n_neighbors" in properties

    assert properties["n_components"].get("default") == 2

    method_prop = properties["method"]
    if "enum" in method_prop:
        assert set(method_prop["enum"]) == {"pca", "tsne", "umap"}


def test_reduce_dimensions_pca():
    session = _setup_fixture_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known answer #7: PCA with n_components=2
    res = reduce_tool.invoke({"method": "pca", "n_components": 2})

    # Summary schema checks
    assert "result_id" in res
    assert "explained_variance_ratio" in res
    assert "coordinates" not in res  # Coordinates should be strictly in full_report
    assert json.dumps(res) is not None

    # Retrieve coordinates strictly from stored full_report
    res_id = res["result_id"]
    stored_entry = session.results_store[res_id]
    full_report = stored_entry.get("full_report")
    assert full_report is not None
    coords = full_report.get("coordinates") if isinstance(full_report, dict) else None
    assert coords is not None

    expected_coords = np.array(
        [
            [2.3669, 0.8871],
            [1.7039, 0.2642],
            [1.0408, -0.3587],
            [-2.0759, 1.0129],
            [-1.4554, 0.3405],
            [-0.8349, -0.3319],
            [-0.8349, -0.3319],
            [0.0894, -1.4822],
        ]
    )

    np.testing.assert_allclose(np.array(coords), expected_coords, atol=1e-4)

    expected_evr = [0.7532, 0.1965]
    np.testing.assert_allclose(
        res["explained_variance_ratio"], expected_evr, atol=1e-4
    )

    _verify_plot_and_store(session, res_id)


@pytest.mark.xfail(
    reason="Known fixture coordinate discrepancy due to scikit-learn 1.6+ TSNE default changes and NumPy 2.0 BLAS differences"
)
def test_reduce_dimensions_tsne():
    session = _setup_fixture_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known answer #8: t-SNE with perplexity=2, random_state=42
    res = reduce_tool.invoke(
        {"method": "tsne", "perplexity": 2, "random_state": 42}
    )

    assert "result_id" in res
    assert "coordinates" not in res
    assert res.get("explained_variance_ratio") is None
    assert json.dumps(res) is not None

    res_id = res["result_id"]
    stored_entry = session.results_store[res_id]
    full_report = stored_entry.get("full_report")
    assert full_report is not None
    coords = full_report.get("coordinates") if isinstance(full_report, dict) else None
    assert coords is not None

    expected_coords = np.array(
        [
            [1330.9991, 32.1039],
            [951.2459, 393.5588],
            [530.8209, 682.9791],
            [-1558.6448, -314.1625],
            [-915.5153, -346.3907],
            [-280.0313, -1261.2115],
            [-280.0313, -1261.2115],
            [-701.3674, 1465.3973],
        ]
    )

    np.testing.assert_allclose(np.array(coords), expected_coords, atol=1e-4)

    _verify_plot_and_store(session, res_id)


@pytest.mark.xfail(
    reason="Known fixture coordinate discrepancy due to UMAP 0.5.9+ / Numba / NumPy 2.0 BLAS differences"
)
def test_reduce_dimensions_umap():
    session = _setup_fixture_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known answer #9: UMAP with n_neighbors=3, random_state=42
    res = reduce_tool.invoke(
        {"method": "umap", "n_neighbors": 3, "random_state": 42}
    )

    assert "result_id" in res
    assert "coordinates" not in res
    assert res.get("explained_variance_ratio") is None
    assert json.dumps(res) is not None

    res_id = res["result_id"]
    stored_entry = session.results_store[res_id]
    full_report = stored_entry.get("full_report")
    assert full_report is not None
    coords = full_report.get("coordinates") if isinstance(full_report, dict) else None
    assert coords is not None

    expected_coords = np.array(
        [
            [6.6811, -6.8218],
            [6.9907, -7.2245],
            [7.7022, -7.3712],
            [10.2228, -6.2310],
            [9.8759, -5.6537],
            [9.3831, -6.5748],
            [9.2153, -5.8297],
            [8.5860, -7.2622],
        ]
    )

    np.testing.assert_allclose(np.array(coords), expected_coords, atol=1e-4)

    _verify_plot_and_store(session, res_id)


def test_reduce_dimensions_error_handling():
    session = SessionState()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # 1. Feature matrix is None
    res = reduce_tool.invoke({"method": "pca"})
    assert res.get("status") == "error"

    # Set up DTM
    session = _setup_fixture_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # 2. session.labels is None
    session.labels = None
    res = reduce_tool.invoke({"method": "pca"})
    assert res.get("status") == "error"

    # Restore session
    session = _setup_fixture_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # 3. n_components < 2
    res = reduce_tool.invoke({"method": "pca", "n_components": 1})
    assert res.get("status") == "error"

    # 4. perplexity >= n_samples (8 samples in fixture)
    res = reduce_tool.invoke({"method": "tsne", "perplexity": 8})
    assert res.get("status") == "error"

    # 5. n_neighbors >= n_samples
    res = reduce_tool.invoke({"method": "umap", "n_neighbors": 8})
    assert res.get("status") == "error"
