"""
Tools for text feature filtering based on variance and statistical correlation.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function returning filtering tools bound to the given session state.
    """

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """
        Filter document-term matrix features based on variance across all documents.

        Args:
            threshold: Minimum variance required to keep a feature/term. Terms with
                variance strictly less than this threshold will be removed.

        Returns:
            Dict containing result_id, threshold, n_kept, n_removed, kept_terms,
            and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."
            }

        # Convert sparse feature matrix to dense array for variance calculation
        if hasattr(session.feature_matrix, "toarray"):
            X = session.feature_matrix.toarray()
        else:
            X = np.asarray(session.feature_matrix)

        feature_names = np.array(session.feature_names)

        # Calculate population/sample variance per term (column-wise variance)
        # ddof=0 calculates population variance matching numpy default / scikit-learn VarianceThreshold
        variances = np.var(X, axis=0, ddof=0)

        # Determine kept vs removed mask based on threshold
        kept_mask = variances >= threshold
        removed_mask = ~kept_mask

        kept_terms = feature_names[kept_mask].tolist()
        removed_terms = feature_names[removed_mask].tolist()

        # Build full_report DataFrame sorted by variance descending
        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Store result in session
        res_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": res_id,
            "threshold": threshold,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str) -> Dict[str, Any]:
        """
        Calculate Pearson correlation coefficient between term frequencies and a target category.

        Args:
            target_class: The target category name to evaluate correlation against (one-vs-rest).

        Returns:
            Dict containing result_id, target_class, n_terms, and filtered_terms summary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."
            }

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "Dataset not loaded or missing category_name column. Load dataset first."
            }

        labels = session.dataframe["category_name"].tolist()
        if target_class not in labels:
            return {
                "status": "error",
                "message": f"Target class '{target_class}' not found in dataset categories: {set(labels)}"
            }

        # Convert feature matrix to dense array
        if hasattr(session.feature_matrix, "toarray"):
            X = session.feature_matrix.toarray()
        else:
            X = np.asarray(session.feature_matrix)

        feature_names = np.array(session.feature_names)

        # One-vs-rest target vector
        y = (np.array(labels) == target_class).astype(int)

        # Calculate Pearson correlation coefficient per column
        pearson_r_list = []
        for j in range(X.shape[1]):
            col = X[:, j]
            # Handle constant/zero-variance column
            if np.std(col) == 0 or np.std(y) == 0:
                pearson_r_list.append(0.0)
            else:
                r, _ = pearsonr(col, y)
                pearson_r_list.append(0.0 if np.isnan(r) else float(r))

        # Build full_report DataFrame sorted descending by pearson_r
        df_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r_list
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        res_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": res_id,
            "target_class": target_class,
            "n_terms": len(feature_names),
            "top_positive_terms": df_report.head(5)["term"].tolist(),
            "top_negative_terms": df_report.tail(5)["term"].tolist()
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class},
            summary=summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool]
