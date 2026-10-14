"""DTM tools for document-term matrix construction and exploration."""

from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    """Creates and returns all DTM tools bound to the provided session state."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_features: Optional[int] = None,
        lowercase: bool = True
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from text documents in session.dataframe.

        Args:
            ngram_range: Lower and upper boundary of n-grams to extract, as a list of 2 integers [min_n, max_n].
            max_features: Maximum number of terms to keep ordered by term frequency.
            lowercase: Whether to convert all characters to lowercase before tokenizing.

        Returns:
            A summary dictionary containing DTM dimensions, non-zero count, total elements, and sparsity percentage.
        """
        result_id = session.next_result_id("dtm")

        if session.dataframe is None or len(session.dataframe) == 0:
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": "No dataframe found in session. Please load a dataset first."
            }
            session.store_result("build_dtm_tool", {}, summary)
            return summary

        # Extract text column, replacing missing/NaN values with empty string
        texts = session.dataframe["text"].fillna("").astype(str)

        # Convert list to tuple for CountVectorizer compatibility
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            max_features=max_features,
            lowercase=lowercase
        )

        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session attributes
        session.feature_matrix = X
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].to_numpy())

        n_rows, n_cols = X.shape
        non_zero = int(X.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": float(round(sparsity_pct, 4)),
            "vocabulary": feature_names[:10]
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={
                "ngram_range": ngram_range,
                "max_features": max_features,
                "lowercase": lowercase
            },
            summary=summary
        )

        return summary

    return [build_dtm_tool]
