from typing import Literal, Optional, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> Dict[str, Any]:
        """Reduces feature matrix dimensionality (PCA/t-SNE/UMAP) and plots 2D coordinates colored by category.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', 'umap').
            n_components: Number of target dimensions (default 2).
            random_state: Random seed for reproducibility.
            perplexity: t-SNE perplexity parameter (must be < n_samples).
            n_neighbors: UMAP n_neighbors parameter (must be < n_samples).

        Returns:
            Dict containing result_id, method, n_documents, and explained_variance_ratio (PCA only).
        """
        # 1. Validation checks
        if session.feature_matrix is None:
            return {
                "status": "error",
                "message": "No feature matrix found in session. Build DTM first.",
            }

        if session.labels is None:
            return {
                "status": "error",
                "message": "No category labels found in session.",
            }

        if n_components < 2:
            return {
                "status": "error",
                "message": "n_components must be at least 2 for 2D plotting.",
            }

        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X = X.toarray()

        n_samples = X.shape[0]

        if method == "tsne" and perplexity >= n_samples:
            return {
                "status": "error",
                "message": f"perplexity ({perplexity}) must be strictly less than n_samples ({n_samples}).",
            }

        if method == "umap" and n_neighbors >= n_samples:
            return {
                "status": "error",
                "message": f"n_neighbors ({n_neighbors}) must be strictly less than n_samples ({n_samples}).",
            }

        # 2. Perform dimensionality reduction
        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        else:
            return {
                "status": "error",
                "message": f"Unsupported reduction method: {method}",
            }

        # 3. Build plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=session.labels,
            palette="Set2",
            ax=ax,
            s=80,
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Store figure on pending_figure
        session.pending_figure = fig

        # 4. Store result
        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_documents": int(n_samples),
            "explained_variance_ratio": explained_variance_ratio,
        }

        full_report = {
            "coordinates": coords,
        }

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary=summary,
            full_report=full_report,
        )

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarizes/one-hot encodes category labels."""
        return {"status": "not_implemented"}

    return [reduce_dimensions_tool, binarize_labels_tool]
