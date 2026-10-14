import os
import tempfile
from typing import List, Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Discovers frequent word patterns for a specific category in session.dataframe.

        Args:
            category_name: The target category/label name to filter documents by.
            filtering_method: Method to filter local vocabulary ("variance", "tfidf", "term_frequency").
            algorithm: Pattern mining algorithm ("fpgrowth", "topk", "maxfpgrowth").
            min_sup: Minimum support count required for fpgrowth/maxfpgrowth algorithms.
            k: Number of top patterns to return for topk algorithm.

        Returns:
            Dict containing result_id, status, category_name, filtering_method, algorithm, and discovered patterns.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}

        # Parameter validation per spec (no default auto-selection)
        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            return {
                "status": "error",
                "message": f"min_sup parameter is required for algorithm '{algorithm}'."
            }
        if algorithm == "topk" and k is None:
            return {
                "status": "error",
                "message": "k parameter is required for algorithm 'topk'."
            }

        df = session.dataframe
        if "category_name" in df.columns:
            cat_df = df[df["category_name"] == category_name]
        elif "label" in df.columns:
            cat_df = df[df["label"] == category_name]
        else:
            return {"status": "error", "message": "No category column found in dataframe."}

        if cat_df.empty:
            return {
                "status": "error",
                "message": f"Category '{category_name}' not found in dataset."
            }

        # Drop empty texts
        texts = cat_df["text"].dropna().astype(str).tolist()
        texts = [t.strip() for t in texts if t.strip()]

        if not texts:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "No non-empty text documents available for category."
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 1. Local CountVectorizer with English stopwords
        vec = CountVectorizer(stop_words="english")
        dtm = vec.fit_transform(texts)
        feature_names = vec.get_feature_names_out()

        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "No terms survived initial tokenization and stopword removal."
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        dtm_dense = dtm.toarray()

        # 2. Vocabulary Filtering
        kept_indices = []
        if filtering_method == "variance":
            variances = np.var(dtm_dense, axis=0)
            low_pct, high_pct = np.percentile(variances, 5), np.percentile(variances, 95)
            # Inclusive percentile range matching helpers/text_filtering_helpers.py
            for idx, v in enumerate(variances):
                if low_pct <= v <= high_pct:
                    kept_indices.append(idx)

        elif filtering_method == "term_frequency":
            counts = np.sum(dtm_dense, axis=0)
            low_pct, high_pct = np.percentile(counts, 5), np.percentile(counts, 95)
            for idx, c in enumerate(counts):
                if low_pct <= c <= high_pct:
                    kept_indices.append(idx)

        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(dtm_dense).toarray()
            mean_tfidf = np.mean(tfidf_matrix, axis=0)
            cutoff = np.percentile(mean_tfidf, 20)
            for idx, score in enumerate(mean_tfidf):
                if score >= cutoff:
                    kept_indices.append(idx)

        if not kept_indices:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "No terms survived filtering."
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        kept_terms = [feature_names[i] for i in kept_indices]
        filtered_dtm = dtm_dense[:, kept_indices]

        # 3. Convert to Transactional DB & Mine with PAMI
        binary_df = pd.DataFrame((filtered_dtm > 0).astype(int), columns=kept_terms)

        patterns_list = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "db.txt")

            from PAMI.extras.convert import DF2DB
            converter = DF2DB.DF2DB(binary_df)
            converter.convert2TransactionalDatabase(db_path, '>=', 1)

            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(db_path, minSup=min_sup)
                miner.mine()
                res_df = miner.getPatternsAsDataFrame()

            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(db_path, k=k)
                miner.mine()
                res_df = miner.getPatternsAsDataFrame()

            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup)
                miner.mine()
                res_df = miner.getPatternsAsDataFrame()
            else:
                return {"status": "error", "message": f"Unknown algorithm: {algorithm}"}

            if res_df is not None and not res_df.empty:
                # PAMI returns columns like 'Patterns' or 'pat' and 'Support' or 'sup'
                pat_col = [c for c in res_df.columns if "pat" in c.lower()][0]
                sup_col = [c for c in res_df.columns if "sup" in c.lower()][0]

                for _, row in res_df.iterrows():
                    raw_pat = row[pat_col]
                    if isinstance(raw_pat, str):
                        pat_terms = raw_pat.strip().split("\t")
                        pat_terms = [t.strip() for t in pat_terms if t.strip()]
                    elif isinstance(raw_pat, (list, tuple)):
                        pat_terms = list(raw_pat)
                    else:
                        pat_terms = [str(raw_pat)]

                    patterns_list.append({
                        "pattern": pat_terms,
                        "support": int(row[sup_col])
                    })

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "status": "success",
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns_list
        }

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list
        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm
        }, summary)

        return summary

    return [mine_patterns_tool]
