"""
Entity-Level Decision Engine for Business Entity Resolution.
Converts pairwise candidate probabilities and feature signals into final predicted
match sets {matched S2/S3 IDs} per Source 1 entity, specifically optimizing Macro F0.5.
Supports global thresholding, source-specific thresholds, margin rules, candidate density adaptation,
strong evidence composite rules, and singleton-aware gating.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from evaluation import Evaluator, compute_entity_metrics


class EntityDecisionEngine:
    """
    Engine to convert candidate-pair predictions into entity-level sets
    and evaluate Entity-Level Macro F0.5.
    """

    def __init__(
        self,
        strategy: str = "global_threshold",
        threshold: float = 0.85,
        threshold_s2: Optional[float] = None,
        threshold_s3: Optional[float] = None,
        margin: float = 0.05,
        density_penalty: bool = False,
        use_strong_evidence: bool = False,
        singleton_gate: float = 0.65,
    ):
        self.strategy = strategy
        self.threshold = threshold
        self.threshold_s2 = threshold_s2 if threshold_s2 is not None else threshold
        self.threshold_s3 = threshold_s3 if threshold_s3 is not None else threshold
        self.margin = margin
        self.density_penalty = density_penalty
        self.use_strong_evidence = use_strong_evidence
        self.singleton_gate = singleton_gate

    def predict_entity_matches(
        self,
        candidate_list: List[Dict],
    ) -> Set[str]:
        """
        Given a list of candidate dictionaries for a single S1 entity:
        [
            {
                "cand_id": str,
                "source": "S2" or "S3",
                "prob": float,
                "features": Dict[str, float] (optional)
            },
            ...
        ]
        Returns the set of accepted candidate IDs.
        """
        if not candidate_list:
            return set()

        num_cands = len(candidate_list)
        accepted_ids = set()

        # Sort candidates descending by probability
        sorted_cands = sorted(candidate_list, key=lambda c: c["prob"], reverse=True)
        top_prob = sorted_cands[0]["prob"]

        # Singleton-aware gating: If top candidate is weak, predict singleton
        if top_prob < self.singleton_gate:
            return set()

        # Dynamic density adjustment
        base_thresh = self.threshold
        if self.density_penalty and num_cands > 25:
            base_thresh = min(0.95, base_thresh + 0.05)

        for cand in sorted_cands:
            prob = cand["prob"]
            source = cand.get("source", "S2")
            feats = cand.get("features", {})

            # Source specific threshold
            th = self.threshold_s2 if source == "S2" else self.threshold_s3
            if self.density_penalty and num_cands > 25:
                th = min(0.95, th + 0.05)

            # Strategy-specific logic
            if self.strategy == "global_threshold":
                if prob >= base_thresh:
                    accepted_ids.add(cand["cand_id"])

            elif self.strategy == "source_specific":
                if prob >= th:
                    accepted_ids.add(cand["cand_id"])

            elif self.strategy == "margin_relative":
                # Must beat base threshold and be within margin of the top candidate
                if prob >= base_thresh and (top_prob - prob) <= self.margin:
                    accepted_ids.add(cand["cand_id"])

            elif self.strategy == "composite_evidence":
                # Primary probability gate
                passed = prob >= base_thresh

                # Strong evidence recovery
                if not passed and self.use_strong_evidence and feats:
                    name_sim = feats.get("name_levenshtein_sim", 0.0)
                    addr_sim = feats.get("addr_levenshtein_sim", 0.0)
                    postal_match = feats.get("addr_postal_exact", 0.0)
                    trans_sim = feats.get("name_translit_sim", 0.0)
                    missing_addr = feats.get("is_cand_addr_missing", 0.0) or feats.get("is_s1_addr_missing", 0.0)

                    # Rule 1: Very strong name + address agreement or missing address
                    if name_sim >= 0.90 and (addr_sim >= 0.40 or missing_addr):
                        passed = True
                    # Rule 2: Strong postal + strong name
                    elif postal_match == 1.0 and name_sim >= 0.75:
                        passed = True
                    # Rule 3: High transliteration + address agreement
                    elif trans_sim >= 0.85 and addr_sim >= 0.40:
                        passed = True

                if passed:
                    accepted_ids.add(cand["cand_id"])

            else:
                # Default fallback
                if prob >= base_thresh:
                    accepted_ids.add(cand["cand_id"])

        return accepted_ids

    def evaluate_entity_predictions(
        self,
        s1_predictions: Dict[str, Set[str]],
        s1_ground_truth: Dict[str, List[str]],
    ) -> Dict[str, Union[int, float]]:
        """
        Evaluate full set of predictions against ground truth using Evaluator.
        """
        evaluator = Evaluator()
        for s1_id, true_mids in s1_ground_truth.items():
            true_set = set(true_mids)
            pred_set = s1_predictions.get(s1_id, set())
            evaluator.update(s1_id, true_set, pred_set)

        return evaluator.compute()


def run_threshold_sweep(
    all_s1_candidates: Dict[str, List[Dict]],
    s1_ground_truth: Dict[str, List[str]],
    thresholds: List[float],
    strategy: str = "global_threshold",
) -> List[Dict]:
    """
    Sweep across thresholds and evaluate entity-level macro F0.5.
    """
    results = []
    for th in thresholds:
        engine = EntityDecisionEngine(strategy=strategy, threshold=th, singleton_gate=min(0.70, th))
        preds: Dict[str, Set[str]] = {}

        for s1_id in s1_ground_truth.keys():
            cands = all_s1_candidates.get(s1_id, [])
            preds[s1_id] = engine.predict_entity_matches(cands)

        metrics = engine.evaluate_entity_predictions(preds, s1_ground_truth)
        metrics["threshold"] = round(th, 4)
        results.append(metrics)

    return results
