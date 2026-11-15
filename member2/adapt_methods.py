"""Leakage-guarded SA-FARI selector-score adaptation.

This module operates only on already produced camera-only selector scores. The
k=0 operation is an unsupervised rank adjustment, not probability calibration.
Few-label operations require an explicitly supplied source-derived prior; target
scores are never relabelled as a source prior.
"""
from __future__ import annotations
import csv
import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

SEEDS = (0, 1, 2, 3, 4)
ALLOWED_K = (0, 1, 3)
_FORBIDDEN = {"audio_emb", "audioset_probs", "audio_prediction", "audio_predictions", "audio_confidence", "species", "location_id", "project", "site_id", "true_label", "correct_audio", "correct_camera", "usefulness"}

def _validate_k_seed(k: int, seed: int) -> None:
    if k not in ALLOWED_K: raise ValueError("k must be one of 0, 1, 3")
    if seed not in SEEDS: raise ValueError("seed must be one of 0, 1, 2, 3, 4")

def _score_rows(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    result, ids = [], set()
    for row in rows:
        if not isinstance(row, Mapping): raise TypeError("each score row must be a mapping")
        forbidden = _FORBIDDEN.intersection(row)
        if forbidden: raise ValueError("forbidden selector field(s): " + ", ".join(sorted(forbidden)))
        clip = row.get("video_name")
        if not isinstance(clip, str) or not clip: raise ValueError("every score row needs a non-empty video_name")
        if clip in ids: raise ValueError("duplicate video_name: " + clip)
        try: score = float(row["predicted_probability"])
        except (KeyError, TypeError, ValueError) as exc: raise ValueError("every score row needs predicted_probability") from exc
        if not math.isfinite(score) or not 0 <= score <= 1: raise ValueError("predicted_probability must be finite and in [0, 1]")
        ids.add(clip); result.append({"video_name": clip, "predicted_probability": score})
    if not result: raise ValueError("at least one target score is required")
    return result

def no_label_rank_adjust(target_scores: Iterable[Mapping[str, object]]) -> list[dict]:
    """Return target empirical-midrank values: ranking scores, not probabilities."""
    rows = _score_rows(target_scores)
    ordered = sorted(rows, key=lambda r: (r["predicted_probability"], r["video_name"]))
    n, ranks, start = len(ordered), {}, 0
    while start < n:
        end = start + 1
        while end < n and ordered[end]["predicted_probability"] == ordered[start]["predicted_probability"]: end += 1
        for row in ordered[start:end]: ranks[row["video_name"]] = ((start + 1) + end) / (2.0 * n)
        start = end
    return [{"video_name": row["video_name"], "selector_score": ranks[row["video_name"]], "score_kind": "unsupervised_target_midrank", "method": "k0_no_label", "k": 0, "seed": ""} for row in rows]

def protocol_ids(draws_path: str, k: int, seed: int) -> dict[str, str]:
    """Return approved ``video_name: species`` draw pairs after validation."""
    _validate_k_seed(k, seed)
    if k == 0: return {}
    with open(draws_path, newline="", encoding="utf-8") as handle:
        draws = [r for r in csv.DictReader(handle) if r.get("k") == str(k) and r.get("seed") == str(seed)]
    if not draws or any(not r.get("video_name") or not r.get("species") for r in draws): raise ValueError("missing or malformed protocol draws")
    approved = {r["video_name"]: r["species"] for r in draws}
    if len(approved) != len(draws): raise ValueError("duplicate protocol adaptation video_name")
    if any(count != k for count in Counter(approved.values()).values()): raise ValueError("protocol draws are not exactly k shots per represented species")
    return approved

def _validate_protocol(expected: Mapping[str, str], observed: set[str], k: int) -> None:
    if not isinstance(expected, Mapping) or not expected: raise ValueError("expected_adaptation_ids is mandatory and must be protocol_ids(...) output")
    if any(not isinstance(i, str) or not isinstance(s, str) or not s for i, s in expected.items()): raise ValueError("expected_adaptation_ids must map non-empty IDs to species")
    if observed != set(expected): raise ValueError("adaptation IDs do not match the expert-adaptation protocol")
    if any(count != k for count in Counter(expected.values()).values()): raise ValueError("approved IDs are not exactly k shots per represented species")

def few_label_recalibrate(target_scores: Iterable[Mapping[str, object]], adaptation_labels: Iterable[Mapping[str, object]], *, k: int, seed: int, evaluation_ids: Sequence[str], expected_adaptation_ids: Mapping[str, str], source_prior: float, prior_strength: float = 10.0) -> list[dict]:
    """Apply prevalence/odds recalibration using approved usefulness labels only.

    ``source_prior`` must be a source-only prediction mean or documented
    source-derived estimate. It is required so a target-score mean cannot be
    misdescribed as a source prior.
    """
    _validate_k_seed(k, seed)
    if k == 0: raise ValueError("use no_label_rank_adjust for k=0")
    if not math.isfinite(source_prior) or not 0 < source_prior < 1: raise ValueError("source_prior must be finite and strictly between 0 and 1")
    if not math.isfinite(prior_strength) or prior_strength <= 0: raise ValueError("prior_strength must be positive")
    scores = _score_rows(target_scores); by_id = {r["video_name"]: r for r in scores}
    labels, seen = [], set()
    for row in adaptation_labels:
        if set(row) != {"video_name", "usefulness"}: raise ValueError("labels may contain only video_name and usefulness")
        clip, label = row["video_name"], row["usefulness"]
        if not isinstance(clip, str) or clip in seen or clip not in by_id: raise ValueError("labels must be unique target score IDs")
        if label not in (0, 1, False, True, "0", "1"): raise ValueError("usefulness labels must be binary")
        seen.add(clip); labels.append(int(label))
    _validate_protocol(expected_adaptation_ids, seen, k)
    evaluation = set(evaluation_ids)
    if len(evaluation) != len(evaluation_ids): raise ValueError("duplicate evaluation video_name")
    if seen & evaluation: raise ValueError("adaptation/evaluation overlap")
    if not evaluation or not evaluation.issubset(by_id): raise ValueError("evaluation IDs must be non-empty target score IDs")
    target_mean = min(1 - 1e-8, max(1e-8, (prior_strength * source_prior + sum(labels)) / (prior_strength + len(labels))))
    shift = math.log(target_mean / (1 - target_mean)) - math.log(source_prior / (1 - source_prior))
    output = []
    for clip in evaluation_ids:
        p = min(1 - 1e-8, max(1e-8, by_id[clip]["predicted_probability"]))
        adjusted = 1 / (1 + math.exp(-max(-35, min(35, math.log(p / (1 - p)) + shift))))
        output.append({"video_name": clip, "selector_score": adjusted, "score_kind": "label_supported_probability", "method": "few_label_prevalence_odds", "k": k, "seed": seed})
    return output
