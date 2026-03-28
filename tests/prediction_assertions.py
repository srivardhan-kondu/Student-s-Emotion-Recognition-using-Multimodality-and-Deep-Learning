"""
Strict checks for emotion prediction dicts — use in integration tests.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from config import EMOTIONS, NUM_EMOTIONS


def assert_probability_vector(
    probs: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> np.ndarray:
    """
    Returns probs as float64 (6,) after validating:
    - shape (NUM_EMOTIONS,)
    - non-negative (within atol)
    - sums to 1 (within rtol)
    """
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    assert arr.shape == (NUM_EMOTIONS,), (
        f"Expected probabilities shape ({NUM_EMOTIONS},), got {arr.shape}"
    )
    assert np.all(arr >= -atol), f"Negative probabilities: min={arr.min()}"
    s = float(arr.sum())
    assert abs(s - 1.0) <= rtol, f"Probabilities sum to {s}, expected 1.0"
    return arr


def assert_modality_prediction(result: Mapping[str, Any]) -> None:
    """Validate predict_from_image / audio / text return shape."""
    assert isinstance(result, Mapping), f"Expected dict-like result, got {type(result)}"
    assert "emotion" in result and "confidence" in result and "probabilities" in result
    assert result["emotion"] in EMOTIONS, (
        f"emotion {result['emotion']!r} not in canonical EMOTIONS={EMOTIONS}"
    )
    conf = float(result["confidence"])
    assert 0.0 <= conf <= 1.0 + 1e-5, f"confidence out of range: {conf}"
    probs = assert_probability_vector(result["probabilities"])
    # Top class should match argmax (numerical tolerance)
    idx = int(np.argmax(probs))
    assert EMOTIONS[idx] == result["emotion"], (
        f"argmax label {EMOTIONS[idx]} != reported emotion {result['emotion']}"
    )


def assert_fusion_result(result: Mapping[str, Any]) -> None:
    """Validate multimodal fuse() output."""
    assert isinstance(result, Mapping), f"Expected dict-like result, got {type(result)}"
    for key in ("emotion", "confidence", "emotion_scores", "modalities_used"):
        assert key in result, f"Missing key {key!r}"
    assert result["emotion"] in EMOTIONS
    assert 0.0 <= float(result["confidence"]) <= 1.0 + 1e-5
    scores = result["emotion_scores"]
    assert isinstance(scores, Mapping) and len(scores) == NUM_EMOTIONS
    for e in EMOTIONS:
        assert e in scores, f"Missing emotion score for {e!r}"
        v = float(scores[e])
        assert 0.0 <= v <= 1.0 + 1e-5, f"score for {e} out of range: {v}"
    vec = np.array([scores[e] for e in EMOTIONS], dtype=np.float64)
    assert_probability_vector(vec)
    mu = result["modalities_used"]
    assert isinstance(mu, (list, tuple)), "modalities_used should be a list"
    for m in mu:
        assert m in ("facial", "speech", "text"), f"unknown modality {m!r}"


def format_prediction_report(
    title: str,
    result: Mapping[str, Any],
    emotions: Sequence[str] | None = None,
) -> str:
    """Human-readable block for pytest failure messages or logging."""
    emotions = list(emotions or EMOTIONS)
    lines = [f"=== {title} ===", f"  emotion:     {result.get('emotion')}", f"  confidence:  {result.get('confidence')!r}"]
    probs = result.get("probabilities")
    if probs is not None:
        arr = np.asarray(probs, dtype=np.float64)
        order = np.argsort(-arr)
        ranked = ", ".join(f"{emotions[i]}={arr[i]:.4f}" for i in order[:6])
        lines.append(f"  ranked:      {ranked}")
    return "\n".join(lines)
