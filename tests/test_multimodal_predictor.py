"""
Integration tests: predictor pipelines + strict output contracts.

Run (project root, venv active):
  python -m pytest tests/test_multimodal_predictor.py -v -ra

Use -ra to see skip reasons when saved_models weights are missing.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
import soundfile as sf

from config import EMOTIONS, MODEL_SAVE_PATHS, NUM_EMOTIONS, PROJECT_ROOT

from prediction_assertions import (
    assert_fusion_result,
    assert_modality_prediction,
    format_prediction_report,
)


def _file_exists(path: Path) -> bool:
    return path.is_file()


def _text_model_ready() -> bool:
    d = PROJECT_ROOT / "saved_models" / "text_bert_model"
    return d.is_dir() and (d / "config.json").is_file()


SPEECH_PATH = Path(MODEL_SAVE_PATHS["speech"])
FACIAL_PATH = Path(MODEL_SAVE_PATHS["facial"])

SKIP_SPEECH = not _file_exists(SPEECH_PATH)
SKIP_FACIAL = not _file_exists(FACIAL_PATH)
SKIP_TEXT = not _text_model_ready()

REASON_SPEECH = f"Speech weights not found (expected {SPEECH_PATH})"
REASON_FACIAL = f"Facial weights not found (expected {FACIAL_PATH})"
REASON_TEXT = (
    "Text model missing (expected saved_models/text_bert_model/ with config.json)"
)


@pytest.fixture
def tmp_wav_22050() -> str:
    """~1.5 s synthetic WAV at 22050 Hz (matches AudioFeatureExtractor)."""
    sr = 22050
    t = np.linspace(0.0, 1.5, int(sr * 1.5), dtype=np.float32)
    wav = 0.08 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, wav, sr)
    yield path
    os.unlink(path)


@pytest.fixture
def dummy_image_path() -> str:
    img = np.full((120, 120, 3), 128, dtype=np.uint8)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, img)
    yield path
    os.unlink(path)


@pytest.mark.integration
@pytest.mark.skipif(SKIP_FACIAL, reason=REASON_FACIAL)
def test_predict_from_image_mocked_face(integration_predictor, dummy_image_path):
    """Facial CNN path with a deterministic crop (detector mocked)."""
    pred = integration_predictor
    if pred.facial_model.model is None:
        pytest.skip("Facial Keras model failed to load")

    face = np.random.default_rng(42).integers(40, 200, (48, 48), dtype=np.uint8)

    with mock.patch.object(
        pred.face_detector,
        "detect_faces",
        return_value=[(0, 0, 48, 48)],
    ):
        result = pred.predict_from_image(dummy_image_path)

    assert result is not None, "predict_from_image returned None with mocked face"
    try:
        assert_modality_prediction(result)
    except AssertionError as e:
        pytest.fail(f"{e}\n{format_prediction_report('facial', result)}")


@pytest.mark.integration
@pytest.mark.skipif(SKIP_SPEECH, reason=REASON_SPEECH)
def test_predict_from_audio_schema(integration_predictor, tmp_wav_22050):
    pred = integration_predictor
    if pred.speech_model.model is None:
        pytest.skip("Speech Keras model failed to load")

    result = pred.predict_from_audio(tmp_wav_22050)
    assert result is not None, "predict_from_audio returned None"
    try:
        assert_modality_prediction(result)
    except AssertionError as e:
        pytest.fail(f"{e}\n{format_prediction_report('speech', result)}")


@pytest.mark.integration
@pytest.mark.skipif(SKIP_TEXT, reason=REASON_TEXT)
def test_predict_from_text_schema(integration_predictor):
    pred = integration_predictor
    if pred.text_analyzer.model is None:
        pytest.skip("Text HF model failed to load")

    result = pred.predict_from_text(
        "I am so excited and happy about this exam!"
    )
    assert result is not None
    try:
        assert_modality_prediction(result)
    except AssertionError as e:
        pytest.fail(f"{e}\n{format_prediction_report('text', result)}")


def _modality_dict(emotion_index: int) -> dict:
    p = np.full(NUM_EMOTIONS, 0.02, dtype=np.float64)
    p[emotion_index] = 0.90
    p /= p.sum()
    return {
        "emotion": EMOTIONS[emotion_index],
        "confidence": float(p.max()),
        "probabilities": p,
    }


@pytest.mark.integration
def test_predict_multimodal_fusion_perfect_schema(integration_predictor):
    """
    End-to-end multimodal call: three valid probability vectors → fusion output
    satisfies the same contract as the dashboard (emotion_scores, sum to 1, …).
    """
    pred = integration_predictor

    with (
        mock.patch.object(
            pred,
            "predict_from_image",
            return_value=_modality_dict(0),
        ),
        mock.patch.object(
            pred,
            "predict_from_audio",
            return_value=_modality_dict(1),
        ),
        mock.patch.object(
            pred,
            "predict_from_text",
            return_value=_modality_dict(2),
        ),
    ):
        fused = pred.predict_multimodal(
            image_path="dummy.jpg",
            audio_path="dummy.wav",
            text="dummy",
        )

    try:
        assert_fusion_result(fused)
    except AssertionError as e:
        pytest.fail(f"{e}\nFused emotion={fused.get('emotion')!r} scores={fused.get('emotion_scores')}")


@pytest.mark.integration
@pytest.mark.skipif(
    SKIP_SPEECH or SKIP_TEXT or SKIP_FACIAL,
    reason="Requires facial + speech + text weights for live triple run",
)
def test_predict_multimodal_live_all_modalities(
    integration_predictor,
    tmp_wav_22050,
    dummy_image_path,
):
    """
    Full stack without mocking (slow). Fails only if outputs are malformed.
    Facial path uses mocked detector so it does not depend on Haar finding a face.
    """
    pred = integration_predictor
    if (
        pred.facial_model.model is None
        or pred.speech_model.model is None
        or pred.text_analyzer.model is None
    ):
        pytest.skip("One or more models failed to load")

    face = np.random.default_rng(7).integers(40, 200, (48, 48), dtype=np.uint8)
    with mock.patch.object(
        pred.face_detector,
        "detect_and_extract_faces",
        return_value=[face.copy()],
    ):
        fused = pred.predict_multimodal(
            image_path=dummy_image_path,
            audio_path=tmp_wav_22050,
            text="I feel quite neutral about the results.",
        )

    try:
        assert_fusion_result(fused)
    except AssertionError as e:
        pytest.fail(f"{e}\nFused keys={list(fused.keys())}")
