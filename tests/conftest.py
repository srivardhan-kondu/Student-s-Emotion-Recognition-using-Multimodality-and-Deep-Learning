"""
Shared pytest fixtures (see pytest.ini for markers and addopts).
"""

import pytest


@pytest.fixture(scope="module")
def integration_predictor():
    """
    One MultimodalEmotionPredictor with load_models() per test module.
    """
    from fusion.multimodal_predictor import MultimodalEmotionPredictor

    p = MultimodalEmotionPredictor()
    p.load_models()
    return p
