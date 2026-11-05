from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.models import registry
from backend.config import settings


def test_metadata_routes_never_deserialize_models():
    registry._cache.clear()
    with patch.object(registry.joblib, 'load', side_effect=AssertionError('model load')):
        with TestClient(app) as client:
            for path in ['/api/health', '/api/phases', '/api/therapeutic-areas',
                         '/api/input-ranges?phase=P2', '/api/analytics?phase=P2']:
                assert client.get(path).status_code == 200, path
    assert not registry._cache


def test_concurrent_load_and_eviction_preserve_predictions():
    registry._cache.clear()
    with patch.object(registry.joblib, 'load', wraps=registry.joblib.load) as loader:
        with ThreadPoolExecutor(max_workers=4) as pool:
            bundles = list(pool.map(lambda _: registry.load('P2'), range(4)))
        cold_load_calls = loader.call_count
        assert cold_load_calls > 0
        assert all(x is bundles[0] for x in bundles)
        registry.load('P2')
        assert loader.call_count == cold_load_calls
    first = bundles[0]['heads']['duration']
    from backend.models.inference import _build_input_row
    frame = _build_input_row('P2', 'Oncology/Solid Tumours', 200, 25, 'DRUG', 'US', ['SAFETY'])
    expected = first.predict(frame)
    registry.load('P3')
    assert len(registry._cache) <= settings.cached_phase_limit
    reloaded = registry.load('P2')['heads']['duration']
    import numpy as np
    np.testing.assert_allclose(reloaded.predict(frame), expected, rtol=1e-12, atol=1e-10)
    # A bundle held by an in-flight caller remains valid after eviction.
    np.testing.assert_allclose(first.predict(frame), expected, rtol=1e-12, atol=1e-10)
