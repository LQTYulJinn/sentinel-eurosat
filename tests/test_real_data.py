"""Checks the assumed band order against the physics of the real patches (skipped without the cache)."""
import numpy as np, pytest
from sentinel_eurosat.data import CACHE
from sentinel_eurosat.indices import IDX, ndvi, ndwi

pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="run build_cache() first")


@pytest.fixture(scope="module")
def sample():
    d = dict(np.load(CACHE)); X = np.load(CACHE.with_suffix(".X.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(d["y"]), 1500, replace=False))
    return X[idx].astype(np.float32), d["y"][idx], list(d["classes"])


def mean_of(x, band):
    return x[:, IDX[band]].mean()


def test_band_order_matches_physics(sample):
    x, _, _ = sample
    assert mean_of(x, "B10") < 50                       # cirrus band is ~0 at the surface
    assert mean_of(x, "B8A") >= mean_of(x, "B08") * 0.95  # narrow NIR tracks wide NIR
    assert mean_of(x, "B12") < mean_of(x, "B11")        # SWIR2 below SWIR1
    assert mean_of(x, "B04") < mean_of(x, "B08")        # vegetation-dominated scenes: red below NIR


def test_indices_separate_classes_as_expected(sample):
    x, y, cl = sample
    xt = x.transpose(1, 0, 2, 3)
    v, w = ndvi(xt).mean((1, 2)), ndwi(xt).mean((1, 2))
    forest, sea = y == cl.index("Forest"), y == cl.index("SeaLake")
    assert v[forest].mean() > 0.5 and v[sea].mean() < 0
    assert w[sea].mean() > 0.3 and w[forest].mean() < 0
