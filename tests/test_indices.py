import numpy as np
from sentinel_eurosat.indices import BANDS, IDX, ndvi, ndwi, ndbi


def pix(**v):
    x = np.zeros((13, 2, 2), np.float32)
    for b, val in v.items():
        x[IDX[b]] = val
    return x


def test_band_order_has_13_unique():
    assert len(BANDS) == 13 and len(set(BANDS)) == 13 and BANDS[3] == "B04" and BANDS[-1] == "B8A"


def test_vegetation_is_positive_ndvi_water_is_negative():
    veg = pix(B04=500, B08=4000)
    water = pix(B03=1000, B04=600, B08=300)
    assert ndvi(veg).min() > 0.7
    assert ndvi(water).max() < 0
    assert ndwi(water).min() > 0.5 and ndwi(veg).max() < 0


def test_scale_invariant_and_zero_safe():
    a = pix(B04=500, B08=4000)
    assert np.allclose(ndvi(a), ndvi(a * 3.7))
    assert np.all(ndvi(pix()) == 0)  # no NaN on all-zero pixels


def test_builtup_positive_ndbi():
    assert ndbi(pix(B11=2500, B08=1500)).min() > 0
