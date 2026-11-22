"""Spectral indices for Sentinel-2 L1C patches. x: (13, H, W), band order below."""
import numpy as np

# EuroSAT_MS files carry no band metadata. The order is alphanumeric (B8A last), NOT spectral order; it was
# identified from the data (see tests/test_real_data.py): B10 is ~0 everywhere, B8A >= B08, B12 < B11.
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
IDX = {b: i for i, b in enumerate(BANDS)}
RGB = [IDX["B04"], IDX["B03"], IDX["B02"]]


def nd(a, b):
    """Normalised difference (a-b)/(a+b); 0 where both are 0. Scale-invariant, so raw DN and reflectance agree."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    s = a + b
    return np.divide(a - b, s, out=np.zeros_like(s), where=s != 0)


def ndvi(x):  # vegetation: (NIR - Red) / (NIR + Red)
    return nd(x[IDX["B08"]], x[IDX["B04"]])


def ndwi(x):  # open water (McFeeters): (Green - NIR) / (Green + NIR)
    return nd(x[IDX["B03"]], x[IDX["B08"]])


def ndbi(x):  # built-up: (SWIR1 - NIR) / (SWIR1 + NIR)
    return nd(x[IDX["B11"]], x[IDX["B08"]])
