import numpy as np, rasterio
from rasterio.transform import from_origin
from sentinel_eurosat.data import read_patch, spatial_split
from sentinel_eurosat.geo import footprint_lonlat


def make_tif(path, x0=500000.0, y0=5000000.0, epsg=32632):
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=13, dtype="uint16",
                       crs=f"EPSG:{epsg}", transform=from_origin(x0, y0, 10, 10)) as ds:
        ds.write(np.arange(13, dtype="uint16")[:, None, None] * np.ones((13, 64, 64), "uint16"))


def test_read_patch_georeference(tmp_path):
    p = tmp_path / "a.tif"; make_tif(p)
    x, lon, lat, epsg, bounds = read_patch(p.read_bytes())
    assert x.shape == (13, 64, 64) and x[5, 0, 0] == 5  # bands keep their order
    assert epsg == 32632 and bounds == (500000.0, 4999360.0, 500640.0, 5000000.0)  # 64 px * 10 m = 640 m
    assert abs(lon - 9.0) < 0.01       # UTM 32N central meridian is 9 E; x=500320 is ~320 m east of it
    assert abs(lat - 45.151) < 0.002   # 45 N is 4,982,950 m (k0 * arc length); +16,730 m / 111,090 m per deg = 45.151


def test_footprint_is_640m_square_in_lonlat():
    ring = footprint_lonlat((500000.0, 4999360.0, 500640.0, 5000000.0), 32632)
    assert ring[0] == ring[-1] and len(ring) == 5
    lons, lats = np.array(ring)[:, 0], np.array(ring)[:, 1]
    # 640 m north-south = 0.00575 deg; east-west at 45 N = 640 / (111320 * cos 45) deg
    assert abs((lats.max() - lats.min()) - 640 / 110950) < 2e-4
    assert abs((lons.max() - lons.min()) - 640 / (111320 * np.cos(np.radians(45)))) < 3e-4


def test_pixel_roundtrip(tmp_path):
    p = tmp_path / "b.tif"; make_tif(p)
    with rasterio.open(p) as ds:
        x, y = ds.xy(10, 20)
        assert ds.index(x, y) == (10, 20)


def test_spatial_split_never_shares_a_cell():
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(-10, 30, 5000), rng.uniform(36, 60, 5000)
    s = spatial_split(lon, lat, cell_deg=1.0)
    cell = np.floor(lon).astype(int) * 1000 + np.floor(lat).astype(int)
    for c in np.unique(cell):
        assert len(set(s[cell == c])) == 1
    assert set(s) == {0, 1, 2} and 0.6 < (s == 0).mean() < 0.8
