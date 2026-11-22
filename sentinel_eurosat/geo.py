"""Georeferencing helpers: patch footprints in WGS84 and a GeoJSON export (open in QGIS or geojson.io)."""
import json
from rasterio.warp import transform as warp_transform


def footprint_lonlat(bounds, epsg):
    """Native-CRS bounds (left, bottom, right, top) -> closed lon/lat ring."""
    l, b, r, t = bounds
    xs, ys = [l, r, r, l, l], [b, b, t, t, b]
    lon, lat = warp_transform(f"EPSG:{epsg}", "EPSG:4326", xs, ys)
    return [[float(a), float(c)] for a, c in zip(lon, lat)]


def to_geojson(d, idx, true, pred, path):
    feats = [{"type": "Feature",
              "properties": {"name": str(d["names"][i]), "true": str(d["classes"][true[k]]),
                             "pred": str(d["classes"][pred[k]]), "correct": bool(true[k] == pred[k]),
                             "epsg": int(d["epsg"][i])},
              "geometry": {"type": "Polygon", "coordinates": [footprint_lonlat(d["bounds"][i], int(d["epsg"][i]))]}}
             for k, i in enumerate(idx)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
