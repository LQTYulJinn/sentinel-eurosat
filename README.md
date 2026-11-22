# sentinel-eurosat

Does using all 13 Sentinel-2 bands help a land-use classifier compared with RGB only? Practice project for
multispectral remote sensing: reading georeferenced GeoTIFFs with rasterio, spectral indices, georeferencing, and a
controlled RGB vs multispectral comparison.

**Data:** EuroSAT_MS (Helber et al.), 27,000 Sentinel-2 L1C patches, 64x64 px (~10 m), 13 bands, 10 classes, 34 European
countries, UTM zones 27N-36N. **Task:** patch classification. EuroSAT has no segmentation masks, so this is *not* a U-Net.

## Results (test set, mean ± std over 3 seeds)

| input | channels | accuracy | macro-F1 |
|---|---|---|---|
| RGB (B04, B03, B02) | 3 | 0.914 ± 0.003 | 0.905 ± 0.003 |
| RGB + NDVI + NDWI + NDBI | 6 | 0.927 ± 0.006 | 0.921 ± 0.005 |
| **All 13 bands** | 13 | **0.954 ± 0.004** | **0.951 ± 0.004** |
| gradient boosting, patch mean/std of RGB (1 run) | - | 0.729 | 0.691 |
| gradient boosting, patch mean/std of 13 bands + indices (1 run) | - | 0.848 | 0.828 |

Same CNN body (100k parameters, 4 conv blocks), same split, same seeds, same 10 epochs; only the input channels differ.
13 bands beat RGB by +0.040 to +0.050 macro-F1 in every seed. Adding three indices to RGB gives about +0.016.
Largest per-class gains (F1, RGB to 13 bands): PermanentCrop 0.79 to 0.92, River 0.87 to 0.96, AnnualCrop 0.85 to 0.92.
**Forest got slightly worse** (0.979 to 0.966). Full tables: `reports/results.md`.

## What it covers
- **GeoTIFF + georeferencing** (`data.py`, `geo.py`): CRS, affine transform, patch centre and footprint reprojected to
  WGS84, GeoJSON export of every test patch with true/predicted class (`reports/test_predictions_ms13.geojson`, open in
  QGIS or geojson.io). Pixel size is ~9.99 x 9.99 m, not exactly 10 m. Tests check the maths against hand-computed UTM values.
- **Spectral indices** (`indices.py`): NDVI, NDWI, NDBI. Class means behave as expected (Forest NDVI 0.72, SeaLake NDWI 0.46).
- **Spatial split**: whole 1x1 degree lon/lat cells go to train/val/test (17,712 / 3,856 / 5,432 patches). EuroSAT cuts many
  neighbouring patches out of one scene, so a random split leaks. Validation picks the epoch; test is read once per run.

## A bug caught on the way
The files carry no band metadata. I first assumed the spectral order (B01..B08, B8A, B09..B12). Real spectra said
otherwise (band 9 was ~0 everywhere, the last band exceeded NIR). The order is alphanumeric: **B01..B09, B10, B11, B12, B8A**.
Found before any training, and now guarded by `tests/test_real_data.py`.

## Limitations
- One spatial split. Seeds vary initialisation and batch order only, so the std does not capture split variance.
  Validation class counts are uneven (130 to 607 per class), so validation macro-F1 is noisy.
- Three seeds, small CNN trained from scratch for 10 epochs, no tuning. Published EuroSAT numbers (~98%) use random
  splits and pretrained networks, so they are not comparable with these.
- Classification only. No segmentation, SAR, Landsat, hyperspectral or atmospheric correction (L1C top-of-atmosphere).
- The two gradient-boosting rows are single runs.

## Run it
```bash
pip install -r requirements.txt
# download EuroSAT_MS.zip (2 GB) from https://zenodo.org/records/7711810 into data/
python -m pytest -q tests                       # 10 tests (2 need the cache)
python -m sentinel_eurosat.train --epochs 10    # ~1 h on an 8-core CPU, no GPU needed
python -m sentinel_eurosat.report
```
