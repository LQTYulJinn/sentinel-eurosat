"""EuroSAT_MS loader: reads the 13-band GeoTIFFs once into a cache with each patch's georeferenced centre and bounds."""
import io, zipfile, pathlib
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import transform as warp_transform

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "EuroSAT_MS.zip"
CACHE = ROOT / "cache" / "eurosat.npz"


def read_patch(raw: bytes):
    """-> (13,H,W) uint16 array, centre (lon, lat) in WGS84, EPSG code, transform."""
    with MemoryFile(raw) as mf, mf.open() as ds:
        x = ds.read()
        cx, cy = ds.transform * (ds.width / 2, ds.height / 2)
        lon, lat = warp_transform(ds.crs, "EPSG:4326", [cx], [cy])
        return x, lon[0], lat[0], ds.crs.to_epsg(), tuple(ds.bounds)


def build_cache(zip_path=ZIP, out=CACHE):
    out.parent.mkdir(exist_ok=True)
    y, lon, lat, epsg, names, bounds = [], [], [], [], [], []
    with zipfile.ZipFile(zip_path) as z:
        tifs = sorted(n for n in z.namelist() if n.lower().endswith(".tif"))
        X = np.lib.format.open_memmap(out.with_suffix(".X.npy"), "w+", np.uint16, (len(tifs), 13, 64, 64))
        for i, n in enumerate(tifs):
            X[i], lo, la, ep, bd = read_patch(z.read(n))
            lon.append(lo); lat.append(la); epsg.append(ep or 0); names.append(n); bounds.append(bd)
            y.append(pathlib.PurePosixPath(n).parent.name)
        X.flush()
    classes = sorted(set(y))
    np.savez(out, y=np.array([classes.index(c) for c in y]), classes=np.array(classes), lon=np.array(lon),
             lat=np.array(lat), epsg=np.array(epsg), names=np.array(names), bounds=np.array(bounds))


def load():
    if not CACHE.exists():
        build_cache()
    d = dict(np.load(CACHE))
    d["X"] = np.load(CACHE.with_suffix(".X.npy"), mmap_mode="r")  # 2.9 GB, kept on disk
    return d


def spatial_split(lon, lat, cell_deg=1.0, fracs=(0.7, 0.15, 0.15), seed=0):
    """Assign whole lon/lat cells to train/val/test so neighbouring patches of one scene never straddle splits."""
    cell = np.floor(lon / cell_deg).astype(int) * 100003 + np.floor(lat / cell_deg).astype(int)
    cells = np.unique(cell)
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    n = len(cells)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    part = {c: 0 for c in cells[:a]} | {c: 1 for c in cells[a:b]} | {c: 2 for c in cells[b:]}
    return np.array([part[c] for c in cell])
