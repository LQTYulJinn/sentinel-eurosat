"""RGB vs multispectral CNN on EuroSAT, identical architecture, spatial split, test read once per run."""
import argparse, json, time, pathlib
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from .data import load, spatial_split
from .indices import RGB, ndvi, ndwi, ndbi
from .model import Net

OUT = pathlib.Path(__file__).resolve().parents[1] / "reports"
CH = {"rgb": 3, "rgb_idx": 6, "ms13": 13}  # rgb_idx = RGB + NDVI + NDWI + NDBI


def channels(xb, variant):
    """xb: (N,13,H,W) uint16 -> float32 (N,C,H,W) reflectance-like (DN/10000)."""
    x = xb.astype(np.float32) / 10000.0
    if variant == "ms13":
        return x
    rgb = x[:, RGB]
    if variant == "rgb":
        return rgb
    xt = x.transpose(1, 0, 2, 3)  # indices index the band axis first
    return np.concatenate([rgb, np.stack([ndvi(xt), ndwi(xt), ndbi(xt)], 1)], 1)


def augment(x, rng):  # one random flip/rot per batch: valid for nadir imagery
    x = np.rot90(x, rng.integers(4), (2, 3))
    if rng.random() < 0.5:
        x = x[..., ::-1]
    return np.ascontiguousarray(x)


def predict(net, X, variant, mean, std, bs=512):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = (channels(X[i:i + bs], variant) - mean) / std
            out.append(net(torch.from_numpy(xb)).argmax(1).numpy())
    return np.concatenate(out)


def run(d, split, variant, seed, epochs, bs=128):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X, y = d["X"], d["y"]
    tr, va, te = (np.where(split == k)[0] for k in range(3))
    sample = channels(X[rng.choice(tr, 2000, replace=False)], variant)
    mean = sample.mean((0, 2, 3), keepdims=True); std = sample.std((0, 2, 3), keepdims=True) + 1e-6
    net = Net(CH[variant]); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state = -1, None
    for ep in range(epochs):
        net.train(); perm = rng.permutation(tr)
        for i in range(len(tr) // bs):
            idx = np.sort(perm[i * bs:(i + 1) * bs])
            xb = augment((channels(X[idx], variant) - mean) / std, rng)
            loss = lossf(net(torch.from_numpy(xb)), torch.from_numpy(y[idx]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        f1 = f1_score(y[va], predict(net, X[va], variant, mean, std), average="macro")
        if f1 > best:  # model selection on validation only
            best, best_state = f1, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    pred = predict(net, X[te], variant, mean, std)  # test read once
    np.save(OUT / f"pred_{variant}_s{seed}.npy", pred)
    return dict(variant=variant, seed=seed, params=sum(p.numel() for p in net.parameters()), val_macro_f1=best,
                test_acc=accuracy_score(y[te], pred), test_macro_f1=f1_score(y[te], pred, average="macro"))


def baseline(d, split):
    """Classical reference: per-patch mean/std of every band (+ index means) -> gradient boosting."""
    def feats(xb):
        x = xb.astype(np.float32) / 10000; xt = x.transpose(1, 0, 2, 3)
        idx = np.stack([ndvi(xt).mean((1, 2)), ndwi(xt).mean((1, 2)), ndbi(xt).mean((1, 2))], 1)
        return np.concatenate([x.mean((2, 3)), x.std((2, 3)), idx], 1)  # 13 means, 13 stds, 3 index means
    F = np.concatenate([feats(d["X"][i:i + 2000]) for i in range(0, len(d["y"]), 2000)])
    rgb_cols = RGB + [13 + c for c in RGB]
    sets = {"hgb_rgb_stats": F[:, rgb_cols], "hgb_ms_stats+indices": F}
    y, res = d["y"], []
    for name, G in sets.items():
        m = HistGradientBoostingClassifier(random_state=0).fit(G[split == 0], y[split == 0])
        p = m.predict(G[split == 2])
        res.append(dict(variant=name, seed=0, test_acc=accuracy_score(y[split == 2], p),
                        test_macro_f1=f1_score(y[split == 2], p, average="macro")))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["rgb", "rgb_idx", "ms13"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N patches")
    a = ap.parse_args()
    torch.set_num_threads(8); OUT.mkdir(exist_ok=True)
    d = load()
    if a.limit:
        sel = np.random.default_rng(0).choice(len(d["y"]), a.limit, replace=False)
        sel.sort(); d = {k: v[sel] if k in ("X", "y", "lon", "lat") else v for k, v in d.items()}
    split = spatial_split(d["lon"], d["lat"])
    print("split sizes", [int((split == k).sum()) for k in range(3)], flush=True)
    results = baseline(d, split)
    for r in results: print(r, flush=True)
    for v in a.variants:
        for s in a.seeds:
            t = time.time(); r = run(d, split, v, s, a.epochs); r["seconds"] = round(time.time() - t)
            print(r, flush=True); results.append(r)
            (OUT / "runs.json").write_text(json.dumps(results, indent=1))
