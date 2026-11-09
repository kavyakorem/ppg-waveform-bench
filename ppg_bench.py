"""Deterministic, simulation-only optical waveform benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

FS, SECONDS = 64, 16
FEATURES = [
    "peak_bpm", "spectral_bpm", "autocorr_bpm", "band_dominance",
    "interval_cv", "peak_prominence", "missing_fraction", "rail_fraction",
    "pulsatile_fraction", "width_ms", "cycle_agreement", "rise_slope",
]


def simulate(seed: int, subjects: int, windows: int):
    """Draw independent virtual subjects with shared within-subject morphology."""
    rng = np.random.default_rng(seed)
    t = np.arange(FS * SECONDS) / FS
    for subject in range(subjects):
        base_rate = rng.uniform(54, 132)
        width = rng.uniform(0.07, 0.14)
        notch = rng.uniform(0.08, 0.31)
        gain = rng.uniform(0.55, 1.6)
        for w in range(windows):
            mode = ("clean", "drift", "motion", "dropout", "mixed")[w % 5]
            bpm = np.clip(base_rate + rng.normal(0, 6), 45, 145)
            f = bpm / 60
            phase = (f*t + rng.random() + 0.045*np.sin(2*np.pi*0.17*t)) % 1
            pulse = phase**2 * np.exp(-phase/width)
            pulse = pulse/pulse.max() + notch*np.exp(-((phase-0.57)/0.09)**2)
            x = gain*pulse*(1 + 0.13*np.sin(2*np.pi*0.26*t)) + rng.normal(0, .025, t.size)
            if mode in {"drift", "mixed"}:
                x += rng.uniform(0.2, 0.7)*np.sin(2*np.pi*0.24*t+rng.random())
            if mode in {"motion", "mixed"}:
                mf = rng.uniform(.7, 2.65)
                start = rng.uniform(1, 7)
                envelope = np.exp(-0.5*((t-start-3)/rng.uniform(1.2, 4.2))**2)
                x += envelope*rng.uniform(.16, .68)*np.sin(2*np.pi*mf*t+rng.random())
                x += gaussian_filter1d(rng.normal(0, .45, t.size), sigma=1.2)*envelope
            if mode in {"dropout", "mixed"}:
                start = rng.integers(FS, x.size-2*FS)
                x[start:start+rng.integers(FS//4, FS*2)] = np.nan
                x = np.minimum(x, np.nanquantile(x, rng.uniform(.81, .97)))
            yield subject, w, mode, float(bpm), x


def _spectral_rate(freq, power):
    bands = (freq >= .7) & (freq <= 2.6)
    f, p = freq[bands], power[bands]
    i = int(np.argmax(p))
    concentration = p[i] / max(np.sum(p), 1e-12)
    # Three-bin quadratic interpolation reduces discretization error.
    offset = 0.0
    if 0 < i < len(p)-1:
        a, b, c = np.log(np.maximum(p[i-1:i+2], 1e-12))
        denominator = a - 2*b + c
        if denominator < -1e-12:
            offset = float(np.clip(.5*(a-c)/denominator, -.5, .5))
    return float(60*(f[i]+offset*(f[1]-f[0]))), float(concentration)


def extract(raw):
    x = np.asarray(raw, dtype=float)
    if x.shape != (FS*SECONDS,):
        raise ValueError(f"expected {FS*SECONDS} samples at {FS} Hz")
    good = np.isfinite(x)
    missing = 1 - good.mean()
    if good.sum() < FS*4 or np.ptp(x[good]) < 1e-8:
        return dict.fromkeys(FEATURES, np.nan) | {
            "missing_fraction": float(missing), "rail_fraction": 1.0
        }
    samples = np.arange(len(x))
    filled = np.interp(samples, samples[good], x[good])
    rail = np.isclose(x[good], np.max(x[good]), rtol=0, atol=1e-9).sum()/len(x)
    clean = sosfiltfilt(butter(3, [.55, 5], fs=FS, btype="bandpass", output="sos"), filled)
    scale = np.median(np.abs(clean-np.median(clean)))
    y = (clean-np.median(clean))/max(scale, 1e-9)
    f, p = welch(y, FS, nperseg=len(y), nfft=4096)
    spectral, dominance = _spectral_rate(f, p)
    peaks, props = find_peaks(y, distance=int(FS*.34), prominence=.66, width=1)
    intervals = np.diff(peaks) / FS
    plausible = intervals[(intervals >= .37) & (intervals <= 1.46)]
    if plausible.size >= 3:
        peak_rate = 60 / np.median(plausible)
        interval_cv = np.std(plausible) / np.mean(plausible)
    else:
        peak_rate, interval_cv = np.nan, np.nan
    yy = y-y.mean()
    autocorrelation = np.correlate(yy, yy, mode="full")[len(yy)-1:]
    lo, hi = round(FS/2.6), round(FS/.70)
    autocorr_rate = 60*FS/(lo+np.argmax(autocorrelation[lo:hi+1]))
    diffs = np.abs(np.array([spectral, autocorr_rate, peak_rate])-spectral)
    return dict(zip(FEATURES, map(float, [
        peak_rate, spectral, autocorr_rate, dominance, interval_cv,
        np.median(props["prominences"]) if len(peaks) else np.nan, missing, rail,
        np.var(clean)/max(np.var(filled), 1e-12),
        np.median(props["widths"])*1000/FS if len(peaks) else np.nan,
        np.nanmax(diffs), np.quantile(np.diff(y), .95)*FS,
    ])))


def split_groups(groups, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    trainval, test = next(splitter.split(groups, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=1/3, random_state=seed+1)
    train, val = next(inner.split(trainval, groups=groups[trainval]))
    return trainval[train], trainval[val], test


def _bootstrap_ci(errors, ids, seed):
    buckets = [errors[ids == s] for s in np.unique(ids)]
    rng = np.random.default_rng(seed)
    values = [np.mean(np.concatenate([buckets[i] for i in rng.integers(0, len(buckets), len(buckets))]))
              for _ in range(400)]
    return [float(v) for v in np.quantile(values, [.025, .975])]


def _scores(reference, predicted):
    good = np.isfinite(predicted)
    return {
        "n": int(good.sum()), "coverage": float(good.mean()),
        "mae_bpm": float(np.abs(reference[good]-predicted[good]).mean()) if good.any() else None,
        "p90_bpm": float(np.quantile(np.abs(reference[good]-predicted[good]), .9)) if good.any() else None,
    }


def run(seed: int, subjects: int, windows: int, output: Path):
    if subjects < 12 or windows < 5:
        raise ValueError("need at least 12 virtual subjects and five windows each")
    df = pd.DataFrame([
        {"subject": s, "window": w, "artifact": mode, "reference_bpm": label, **extract(x)}
        for s,w,mode,label,x in simulate(seed, subjects, windows)
    ])
    x, y = df[FEATURES], df.reference_bpm.to_numpy()
    ids = df.subject.to_numpy()
    train, val, test = split_groups(ids, seed)
    model = HistGradientBoostingRegressor(max_iter=115, max_leaf_nodes=15,
                                          learning_rate=.075, random_state=seed)
    model.fit(x.iloc[train], y[train])
    pred = model.predict(x)
    # Select the uncertainty rule using validation groups only.
    quality = (df.band_dominance - .018*df.cycle_agreement.clip(upper=35)
               - 1.5*df.missing_fraction - df.rail_fraction).to_numpy()
    candidates = [-np.inf, *np.quantile(quality[val], np.linspace(.05, .25, 9))]
    eligible = [q for q in candidates if (quality[val] >= q).mean() >= .80]
    gate = min(eligible, key=lambda q: np.abs(y[val][quality[val] >= q]-pred[val][quality[val] >= q]).mean())
    spectral = df.spectral_bpm.to_numpy()
    # Route stable windows to Welch and ambiguous windows to the learned model.
    # The candidate extremes also let validation select either model for all windows.
    def routed(threshold):
        return np.where(quality >= threshold, spectral, pred)
    routing_options = [-np.inf, *np.quantile(quality[val], np.linspace(.05, .95, 19)), np.inf]
    route_at = min(routing_options, key=lambda q: np.abs(y[val]-routed(q)[val]).mean())
    hybrid = routed(route_at)
    method = {
        "peak": df.peak_bpm.to_numpy(),
        "welch": spectral,
        "gradient_boosting": pred,
        "gated_gradient_boosting": np.where(quality >= gate, pred, np.nan),
        "quality_routed": hybrid,
    }
    by_artifact = {}
    for artifact in df.artifact.unique():
        ix = test[df.artifact.iloc[test].to_numpy() == artifact]
        by_artifact[artifact] = {key: _scores(y[ix], values[ix]) for key,values in method.items()}
    error = np.abs(y[test]-pred[test])
    accepted = test[quality[test] >= gate]
    summary = {
        "data": "synthetic only", "seed": seed, "virtual_subjects": subjects, "windows": len(df),
        "test_virtual_subjects": int(len(np.unique(ids[test]))),
        "split_subjects": {name: sorted(np.unique(ids[ix]).tolist())
                           for name,ix in (("train",train),("validation",val),("test",test))},
        "validation_gate": None if np.isneginf(gate) else float(gate),
        "validation_coverage": float((quality[val] >= gate).mean()),
        "routing_threshold": float(route_at) if np.isfinite(route_at) else None,
        "routing_policy": "all_welch" if np.isneginf(route_at) else
                          "all_gradient_boosting" if np.isposinf(route_at) else "quality_threshold",
        "test_fraction_routed_to_welch": float((quality[test] >= route_at).mean()),
        "test_metrics": {name: _scores(y[test], values[test]) for name,values in method.items()},
        "test_same_accepted_windows": {
            name: _scores(y[accepted], values[accepted]) for name,values in method.items()
        } if accepted.size else {},
        "test_gradient_boosting_cluster_bootstrap_mae_ci95_bpm": _bootstrap_ci(error, ids[test], seed),
        "by_artifact": by_artifact,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    results = df.iloc[test][["subject","window","artifact","reference_bpm"]].copy()
    for name, values in method.items():
        results[name+"_bpm"] = values[test]
    results.assign(quality_score=quality[test]).to_csv(output/"heldout_predictions.csv", index=False)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--subjects", type=int, default=80)
    p.add_argument("--windows", type=int, default=10, help="windows per virtual subject")
    p.add_argument("--output", type=Path, default=Path("results"))
    args = p.parse_args()
    print(json.dumps(run(args.seed,args.subjects,args.windows,args.output)["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
