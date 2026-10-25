"""Flux-model classes and feature engineering (kept importable so that pickled models load anywhere)."""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import optimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from epngcat.config import OUT, SEED

CH = "RGB"


def features(d):
    """Feature matrix for the flux models."""
    f = pd.DataFrame(index=d.index)
    for c in CH:
        for tag in ("_n", "_nw", "", "_w", "_s"):
            f[f"A_{c}{tag}"] = d.get(f"A_{c}{tag}", np.nan)
        for rad in (6, 10):
            f[f"ap{rad}_{c}"] = d[f"ap{rad}_{c}"]
        for k in ("bkg", "noise", "raw", "rawbkg", "sbkg"):
            f[f"{k}_{c}"] = d[f"{k}_{c}"]
        for k in ("H_A", "H_R50", "H_WIDTH", "H_CLIP_WIDTH", "H_OK"):
            f[f"{k}_{c}"] = d.get(f"{k}_{c}", np.nan)
    for k in (
        "width_n",
        "width",
        "chi2_n",
        "chi2_nw",
        "sat_area",
        "sat_pix",
        "snr",
        "jac_area",
        "jac_anis",
        "n_neigh",
        "neigh_dist",
        "halo_shift",
        "halo_ok",
    ):
        f[k] = d.get(k, np.nan)
    f["neigh_dist"] = f.neigh_dist.clip(upper=60)
    f["abs_b"] = np.abs(d.b.values)
    return f.astype(float)


def slog(x, s=1e-3):
    """Signed log transform."""
    return np.sign(x) * np.log10(1 + np.abs(x) / s)


class ParametricFlux:
    """log F = a + b * log10(sum_c k_c X_c) for X = PSF amplitudes or aperture sums."""

    def __init__(self, cols):
        self.cols = cols

    def _pred(self, p, x):
        s = x @ np.exp(p[2:])
        return p[0] + p[1] * np.log10(np.clip(s, 1e-6, None))

    def fit(self, f, y):
        """Fit the model to features f and log fluxes y."""
        x = np.nan_to_num(f[self.cols].values)
        self.p = optimize.least_squares(
            lambda p: self._pred(p, x) - y, [-12.0, 1.0, 0.0, 0.0, 0.0], loss="soft_l1", f_scale=0.2
        ).x
        return self

    def predict(self, f):
        """Predict log fluxes for features f."""
        return self._pred(self.p, np.nan_to_num(f[self.cols].values))


class RidgeFlux:
    """Ridge regression on signed-log features."""

    def fit(self, f, y):
        """Fit the model to features f and log fluxes y."""
        self.cols = f.columns
        self.m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        self.m.fit(self._x(f), y)
        return self

    def _x(self, f):
        x = f[self.cols].copy()
        for c in x:
            if c.startswith(("A_", "ap")):
                x[c] = slog(x[c])
        return x.fillna(x.median()).fillna(0).values

    def predict(self, f):
        """Predict log fluxes for features f."""
        return self.m.predict(self._x(f))


def internal_split(d):
    """Tile-block based 80/20 split of the training set (for early stopping / tuning)."""
    blocks = pd.read_parquet(OUT / "04_tiles" / "tiles.parquet").set_index("tile").block
    b = blocks.reindex(d.tile).values
    ub = np.unique(b)
    pick = set(np.random.default_rng(SEED).choice(ub, max(1, len(ub) // 5), replace=False))
    return np.array([x in pick for x in b])


class XGBFlux:
    """XGBoost with a small hyper-parameter search on an internal train split."""

    GRID = [
        {"max_depth": d, "min_child_weight": w, "objective": o}
        for d in (4, 6)
        for w in (5, 20)
        for o in ("reg:squarederror", "reg:pseudohubererror")
    ]

    def fit(self, f, y, inner):
        """Fit the model to features f and log fluxes y.

        Each GRID setting is trained on ~inner with early stopping on `inner` (bool mask) and scored
        by the robust (MAD) scatter of its `inner` residuals; the best setting is then refitted on
        all rows with the early-stopped number of trees.
        """
        best = None
        for g in self.GRID:
            m = xgb.XGBRegressor(
                n_estimators=3000,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                early_stopping_rounds=100,
                random_state=SEED,
                n_jobs=8,
                **({"huber_slope": 0.2} if "huber" in g["objective"] else {}),
                **g,
            )
            m.fit(f[~inner], y[~inner], eval_set=[(f[inner], y[inner])], verbose=False)
            r = y[inner] - m.predict(f[inner])
            s = 1.4826 * np.median(np.abs(r - np.median(r)))
            if best is None or s < best[0]:
                best = (s, g, m.best_iteration)
        self.params = {**best[1], "n_estimators": best[2] + 1}
        self.m = xgb.XGBRegressor(
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=8,
            **({"huber_slope": 0.2} if "huber" in best[1]["objective"] else {}),
            **self.params,
        )
        self.m.fit(f, y)
        return self

    def predict(self, f):
        """Predict log fluxes for features f."""
        return self.m.predict(f)


class HGBFlux:
    """sklearn HistGradientBoosting with a small search on an internal train split."""

    GRID = [{"loss": lo, "max_leaf_nodes": n} for lo in ("squared_error", "absolute_error") for n in (15, 31)]

    def fit(self, f, y, inner):
        """Fit the model to features f and log fluxes y; `inner` selects the rows that score GRID."""
        best = None
        for g in self.GRID:
            m = HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=2000, early_stopping=False, random_state=SEED, **g
            )
            m.fit(f[~inner], y[~inner])
            # Staged predictions give the best iteration count on the inner split.
            scores = []
            for k, p in enumerate(m.staged_predict(f[inner])):
                if k % 25 == 0:
                    r = y[inner] - p
                    scores.append((1.4826 * np.median(np.abs(r - np.median(r))), k + 1))
            s, it = min(scores)
            if best is None or s < best[0]:
                best = (s, g, it)
        self.params = {**best[1], "max_iter": best[2]}
        self.m = HistGradientBoostingRegressor(
            learning_rate=0.05, early_stopping=False, random_state=SEED, **self.params
        )
        self.m.fit(f, y)
        return self

    def predict(self, f):
        """Predict log fluxes for features f."""
        return self.m.predict(f)
