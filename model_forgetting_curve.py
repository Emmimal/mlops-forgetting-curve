"""
model_forgetting_curve.py
=========================
Measure how quickly your production model loses learned patterns —
and use it to decide exactly when to retrain.

Inspired by the Ebbinghaus Forgetting Curve (1885), adapted for ML systems.

The tracker fits:
    R(t) = R_baseline * exp(-lambda * t)

where  R(t)    = metric at time t (days since deployment)
       lambda  = decay rate  (higher = faster forgetting)

Two forgetting regimes are detected automatically:
  - SMOOTH  : classic exponential decay (high R²) → gradual drift
  - EPISODIC: step-change / shock-driven (low R²)  → sudden concept drift

Author  : [Your Name]
License : MIT
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Wedge, FancyArrowPatch
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.animation as animation
    _MPL = True
except ImportError:
    _MPL = False
    warnings.warn("matplotlib not found - visualisations disabled.")

try:
    from scipy.optimize import curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("scipy not found - using numpy OLS fallback.")


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance observation."""
    timestamp:    datetime
    metric_value: float
    n_samples:    int = 0
    metadata:     Dict = field(default_factory=dict)


@dataclass
class ForgettingReport:
    """Full diagnostic report from ModelForgettingTracker.report()."""
    baseline_metric:             float
    current_metric:              float
    retention_ratio:             float
    predicted_days_to_threshold: Optional[float]
    retrain_recommended:         bool
    decay_rate:                  float
    half_life_days:              Optional[float]
    forgetting_speed:            str
    forgetting_regime:           str          # "smooth" | "episodic" | "unknown"
    snapshots_used:              int
    fit_r_squared:               Optional[float]
    recommended_retrain_date:    Optional[datetime]
    worst_drop_week:             Optional[int]   # 1-based week NUMBER where the drop lands
    worst_drop_magnitude:        Optional[float] # absolute recall drop into that week


# ─────────────────────────────────────────────────────────────────────────────
# Core tracker
# ─────────────────────────────────────────────────────────────────────────────

class ModelForgettingTracker:
    """
    Track and quantify how quickly a production ML model's performance
    decays over time using an exponential forgetting-curve model.

    Automatically distinguishes between:
      - Smooth / gradual drift  (R² > 0.4)  → classic Ebbinghaus regime
      - Episodic / shock-driven (R² ≤ 0.4)  → sudden concept-drift regime

    Parameters
    ----------
    metric_name      : Human-readable name (e.g. "F1", "AUC").
    higher_is_better : Whether higher metric value means better performance.
    retrain_threshold: Fractional drop from baseline triggering retrain alert.
                       Default 0.07 = alert when metric drops >= 7%.
    window_size      : Rolling window for smoothing.
    baseline_window  : Initial snapshots used to establish baseline.
    smooth_r2_cutoff : R² above which forgetting is classified as "smooth".
    baseline_method  : How to compute baseline from the first baseline_window
                       snapshots. Options:
                         "mean"   — arithmetic mean (default, robust)
                         "max"    — peak performance (conservative, catches
                                    any subsequent drop immediately)
                         "top3"   — mean of the top-3 values (balanced:
                                    ignores warm-up noise, tracks true peak)

    Quick start
    -----------
    >>> tracker = ModelForgettingTracker(metric_name="F1", retrain_threshold=0.07)
    >>> tracker.log(0.923)
    >>> tracker.log(0.917)
    >>> report = tracker.report()
    >>> figs = tracker.plot(save_dir="./charts")
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """

    def __init__(
        self,
        metric_name:       str   = "metric",
        higher_is_better:  bool  = True,
        retrain_threshold: float = 0.07,
        window_size:       int   = 30,
        baseline_window:   int   = 5,
        smooth_r2_cutoff:  float = 0.40,
        baseline_method:   str   = "mean",   # "mean" | "max" | "top3"
    ) -> None:
        if baseline_method not in ("mean", "max", "top3"):
            raise ValueError(
                f"baseline_method must be 'mean', 'max', or 'top3', "
                f"got {baseline_method!r}"
            )
        self.metric_name       = metric_name
        self.higher_is_better  = higher_is_better
        self.retrain_threshold = retrain_threshold
        self.window_size       = window_size
        self.baseline_window   = baseline_window
        self.smooth_r2_cutoff  = smooth_r2_cutoff
        self.baseline_method   = baseline_method
        self._snapshots:      List[PerformanceSnapshot] = []
        self._retrain_events: List[datetime]            = []

    # ── public API ────────────────────────────────────────────────────────────

    def log(
        self,
        metric_value: float,
        timestamp:    Optional[datetime] = None,
        n_samples:    int = 0,
        **metadata,
    ) -> "ModelForgettingTracker":
        """
        Record a new performance observation.

        Parameters
        ----------
        metric_value : Observed metric (e.g. 0.92 for 92% F1).
        timestamp    : UTC observation time; defaults to now.
        n_samples    : Ground-truth labels available in this window.
        **metadata   : Any extra key/value pairs stored on the snapshot.

        Returns self for fluent chaining.
        """
        if not np.isfinite(metric_value):
            raise ValueError(f"metric_value must be finite, got {metric_value!r}")
        self._snapshots.append(PerformanceSnapshot(
            timestamp    = timestamp or datetime.now(timezone.utc),
            metric_value = float(metric_value),
            n_samples    = n_samples,
            metadata     = metadata,
        ))
        return self

    def mark_retrain(
        self,
        timestamp: Optional[datetime] = None,
    ) -> "ModelForgettingTracker":
        """Record that a retrain / redeployment occurred."""
        self._retrain_events.append(timestamp or datetime.now(timezone.utc))
        return self

    def report(self) -> ForgettingReport:
        """
        Compute and return a full ForgettingReport.

        Raises ValueError if fewer than baseline_window snapshots logged.
        """
        snaps = self._snapshots
        if len(snaps) < self.baseline_window:
            raise ValueError(
                f"Need at least {self.baseline_window} snapshots "
                f"(have {len(snaps)})."
            )

        # ── Baseline computation (configurable method) ────────────────────
        baseline_vals = [s.metric_value for s in snaps[:self.baseline_window]]
        if self.baseline_method == "max":
            baseline = float(np.max(baseline_vals))
        elif self.baseline_method == "top3":
            k = min(3, len(baseline_vals))
            baseline = float(np.mean(sorted(baseline_vals, reverse=True)[:k]))
        else:  # "mean" (default)
            baseline = float(np.mean(baseline_vals))

        current   = snaps[-1].metric_value
        retention = current / baseline if baseline != 0 else 1.0

        t_days, y_norm = self._build_series(baseline)
        lam, r2        = self._fit_decay(t_days, y_norm)

        # ── Classify forgetting regime ────────────────────────────────────
        if r2 is None:
            regime = "unknown"
        elif r2 >= self.smooth_r2_cutoff:
            regime = "smooth"
        else:
            regime = "episodic"

        half_life     = np.log(2) / lam if lam > 1e-9 else None
        threshold_ret = 1.0 - self.retrain_threshold

        if lam > 1e-9:
            days_to_thresh = -np.log(threshold_ret) / lam
            elapsed        = (snaps[-1].timestamp - snaps[0].timestamp).total_seconds() / 86_400
            days_remaining = max(0.0, days_to_thresh - elapsed)
        else:
            days_remaining = None

        # ── Retrain recommendation ────────────────────────────────────────
        retrain_recommended = retention <= threshold_ret

        if   lam < 0.005: speed = "stable"
        elif lam < 0.02:  speed = "slow"
        elif lam < 0.06:  speed = "moderate"
        else:             speed = "fast"

        retrain_date = None
        if days_remaining is not None and days_remaining > 0:
            retrain_date = datetime.now(timezone.utc) + timedelta(days=days_remaining)

        # ── Worst single-window drop ──────────────────────────────────────
        # drops[i] = vals[i] - vals[i+1]
        #   → this is the fall FROM week (i+1) INTO week (i+2)  [1-based]
        #
        # BUG FIX: the original code set worst_drop_week = worst_idx + 1,
        # which labelled the HIGH week before the cliff as "worst".
        # The correct label is the week where the drop *lands*:
        #   worst_drop_week = worst_idx + 2   (1-based, destination week)
        #
        # The chart annotation must then index arrays with
        #   wi = worst_drop_week - 1          (0-based)
        # so it places the arrow at the low point, not the peak.
        vals  = [s.metric_value for s in snaps]
        drops = [vals[i-1] - vals[i] for i in range(1, len(vals))]
        if drops:
            worst_idx = int(np.argmax(drops))
            worst_mag = float(drops[worst_idx])
            # +2: worst_idx is 0-based in drops[], drops[i] represents the
            # fall that *arrives* at snapshot index i+1 → week number i+2.
            worst_week = worst_idx + 2
        else:
            worst_week = None
            worst_mag  = None

        return ForgettingReport(
            baseline_metric             = baseline,
            current_metric              = current,
            retention_ratio             = retention,
            predicted_days_to_threshold = days_remaining,
            retrain_recommended         = retrain_recommended,
            decay_rate                  = lam,
            half_life_days              = half_life,
            forgetting_speed            = speed,
            forgetting_regime           = regime,
            snapshots_used              = len(snaps),
            fit_r_squared               = r2,
            recommended_retrain_date    = retrain_date,
            worst_drop_week             = worst_week,
            worst_drop_magnitude        = worst_mag,
        )

    def dataframe(self) -> pd.DataFrame:
        """Return all snapshots as a tidy pandas DataFrame."""
        if not self._snapshots:
            return pd.DataFrame()
        t0   = self._snapshots[0].timestamp
        rows = []
        for s in self._snapshots:
            elapsed = (s.timestamp - t0).total_seconds() / 86_400
            rows.append({
                "timestamp":    s.timestamp,
                "elapsed_days": elapsed,
                "metric_value": s.metric_value,
                "n_samples":    s.n_samples,
                **s.metadata,
            })
        return pd.DataFrame(rows)

    # ── theme helpers ─────────────────────────────────────────────────────────

    def _palette(self, dark_mode: bool) -> dict:
        if dark_mode:
            return dict(
                BG="#0D1117", PANEL_BG="#161B22", TEXT="#E6EDF3",
                SUBTEXT="#8B949E", GRID="#21262D", ACCENT="#58A6FF",
                GOOD="#3FB950", WARN="#D29922", DANGER="#F85149",
                CURVE_CLR="#79C0FF", FIT_CLR="#FF7B72",
                EPISODIC_CLR="#D2A8FF",
            )
        return dict(
            BG="#FAFAFA", PANEL_BG="#FFFFFF", TEXT="#1F2328",
            SUBTEXT="#636E7B", GRID="#D0D7DE", ACCENT="#0969DA",
            GOOD="#1A7F37", WARN="#9A6700", DANGER="#CF222E",
            CURVE_CLR="#0969DA", FIT_CLR="#CF222E",
            EPISODIC_CLR="#8250DF",
        )

    @staticmethod
    def _apply_theme(p: dict) -> None:
        plt.rcParams.update({
            "figure.facecolor": p["BG"],
            "axes.facecolor":   p["PANEL_BG"],
            "axes.edgecolor":   p["GRID"],
            "axes.labelcolor":  p["TEXT"],
            "xtick.color":      p["SUBTEXT"],
            "ytick.color":      p["SUBTEXT"],
            "text.color":       p["TEXT"],
            "grid.color":       p["GRID"],
            "font.family":      "monospace",
        })

    @staticmethod
    def _style_ax(ax, title: str, p: dict) -> None:
        ax.set_title(title, fontsize=11, color=p["TEXT"],
                     pad=8, loc="left", fontweight="bold")
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(p["GRID"])

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(
        self,
        figsize:   Tuple[int, int] = (11, 6),
        save_dir:  Optional[str]   = None,
        dark_mode: bool            = True,
    ) -> List["plt.Figure"]:
        """
        Render four separate publication-ready figures.

        Figures
        -------
        1. Forgetting Curve  - raw + fitted exponential + regime annotation
        2. Retention Ratio   - normalised drift with healthy/danger bands
        3. Rolling Decay     - rolling lambda bar chart (early-warning radar)
        4. Retrain Countdown - semi-circular gauge + key stats block

        Parameters
        ----------
        figsize   : (width, height) per figure in inches.
        save_dir  : Directory to save PNGs. None = skip saving.
        dark_mode : Dark (True) or light (False) theme.

        Returns
        -------
        List of 4 matplotlib Figures.
        """
        if not _MPL:
            raise ImportError("pip install matplotlib")

        p        = self._palette(dark_mode)
        self._apply_theme(p)
        report   = self.report()
        df       = self.dataframe()
        t        = df["elapsed_days"].values
        y        = df["metric_value"].values
        baseline = report.baseline_metric

        # Regime label & colour for annotations
        if report.forgetting_regime == "smooth":
            regime_clr   = p["WARN"]
            regime_label = (f"SMOOTH decay  (R²={report.fit_r_squared:.3f})  "
                            f"— gradual concept drift")
        elif report.forgetting_regime == "episodic":
            regime_clr   = p["EPISODIC_CLR"]
            regime_label = (f"EPISODIC shocks  (R²={report.fit_r_squared:.3f})  "
                            f"— sudden distribution shifts")
        else:
            regime_clr   = p["SUBTEXT"]
            regime_label = "Regime: unknown (insufficient data)"

        stamp = (
            f"metric={self.metric_name}  |  n={report.snapshots_used}  |  "
            f"regime={report.forgetting_regime.upper()}  |  "
            f"speed={report.forgetting_speed.upper()}  |  "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        def _stamp(fig):
            fig.text(0.5, 0.01, stamp, ha="center", va="bottom",
                     fontsize=7.5, color=p["SUBTEXT"])

        figs: List[plt.Figure] = []

        # ── Figure 1: Forgetting Curve ────────────────────────────────────────
        fig1, ax = plt.subplots(figsize=figsize, facecolor=p["BG"])
        fig1.subplots_adjust(bottom=0.14, left=0.10, right=0.97, top=0.88)
        fig1.suptitle("Model Forgetting Curve", fontsize=14,
                      fontweight="bold", color=p["TEXT"], y=0.98)

        # Regime banner
        fig1.text(0.10, 0.91, regime_label, fontsize=8.5,
                  color=regime_clr, fontstyle="italic")

        ax.scatter(t, y, color=p["CURVE_CLR"], s=40, zorder=5,
                   alpha=0.85, linewidths=0, label="Observed weekly Recall")

        if len(y) >= 3:
            roll = pd.Series(y).rolling(min(5, len(y)), center=True).mean().values
            ax.plot(t, roll, color=p["CURVE_CLR"], linewidth=1.8,
                    alpha=0.5, linestyle="--", label="Rolling mean")

        t_fit = np.linspace(0, max(t) * 1.3, 400)
        y_fit = baseline * np.exp(-report.decay_rate * t_fit)

        # ── R² displayed prominently in the fit label ─────────────────────
        r2_txt = (f"   R²={report.fit_r_squared:.4f}"
                  if report.fit_r_squared is not None else "")
        ax.plot(t_fit, y_fit, color=p["FIT_CLR"], linewidth=2.5, zorder=4,
                label=f"Fitted  R(t)=R₀·exp(−{report.decay_rate:.4f}·t){r2_txt}")

        # ── Prominent R² box in the top-right corner ──────────────────────
        if report.fit_r_squared is not None:
            r2_color = (p["GOOD"] if report.fit_r_squared >= self.smooth_r2_cutoff
                        else p["DANGER"])
            r2_box_txt = (
                f"R² = {report.fit_r_squared:.3f}\n"
                f"({'smooth' if report.fit_r_squared >= self.smooth_r2_cutoff else 'episodic'})"
            )
            ax.text(
                0.98, 0.97, r2_box_txt,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=10, fontweight="bold", color=r2_color,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=p["PANEL_BG"],
                    edgecolor=r2_color,
                    linewidth=1.5,
                    alpha=0.9,
                ),
            )

        thresh_y = baseline * (1 - self.retrain_threshold)
        ax.axhline(baseline, color=p["GOOD"], linewidth=1.2,
                   linestyle=":", alpha=0.8, label=f"Baseline  {baseline:.4f}")
        ax.axhline(thresh_y, color=p["DANGER"], linewidth=1.2,
                   linestyle=":", alpha=0.8,
                   label=f"Retrain threshold  (−{self.retrain_threshold*100:.0f}%)")
        ax.fill_between(t_fit, 0, thresh_y, color=p["DANGER"], alpha=0.07)

        # ── Annotate worst-drop week ──────────────────────────────────────
        # worst_drop_week is 1-based and points to the week where the drop
        # LANDS (the low point).  Convert to 0-based array index with -1.
        if report.worst_drop_week is not None:
            wi = report.worst_drop_week - 1          # 0-based array index
            if 0 < wi < len(t):                      # guard: skip first week
                ax.annotate(
                    f"Worst shock\nWeek {report.worst_drop_week}\n"
                    f"−{report.worst_drop_magnitude:.3f}",
                    xy=(t[wi], y[wi]),
                    xytext=(t[wi] + max(t) * 0.05, y[wi] + 0.04),
                    fontsize=7.5, color=p["DANGER"],
                    arrowprops=dict(
                        arrowstyle="->", color=p["DANGER"],
                        lw=1.2, connectionstyle="arc3,rad=0.2",
                    ),
                )

        for rt in self._retrain_events:
            rt_d = (rt - self._snapshots[0].timestamp).total_seconds() / 86_400
            ax.axvline(rt_d, color=p["WARN"], linewidth=1.5,
                       linestyle="-.", alpha=0.85)
            ax.text(rt_d + 0.4, thresh_y * 1.001,
                    "retrain", fontsize=7, color=p["WARN"], va="bottom")

        ax.set_xlabel("Days since deployment / last retrain", fontsize=9)
        ax.set_ylabel(self.metric_name, fontsize=9)
        ax.legend(fontsize=7.5, framealpha=0.15, loc="upper right",
                  facecolor=p["PANEL_BG"], edgecolor=p["GRID"])
        self._style_ax(ax, "[1] Forgetting Curve", p)
        _stamp(fig1)
        figs.append(fig1)

        # ── Figure 2: Retention Ratio ─────────────────────────────────────────
        fig2, ax = plt.subplots(figsize=figsize, facecolor=p["BG"])
        fig2.subplots_adjust(bottom=0.14, left=0.11, right=0.97, top=0.91)
        fig2.suptitle("Retention Ratio Over Time", fontsize=14,
                      fontweight="bold", color=p["TEXT"], y=0.98)

        ret = y / baseline
        ax.plot(t, ret, color=p["ACCENT"], linewidth=2.2, zorder=4)
        ax.scatter(t, ret, color=p["ACCENT"], s=25, zorder=5,
                   alpha=0.75, linewidths=0)
        ax.axhline(1.0, color=p["GOOD"], linewidth=1.2, linestyle=":",
                   alpha=0.8, label="Baseline (100%)")
        ax.axhline(1 - self.retrain_threshold, color=p["DANGER"],
                   linewidth=1.2, linestyle=":", alpha=0.9,
                   label=f"Retrain threshold  {(1-self.retrain_threshold)*100:.0f}%")
        ax.fill_between(t, ret, 1.0,
                        where=(ret >= 1 - self.retrain_threshold),
                        color=p["GOOD"], alpha=0.12, label="Healthy zone")
        ax.fill_between(t, ret, 1.0,
                        where=(ret < 1 - self.retrain_threshold),
                        color=p["DANGER"], alpha=0.18, label="Danger zone")

        breach_mask = ret < (1 - self.retrain_threshold)
        if breach_mask.any():
            ax.scatter(t[breach_mask], ret[breach_mask],
                       color=p["DANGER"], s=55, zorder=6,
                       marker="v", label="Threshold breach")

        ax.set_ylim(max(0.0, ret.min() - 0.08), 1.10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("Days since deployment / last retrain", fontsize=9)
        ax.set_ylabel("Retention  (current / baseline)", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.15, loc="lower left",
                  facecolor=p["PANEL_BG"], edgecolor=p["GRID"])
        self._style_ax(ax, "[2] Retention Ratio Over Time", p)
        _stamp(fig2)
        figs.append(fig2)

        # ── Figure 3: Rolling Decay Rate ──────────────────────────────────────
        fig3, ax = plt.subplots(figsize=figsize, facecolor=p["BG"])
        fig3.subplots_adjust(bottom=0.14, left=0.10, right=0.88, top=0.91)
        fig3.suptitle("Rolling Decay Rate  (Early Warning Radar)", fontsize=14,
                      fontweight="bold", color=p["TEXT"], y=0.98)

        win = max(3, len(t) // 5)
        rl, ce = [], []
        for i in range(win, len(t)):
            sub_t = t[i - win:i] - t[i - win]
            sub_y = np.clip(y[i - win:i] / baseline, 1e-6, None)
            try:
                lam_w, _ = self._fit_decay(sub_t, sub_y)
            except Exception:
                lam_w = 0.0
            rl.append(lam_w)
            ce.append(t[i])

        if rl:
            rl_arr = np.array(rl)
            ce_arr = np.array(ce)
            cmap   = LinearSegmentedColormap.from_list(
                "fg", [p["GOOD"], p["WARN"], p["DANGER"]], N=256)
            norm   = plt.Normalize(0, max(rl_arr.max(), 1e-4))
            colors = cmap(norm(rl_arr))
            bw     = np.diff(ce_arr).mean() * 0.85 if len(ce_arr) > 1 else 1.0
            ax.bar(ce_arr, rl_arr, width=bw, color=colors, alpha=0.85, zorder=3)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig3.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
            cb.set_label("Decay rate λ", fontsize=8, color=p["TEXT"])
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=p["SUBTEXT"], fontsize=7)

            rm = pd.Series(rl_arr).rolling(
                min(5, len(rl_arr)), center=True).mean().values
            ax.plot(ce_arr, rm, color=p["TEXT"], linewidth=1.8,
                    linestyle="--", alpha=0.6, label="Trend")

            lam_mean = rl_arr.mean()
            ax.axhline(lam_mean, color=p["WARN"], linewidth=1.0,
                       linestyle=":", alpha=0.7,
                       label=f"Mean λ = {lam_mean:.5f}")
            accel_mask = rl_arr > lam_mean
            if accel_mask.any():
                ax.fill_between(ce_arr, 0, rl_arr,
                                where=accel_mask,
                                color=p["DANGER"], alpha=0.12,
                                label="Accelerated decay")

            ax.legend(fontsize=8, framealpha=0.15,
                      facecolor=p["PANEL_BG"], edgecolor=p["GRID"])

        ax.set_xlabel("Days since deployment", fontsize=9)
        ax.set_ylabel("Rolling decay rate  λ", fontsize=9)
        self._style_ax(ax, "[3] Rolling Decay Rate  (Early Warning Radar)", p)
        _stamp(fig3)
        figs.append(fig3)

        # ── Figure 4: Retrain Countdown Gauge ─────────────────────────────────
        fig4, ax = plt.subplots(figsize=figsize, facecolor=p["BG"])
        fig4.subplots_adjust(bottom=0.10, left=0.05, right=0.95, top=0.93)
        fig4.suptitle("Retrain Countdown", fontsize=14,
                      fontweight="bold", color=p["TEXT"], y=0.98)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        days_left   = report.predicted_days_to_threshold
        retrain_now = report.retrain_recommended

        if retrain_now:
            gc = p["DANGER"]; lt = "RETRAIN NOW"
            ds = "Threshold reached"; pct = 0.0
        elif days_left is None:
            gc = p["GOOD"]; lt = "STABLE"
            ds = "No retrain foreseen"; pct = 1.0
        else:
            horizon = 60.0
            pct = min(days_left / horizon, 1.0)
            gc  = (p["GOOD"] if pct > 0.5
                   else (p["WARN"] if pct > 0.2 else p["DANGER"]))
            lt  = f"{days_left:.0f} days"
            ds  = "until retrain recommended"

        cx, cy, r = 0.5, 0.60, 0.28
        ax.add_patch(Wedge((cx, cy), r, 180, 360,
                           width=0.07, facecolor=p["GRID"], zorder=2))
        ax.add_patch(Wedge((cx, cy), r, 180, 180 + pct * 180,
                           width=0.07, facecolor=gc, zorder=3, alpha=0.9))
        ax.text(cx, cy + 0.06, lt, ha="center", va="center",
                fontsize=30, fontweight="bold", color=gc, zorder=5)
        ax.text(cx, cy - 0.07, ds, ha="center", va="center",
                fontsize=10, color=p["SUBTEXT"], zorder=5)

        badge_clr = p["EPISODIC_CLR"] if report.forgetting_regime == "episodic" \
                    else (p["WARN"] if report.forgetting_regime == "smooth"
                          else p["SUBTEXT"])
        ax.text(cx, cy - 0.17,
                f"Regime: {report.forgetting_regime.upper()}",
                ha="center", va="center", fontsize=8.5,
                color=badge_clr, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=p["PANEL_BG"],
                          edgecolor=badge_clr, linewidth=1.2))

        stats = [
            ("Baseline",     f"{report.baseline_metric:.4f}"),
            ("Current",      f"{report.current_metric:.4f}"),
            ("Retention",    f"{report.retention_ratio:.1%}"),
            ("Decay λ",      f"{report.decay_rate:.5f}"),
            ("Half-life",    (f"{report.half_life_days:.1f} days"
                              if report.half_life_days else "—")),
            ("R²",           (f"{report.fit_r_squared:.4f}"
                              if report.fit_r_squared is not None else "—")),
            ("Snapshots",    str(report.snapshots_used)),
            # worst_drop_week is now the correct destination week
            ("Worst shock",  (f"Week {report.worst_drop_week}  "
                              f"(−{report.worst_drop_magnitude:.3f})"
                              if report.worst_drop_week else "—")),
        ]
        row_y = 0.35
        for lbl, val in stats:
            ax.text(0.30, row_y, lbl + " :", ha="right",
                    fontsize=9, color=p["SUBTEXT"])
            ax.text(0.32, row_y, val, ha="left",
                    fontsize=9, color=p["TEXT"], fontweight="bold")
            row_y -= 0.042

        rec_color = (p["DANGER"] if retrain_now
                     else (p["WARN"] if (days_left or 999) < 14 else p["GOOD"]))
        rec_msg = (
            "⚠  Retrain immediately."
            if retrain_now
            else (
                f"Retrain by {report.recommended_retrain_date.strftime('%b %d, %Y')}"
                if report.recommended_retrain_date
                else "✓  Model performing well."
            )
        )
        ax.text(cx, 0.04, rec_msg, ha="center", va="bottom",
                fontsize=9, color=rec_color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=p["PANEL_BG"],
                          edgecolor=rec_color, linewidth=1.5))
        self._style_ax(ax, "[4] Retrain Countdown", p)
        ax.set_title("[4] Retrain Countdown", fontsize=11, color=p["TEXT"],
                     pad=8, loc="left", fontweight="bold")
        _stamp(fig4)
        figs.append(fig4)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            names = [
                "fig1_forgetting_curve.png",
                "fig2_retention_ratio.png",
                "fig3_rolling_decay.png",
                "fig4_retrain_countdown.png",
            ]
            for fig, name in zip(figs, names):
                out = os.path.join(save_dir, name)
                fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=p["BG"])
                print(f"Saved -> {out}")

        return figs

    # ── live tracking ─────────────────────────────────────────────────────────

    def live_track(
        self,
        metric_fn:        Callable[[], float],
        interval_seconds: float = 5.0,
        n_samples_fn:     Optional[Callable[[], int]] = None,
        dark_mode:        bool = True,
        figsize:          Tuple[int, int] = (14, 8),
    ) -> None:
        """
        Live-update a 4-panel dashboard by polling metric_fn every
        interval_seconds. Each observation is auto-logged.

        Parameters
        ----------
        metric_fn        : Zero-arg callable returning the latest metric.
        interval_seconds : Polling cadence in seconds. Default 5.
        n_samples_fn     : Optional callable returning sample count.
        dark_mode        : Colour theme.
        figsize          : Overall figure size.
        """
        if not _MPL:
            raise ImportError("pip install matplotlib")

        p = self._palette(dark_mode)
        self._apply_theme(p)

        fig = plt.figure(figsize=figsize, facecolor=p["BG"])
        fig.suptitle(
            f"Live Model Forgetting Tracker  |  {self.metric_name}",
            fontsize=14, fontweight="bold", color=p["TEXT"], y=0.99,
        )
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32,
                                left=0.08, right=0.97, top=0.94, bottom=0.08)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        status = fig.text(
            0.5, 0.005,
            f"Waiting for baseline ({self.baseline_window} snapshots)...",
            ha="center", va="bottom", fontsize=8, color=p["SUBTEXT"],
        )

        def _redraw(_frame):
            try:
                val = float(metric_fn())
            except Exception as exc:
                status.set_text(f"metric_fn error: {exc}")
                return

            n = int(n_samples_fn()) if n_samples_fn else 0
            self.log(val, n_samples=n)

            if len(self._snapshots) < self.baseline_window:
                status.set_text(
                    f"Collecting baseline... "
                    f"({len(self._snapshots)}/{self.baseline_window})"
                )
                return

            try:
                report = self.report()
            except Exception as exc:
                status.set_text(f"report() error: {exc}")
                return

            df       = self.dataframe()
            t        = df["elapsed_days"].values
            y        = df["metric_value"].values
            baseline = report.baseline_metric

            ax1.cla()
            ax1.scatter(t, y, color=p["CURVE_CLR"], s=22, zorder=5,
                        alpha=0.8, linewidths=0)
            t_fit = np.linspace(0, max(t) * 1.3, 300)
            y_fit = baseline * np.exp(-report.decay_rate * t_fit)
            ax1.plot(t_fit, y_fit, color=p["FIT_CLR"], linewidth=2, zorder=4,
                     label=f"λ={report.decay_rate:.4f}")
            ax1.axhline(baseline, color=p["GOOD"], linewidth=1,
                        linestyle=":", alpha=0.7)
            ax1.axhline(baseline * (1 - self.retrain_threshold),
                        color=p["DANGER"], linewidth=1, linestyle=":", alpha=0.8)
            ax1.fill_between(t_fit, 0,
                             baseline * (1 - self.retrain_threshold),
                             color=p["DANGER"], alpha=0.06)
            ax1.legend(fontsize=7, framealpha=0.15,
                       facecolor=p["PANEL_BG"], edgecolor=p["GRID"])
            ax1.set_xlabel("Days", fontsize=8)
            ax1.set_ylabel(self.metric_name, fontsize=8)
            self._style_ax(ax1, "[1] Forgetting Curve", p)

            ax2.cla()
            ret = y / baseline
            ax2.plot(t, ret, color=p["ACCENT"], linewidth=1.8, zorder=4)
            ax2.scatter(t, ret, color=p["ACCENT"], s=18, zorder=5,
                        alpha=0.7, linewidths=0)
            ax2.axhline(1.0, color=p["GOOD"], linewidth=1,
                        linestyle=":", alpha=0.7)
            ax2.axhline(1 - self.retrain_threshold, color=p["DANGER"],
                        linewidth=1, linestyle=":", alpha=0.8)
            ax2.fill_between(t, ret, 1.0,
                             where=(ret >= 1 - self.retrain_threshold),
                             color=p["GOOD"], alpha=0.10)
            ax2.fill_between(t, ret, 1.0,
                             where=(ret < 1 - self.retrain_threshold),
                             color=p["DANGER"], alpha=0.16)
            ax2.set_ylim(max(0, ret.min() - 0.05), 1.06)
            ax2.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax2.set_xlabel("Days", fontsize=8)
            ax2.set_ylabel("Retention", fontsize=8)
            self._style_ax(ax2, "[2] Retention Ratio", p)

            ax3.cla()
            win = max(3, len(t) // 5)
            rl, ce = [], []
            for i in range(win, len(t)):
                sub_t = t[i - win:i] - t[i - win]
                sub_y = np.clip(y[i - win:i] / baseline, 1e-6, None)
                try:
                    lam_w, _ = self._fit_decay(sub_t, sub_y)
                except Exception:
                    lam_w = 0.0
                rl.append(lam_w); ce.append(t[i])
            if rl:
                rl_a   = np.array(rl); ce_a = np.array(ce)
                cmap   = LinearSegmentedColormap.from_list(
                    "fg", [p["GOOD"], p["WARN"], p["DANGER"]], N=256)
                norm   = plt.Normalize(0, max(rl_a.max(), 1e-4))
                colors = cmap(norm(rl_a))
                bw     = np.diff(ce_a).mean() * 0.85 if len(ce_a) > 1 else 1.0
                ax3.bar(ce_a, rl_a, width=bw, color=colors, alpha=0.85, zorder=3)
                rm = pd.Series(rl_a).rolling(
                    min(5, len(rl_a)), center=True).mean().values
                ax3.plot(ce_a, rm, color=p["TEXT"], linewidth=1.3,
                         linestyle="--", alpha=0.55)
            ax3.set_xlabel("Days", fontsize=8)
            ax3.set_ylabel("λ", fontsize=8)
            self._style_ax(ax3, "[3] Rolling Decay Rate", p)

            ax4.cla()
            ax4.set_xlim(0, 1); ax4.set_ylim(0, 1); ax4.axis("off")

            days_left   = report.predicted_days_to_threshold
            retrain_now = report.retrain_recommended

            if retrain_now:
                gc = p["DANGER"]; lt = "RETRAIN NOW"
                ds = "Threshold reached"; pct = 0.0
            elif days_left is None:
                gc = p["GOOD"]; lt = "STABLE"
                ds = "No retrain foreseen"; pct = 1.0
            else:
                horizon = 60.0
                pct = min(days_left / horizon, 1.0)
                gc  = (p["GOOD"] if pct > 0.5
                       else (p["WARN"] if pct > 0.2 else p["DANGER"]))
                lt  = f"{days_left:.0f} d"
                ds  = "until retrain"

            cx, cy, r = 0.5, 0.58, 0.25
            ax4.add_patch(Wedge((cx, cy), r, 180, 360,
                                width=0.07, facecolor=p["GRID"], zorder=2))
            ax4.add_patch(Wedge((cx, cy), r, 180, 180 + pct * 180,
                                width=0.07, facecolor=gc, zorder=3, alpha=0.9))
            ax4.text(cx, cy + 0.05, lt, ha="center", va="center",
                     fontsize=22, fontweight="bold", color=gc, zorder=5)
            ax4.text(cx, cy - 0.07, ds, ha="center", va="center",
                     fontsize=8, color=p["SUBTEXT"], zorder=5)

            live_stats = [
                ("Observed",  str(report.snapshots_used)),
                ("Retention", f"{report.retention_ratio:.1%}"),
                ("λ",         f"{report.decay_rate:.5f}"),
                ("Half-life", (f"{report.half_life_days:.1f}d"
                               if report.half_life_days else "—")),
                ("Regime",    report.forgetting_regime.upper()),
            ]
            ry = 0.28
            for lbl, v in live_stats:
                ax4.text(0.26, ry, lbl + " :", ha="right",
                         fontsize=8, color=p["SUBTEXT"])
                ax4.text(0.28, ry, v, ha="left", fontsize=8,
                         color=p["TEXT"], fontweight="bold")
                ry -= 0.048

            self._style_ax(ax4, "[4] Retrain Countdown", p)
            ax4.set_title("[4] Retrain Countdown", fontsize=10,
                          color=p["TEXT"], pad=6, loc="left", fontweight="bold")
            status.set_text(
                f"Last poll: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}  |  "
                f"Latest {self.metric_name}: {val:.4f}  |  "
                f"Retention: {report.retention_ratio:.1%}  |  "
                f"Regime: {report.forgetting_regime.upper()}"
            )
            fig.canvas.draw_idle()

        ani = animation.FuncAnimation(
            fig, _redraw,
            interval=int(interval_seconds * 1000),
            cache_frame_data=False,
        )
        plt.show()
        return ani

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_series(
        self,
        baseline: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        snaps = self._snapshots
        t0    = snaps[0].timestamp
        if baseline is None:
            baseline_vals = [s.metric_value for s in snaps[:self.baseline_window]]
            if self.baseline_method == "max":
                baseline = float(np.max(baseline_vals))
            elif self.baseline_method == "top3":
                k = min(3, len(baseline_vals))
                baseline = float(np.mean(sorted(baseline_vals, reverse=True)[:k]))
            else:
                baseline = float(np.mean(baseline_vals))
        t_days = np.array(
            [(s.timestamp - t0).total_seconds() / 86_400 for s in snaps])
        y_norm = np.clip(
            np.array([s.metric_value for s in snaps]) / (baseline or 1.0),
            1e-9, None)
        return t_days, y_norm

    @staticmethod
    def _exp_decay(t: np.ndarray, lam: float) -> np.ndarray:
        return np.exp(-lam * t)

    def _fit_decay(
        self,
        t: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, Optional[float]]:
        if _SCIPY and len(t) >= 3:
            try:
                popt, _ = curve_fit(
                    self._exp_decay, t, y,
                    p0=[0.01], bounds=(0, 10), maxfev=5000)
                lam    = float(popt[0])
                y_pred = self._exp_decay(t, lam)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                return lam, r2
            except RuntimeError:
                pass
        safe_y = np.clip(y, 1e-9, None)
        log_y  = np.log(safe_y)
        if len(t) < 2 or t.std() < 1e-9:
            return 0.0, None
        lam = max(0.0, float(-np.polyfit(t, log_y, 1)[0]))
        return lam, None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def load_from_dataframe(
    df:            pd.DataFrame,
    metric_col:    str,
    timestamp_col: str = "timestamp",
    **tracker_kwargs,
) -> ModelForgettingTracker:
    """
    Build a tracker from an existing DataFrame of historical metrics.

    Parameters
    ----------
    df            : DataFrame with timestamp and metric columns.
    metric_col    : Column containing the performance metric.
    timestamp_col : Column containing timestamps.
    **tracker_kwargs : Forwarded to ModelForgettingTracker.__init__.

    Example
    -------
    >>> tracker = load_from_dataframe(df, metric_col="f1_score")
    >>> figs = tracker.plot()
    >>> plt.show()
    """
    df = df.copy().sort_values(timestamp_col)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    tracker = ModelForgettingTracker(**tracker_kwargs)
    for _, row in df.iterrows():
        tracker.log(
            metric_value = float(row[metric_col]),
            timestamp    = row[timestamp_col].to_pydatetime(),
            n_samples    = int(row.get("n_samples", 0)),
        )
    return tracker
