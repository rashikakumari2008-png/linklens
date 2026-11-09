"""
evaluate.py  -  Steps 2-4 of LinkLens: baselines, tuning, evaluation, ablation, threshold sweep, speed test

How the test is kept honest
  * DEV data  (seed 42, the same data as data/reports.csv): used ONLY to choose LinkLens weights and
    every method's threshold.
  * TEST data (five fresh seeds, never seen during tuning): all reported results come from these.

Run:  python evaluate.py          (takes under a minute)
Output goes to the results/ folder: tables (.csv), charts (.png), results_summary.md, best_params.json
"""
import itertools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

import engine as e
from generate_data import generate

DEV_SEED = 42
TEST_SEEDS = [101, 102, 103, 104, 105]
OUT = "results"

LINKLENS = "LinkLens (weighted)"
BASE_A = "Baseline A (text only)"
BASE_B = "Baseline B (identifiers only)"

# chart styling (blue / orange / aqua = the first three validated categorical slots)
COLORS = {LINKLENS: "#2a78d6", BASE_A: "#eb6834", BASE_B: "#1baf7a"}
STYLES = {LINKLENS: "-", BASE_A: "--", BASE_B: ":"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


# ----------------------------------------------------------------------------
# Data and metrics
# ----------------------------------------------------------------------------
def make_dataset(seed, gangs=40, singles=160):
    reports, gang = e.frames_from_rows(generate(seed, gangs, singles))
    feat = e.build_features(reports)
    return dict(seed=seed, n=len(reports), feat=feat, S=e.text_similarity(reports["narrative"]),
                y=e.true_pair_labels(gang), shared=e.baseline_identifier_scores(feat) > 0)


def prf(pred, y):
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_threshold(scores, y):
    """The threshold with the best F1 on the data given (we only ever call this on DEV data)."""
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))])


def full_metrics(scores, d, thr):
    y, shared = d["y"], d["shared"]
    pred = scores >= thr
    p, r, f = prf(pred, y)
    hard, trap = y & ~shared, ~y & shared
    return dict(precision=p, recall=r, f1=f,
                ap=float(average_precision_score(y, scores)),
                recall_hard=float(pred[hard].sum() / max(hard.sum(), 1)),      # true links with NO shared identifier
                trap_false_alarm=float(pred[trap].sum() / max(trap.sum(), 1)))  # unrelated pairs that share an identifier


def method_scores(name, d, params):
    if name == BASE_A:
        return e.baseline_text_scores(d["S"])
    if name == BASE_B:
        return e.baseline_identifier_scores(d["feat"])
    return e.score_c(d["feat"], d["S"], params)["score"]


# ----------------------------------------------------------------------------
# Tuning (DEV data only)
# ----------------------------------------------------------------------------
def tune_weights(dev):
    """Tries every weight mix (steps of 0.1, adding up to 1) and 3 time scales; keeps the best average precision."""
    best, best_ap, tried = None, -1.0, 0
    for wt, wi in itertools.product(range(11), range(11)):
        if wt + wi > 10:
            continue
        for tau in (20.0, 45.0, 90.0):
            p = dict(e.DEFAULT_PARAMS, w_text=wt / 10, w_id=wi / 10, w_time=(10 - wt - wi) / 10, tau=tau)
            ap = average_precision_score(dev["y"], e.score_c(dev["feat"], dev["S"], p)["score"])
            tried += 1
            if ap > best_ap:
                best, best_ap = p, ap
    print(f"  tried {tried} weight settings, best dev AP = {best_ap:.3f}")
    return best


def evaluate_method(name, params, dev, tests):
    thr = best_threshold(method_scores(name, dev, params), dev["y"])
    rows = [full_metrics(method_scores(name, d, params), d, thr) for d in tests]
    return thr, pd.DataFrame(rows)


def mean_std(df):
    out = {}
    for c in df.columns:
        out[c] = df[c].mean()
        out[c + "_std"] = df[c].std(ddof=0)
    return out


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)


def chart_threshold(sweep, chosen):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in (LINKLENS, BASE_A, BASE_B):
        s = sweep[sweep.method == name]
        a1.plot(s.threshold, s.f1, STYLES[name], color=COLORS[name], linewidth=2, label=name)
        t = chosen[name]
        a1.plot([t], [np.interp(t, s.threshold, s.f1)], "o", color=COLORS[name], markersize=7,
                markeredgecolor="white", markeredgewidth=1.5)
    a1.set_title("F1 score as the threshold changes", loc="left", color=INK, fontsize=11, fontweight="bold")
    a1.set_xlabel("matching threshold", color=INK2)
    a1.set_ylabel("F1 (higher is better)", color=INK2)
    a1.set_ylim(0, 1.02)
    a1.legend(frameon=False, fontsize=9, loc="upper right")
    style_axes(a1)
    s = sweep[sweep.method == LINKLENS]
    a2.plot(s.threshold, s.precision, "-", color=COLORS[LINKLENS], linewidth=2, label="precision")
    a2.plot(s.threshold, s.recall, "--", color=COLORS[LINKLENS], linewidth=2, label="recall")
    a2.axvline(chosen[LINKLENS], color=INK2, linewidth=1, linestyle=":")
    a2.text(chosen[LINKLENS] + 0.01, 0.05, "chosen threshold", color=INK2, fontsize=9)
    a2.set_title("LinkLens: precision vs recall trade-off", loc="left", color=INK, fontsize=11, fontweight="bold")
    a2.set_xlabel("matching threshold", color=INK2)
    a2.set_ylim(0, 1.02)
    a2.legend(frameon=False, fontsize=9, loc="lower left")
    style_axes(a2)
    fig.text(0.01, 0.005, "Dots mark each method's threshold, chosen on dev data. Curves are averages over 5 unseen test datasets. "
             "Baseline B only outputs 0 or 1, so its line is flat.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "threshold_sweep.png"), dpi=160)
    plt.close(fig)


def chart_ablation(abl):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = list(abl["variant"])[::-1]
    vals = list(abl["f1"])[::-1]
    errs = list(abl["f1_std"])[::-1]
    ax.barh(labels, vals, xerr=errs, color=COLORS[LINKLENS], height=0.55,
            error_kw=dict(ecolor=INK2, capsize=3, linewidth=1))
    for i, v in enumerate(vals):
        ax.text(v + 0.012 + errs[i], i, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 on unseen test data (higher is better)", color=INK2)
    ax.set_title("Ablation: what happens when one signal is removed", loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation.png"), dpi=160)
    plt.close(fig)


def chart_speed(speed):
    fig, ax = plt.subplots(figsize=(8, 4))
    sizes = sorted(speed.reports.unique())
    x = np.arange(len(sizes))
    width = 0.26
    for k, (col, label, color) in enumerate([("python_s", "Pure Python loops", COLORS[BASE_A]),
                                             ("numpy_s", "NumPy (vectorised)", COLORS[BASE_B]),
                                             ("c_s", "C engine", COLORS[LINKLENS])]):
        vals = [speed[speed.reports == s][col].iloc[0] for s in sizes]
        ax.bar(x + (k - 1) * width, vals, width * 0.92, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} reports\n({int(speed[speed.reports == s].pairs.iloc[0]):,} pairs)" for s in sizes])
    ax.set_ylabel("seconds (log scale, lower is better)", color=INK2)
    ax.set_title("Scoring speed: C vs Python vs NumPy", loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speed.png"), dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Speed test
# ----------------------------------------------------------------------------
def best_of(fn, repeats):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        r = fn()
        times.append(time.perf_counter() - t)
    return min(times), r


def speed_test(params):
    e.load_c_library()                                    # compile first so compile time is not counted
    rows = []
    for gangs, singles in [(40, 160), (80, 320), (120, 480), (200, 800)]:
        d = make_dataset(7, gangs, singles)
        tc, rc = best_of(lambda: e.score_c(d["feat"], d["S"], params), 3)
        tn, rn = best_of(lambda: e.score_numpy(d["feat"], d["S"], params), 3)
        tp, rp = best_of(lambda: e.score_python(d["feat"], d["S"], params), 1)
        diff = max(np.abs(rc["score"] - rp["score"]).max(), np.abs(rc["score"] - rn["score"]).max())
        rows.append(dict(reports=d["n"], pairs=len(rc["score"]), c_s=tc, numpy_s=tn, python_s=tp,
                         c_vs_python=tp / tc, c_vs_numpy=tn / tc, max_score_difference=diff))
        print(f"  {d['n']:>5} reports: C {tc:.3f}s | NumPy {tn:.3f}s | Python {tp:.2f}s | max diff {diff:.1e}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    print("1/6 Building datasets (1 dev + 5 test)...")
    dev = make_dataset(DEV_SEED)
    tests = [make_dataset(s) for s in TEST_SEEDS]
    print(f"  dev: {dev['n']} reports, {int(dev['y'].sum())} true pairs "
          f"({int((dev['y'] & ~dev['shared']).sum())} of them have no shared identifier)")

    print("2/6 Checking that C, Python and NumPy give the same scores...")
    sc = [e.score_c(dev["feat"], dev["S"], e.DEFAULT_PARAMS)["score"],
          e.score_python(dev["feat"], dev["S"], e.DEFAULT_PARAMS)["score"],
          e.score_numpy(dev["feat"], dev["S"], e.DEFAULT_PARAMS)["score"]]
    assert np.allclose(sc[0], sc[1]) and np.allclose(sc[0], sc[2]), "engines disagree!"
    print("  all three agree")

    print("3/6 Tuning LinkLens weights on dev data...")
    best = tune_weights(dev)
    with open(os.path.join(OUT, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("  chosen:", {k: best[k] for k in ("w_text", "w_id", "w_time", "tau")})

    print("4/6 Evaluating on unseen test data...")
    chosen, rows, per_seed = {}, [], []
    for name in (LINKLENS, BASE_A, BASE_B):
        thr, df = evaluate_method(name, best, dev, tests)
        chosen[name] = thr
        rows.append(dict(method=name, threshold=thr, **mean_std(df)))
        per_seed.append(df.assign(method=name, seed=TEST_SEEDS))
    main_tbl = pd.DataFrame(rows)
    main_tbl.to_csv(os.path.join(OUT, "metrics_test.csv"), index=False)
    pd.concat(per_seed).to_csv(os.path.join(OUT, "metrics_per_seed.csv"), index=False)
    print(main_tbl[["method", "threshold", "precision", "recall", "f1", "ap", "recall_hard", "trap_false_alarm"]]
          .round(3).to_string(index=False))

    print("5/6 Ablation and threshold sweep...")
    variants = {"Full LinkLens": best,
                "without text similarity": dict(best, w_text=0.0),
                "without identifiers": dict(best, w_id=0.0),
                "without time": dict(best, w_time=0.0),
                "without conflict penalties": dict(best, pen_cat=0.0, pen_gap=0.0, pen_common=0.0)}
    abl = []
    for vname, vp in variants.items():
        _, df = evaluate_method(LINKLENS, vp, dev, tests)
        abl.append(dict(variant=vname, **mean_std(df)))
    abl = pd.DataFrame(abl)
    abl.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    print(abl[["variant", "precision", "recall", "f1", "ap"]].round(3).to_string(index=False))

    grid = np.round(np.linspace(0, 1, 51), 2)
    sweep_rows = []
    for name in (LINKLENS, BASE_A, BASE_B):
        scores = [method_scores(name, d, best) for d in tests]
        for t in grid:
            prs = np.array([prf(s >= t, d["y"]) for s, d in zip(scores, tests)])
            sweep_rows.append(dict(method=name, threshold=t, precision=prs[:, 0].mean(),
                                   recall=prs[:, 1].mean(), f1=prs[:, 2].mean()))
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(OUT, "threshold_sweep.csv"), index=False)

    print("6/6 Speed test and charts...")
    speed = speed_test(best)
    speed.to_csv(os.path.join(OUT, "speed.csv"), index=False)
    chart_threshold(sweep, chosen)
    chart_ablation(abl)
    chart_speed(speed)
    write_summary(main_tbl, abl, speed, best)
    print(f"Done. Open the {OUT}/ folder (results_summary.md, charts and tables).")


def write_summary(main_tbl, abl, speed, best):
    def pm(r, c):
        return f"{r[c]:.2f} ± {r[c + '_std']:.2f}"
    L = ["# LinkLens results", "",
         "All numbers are averages over 5 unseen synthetic test datasets (± = spread across the 5).",
         "Weights and thresholds were chosen on a separate dev dataset only.", "",
         f"Chosen weights: text {best['w_text']}, identifiers {best['w_id']}, time {best['w_time']} (time scale {best['tau']:.0f} days).", "",
         "## Method comparison", "",
         "| Method | Precision | Recall | F1 | Avg. precision (AP) | Recall on hard links* | False alarms on trap pairs** |",
         "|---|---|---|---|---|---|---|"]
    for _, r in main_tbl.iterrows():
        L.append(f"| {r['method']} | {pm(r, 'precision')} | {pm(r, 'recall')} | {pm(r, 'f1')} | {pm(r, 'ap')} | "
                 f"{pm(r, 'recall_hard')} | {pm(r, 'trap_false_alarm')} |")
    L += ["", "*Hard links = truly linked report pairs that share NO identifier.",
          "**Trap pairs = unrelated report pairs that DO share an identifier (recycled or popular numbers).", "",
          "## Ablation (remove one signal at a time, threshold re-tuned on dev)", "",
          "| Variant | Precision | Recall | F1 |", "|---|---|---|---|"]
    for _, r in abl.iterrows():
        L.append(f"| {r['variant']} | {pm(r, 'precision')} | {pm(r, 'recall')} | {pm(r, 'f1')} |")
    L += ["", "## Speed (scoring every pair)", "",
          "| Reports | Pairs | C (s) | NumPy (s) | Pure Python (s) | C vs Python | C vs NumPy |", "|---|---|---|---|---|---|---|"]
    for _, r in speed.iterrows():
        L.append(f"| {int(r['reports'])} | {int(r['pairs']):,} | {r['c_s']:.3f} | {r['numpy_s']:.3f} | {r['python_s']:.2f} | "
                 f"{r['c_vs_python']:.0f}x | {r['c_vs_numpy']:.1f}x |")
    L += ["", "Scores from C, NumPy and pure Python were checked to be identical (max difference in the speed table CSV).", ""]
    with open(os.path.join(OUT, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
