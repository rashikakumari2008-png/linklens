"""
linkgraph.py  -  the "explainable" layer of LinkLens (used by the Streamlit app)

  analyze()          scores every pair of reports with the C engine (NumPy fallback if C cannot be built)
  suggested_links()  turns scores into a table of suggested links with a confidence tier
  evidence()         the evidence trail for one pair: what supports the link, what argues against it
  build_clusters()   chains links into probable "campaigns" with NetworkX (connected components)

Run  python linkgraph.py  to check the tiers and the chaining on 5 unseen test datasets.
It writes results/tier_check.csv, clusters.csv, clusters_per_seed.csv, chaining_sweep.csv, chaining_summary.md.

Tier rules
  strong     score >= 0.40 and NO conflicting signal (different category, big time gap, very common identifier)
  uncertain  score >= 0.33 otherwise - including high scores that carry a conflicting signal -> a human should look
  weak       below 0.33 (hidden in the app)
Campaigns = reports chained together by links with score >= CLUSTER_EDGE.
"""
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import engine as e

HERE = os.path.dirname(os.path.abspath(__file__))

TIER_STRONG = 0.40
TIER_UNCERTAIN = 0.33
# Chaining threshold. Rule (applied on DEV data only, see main()): the lowest threshold at which campaigns are
# at least 99% precise. Below ~0.38 the groups start merging different gangs together (a sharp cliff).
CLUSTER_EDGE = 0.40
CONFLICT_MASK = e.FLAG_CATEGORY | e.FLAG_GAP | e.FLAG_COMMON
ID_LABELS = {"phone": "phone number", "upi_id": "UPI ID", "url": "website"}


def load_params():
    path = os.path.join(HERE, "results", "best_params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(e.DEFAULT_PARAMS)


def text_model(narratives):
    """Same TF-IDF as engine.text_similarity, but keeps the vectors so we can show shared wording."""
    vec = TfidfVectorizer(preprocessor=e._clean, ngram_range=(1, 2), sublinear_tf=True)
    x = vec.fit_transform(narratives).tocsr()
    s = np.ascontiguousarray((x @ x.T).toarray(), dtype=np.float64)
    return s, x, np.array(vec.get_feature_names_out())


def analyze(reports, params=None):
    """Scores all pairs. Uses the C engine; if gcc is not available (some hosting), falls back to NumPy."""
    params = params or load_params()
    feat = e.build_features(reports)
    s, x, terms = text_model(reports["narrative"])
    try:
        out, engine_name = e.score_c(feat, s, params), "C engine"
    except Exception:
        out, engine_name = e.score_numpy(feat, s, params), "NumPy fallback (C engine could not be built here)"
    I, J = e.pair_index(feat["n"])
    return dict(reports=reports, params=params, feat=feat, S=s, X=x, terms=terms, out=out, I=I, J=J,
                engine=engine_name)


# ----------------------------------------------------------------------------
# Tiers and suggested links
# ----------------------------------------------------------------------------
def tiers_for(score, flags):
    score, flags = np.asarray(score), np.asarray(flags)
    conflict = (flags & CONFLICT_MASK) != 0
    return np.select([(score >= TIER_STRONG) & ~conflict, score >= TIER_UNCERTAIN], ["strong", "uncertain"],
                     default="weak")


def tier_of(score, flags):
    return str(tiers_for(score, flags))


def conflict_text(flags):
    names = [n for b, n in e.FLAG_NAMES.items() if flags & b and b != e.FLAG_CITY]
    return "; ".join(names) if names else "none"


def suggested_links(a, min_score=TIER_UNCERTAIN):
    """Table of suggested links (best first). One row per pair of reports."""
    out, I, J, rep = a["out"], a["I"], a["J"], a["reports"]
    k = np.where(out["score"] >= min_score)[0]
    if len(k) == 0:
        return pd.DataFrame()
    i, j = I[k], J[k]
    ids = rep["report_id"].to_numpy()
    df = pd.DataFrame({
        "pair_index": k, "report_a": ids[i], "report_b": ids[j], "i": i, "j": j,
        "score": out["score"][k], "tier": tiers_for(out["score"][k], out["flags"][k]),
        "text": a["S"][i, j], "identifiers": out["id"][k], "time": out["time"][k], "flags": out["flags"][k],
    })
    df["conflict"] = df["flags"].map(conflict_text)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def pair_k(n, i, j):
    """Position of pair (i, j), i < j, in the engine's output arrays."""
    return i * n - i * (i + 1) // 2 + (j - i - 1)


# ----------------------------------------------------------------------------
# Evidence trail
# ----------------------------------------------------------------------------
def shared_terms(x, terms, i, j, k=4):
    """Words/phrases both narratives use that are distinctive (high TF-IDF in both)."""
    prod = x[i].multiply(x[j]).tocoo()
    if prod.nnz == 0:
        return []
    return [str(terms[prod.col[o]]) for o in np.argsort(-prod.data)[:k]]


def tier_reason(tier, score, flags):
    if tier == "strong":
        return f"Strong: score is at least {TIER_STRONG:.2f} and nothing conflicts."
    if tier == "uncertain":
        if score >= TIER_STRONG:
            return "Uncertain: the score is high, but a conflicting signal means a human should check it."
        return (f"Uncertain: the score is between {TIER_UNCERTAIN:.2f} and {TIER_STRONG:.2f}, "
                "the zone where many suggested links are wrong.")
    return f"Weak: the score is below {TIER_UNCERTAIN:.2f}. Almost none of these are real links."


def evidence(a, i, j):
    """Everything the reviewer needs to judge one pair, as plain data."""
    if i > j:
        i, j = j, i
    n, p, rep, feat = a["feat"]["n"], a["params"], a["reports"], a["feat"]
    k = pair_k(n, i, j)
    out = a["out"]
    score, flags, match = float(out["score"][k]), int(out["flags"][k]), int(out["match"][k])
    text_s, id_s, time_s = float(a["S"][i, j]), float(out["id"][k]), float(out["time"][k])
    gap = int(abs(feat["day"][i] - feat["day"][j]))
    ra, rb = rep.iloc[i], rep.iloc[j]
    support, against, notes = [], [], []

    for t, col in enumerate(e.ID_COLS):
        if match & (1 << t):
            f = int(feat["freq"][i, t])
            if f >= int(p["common_freq"]):
                against.append(f"Shared {ID_LABELS[col]} {ra[col]} appears in {f} reports. It may be popular or "
                               "recycled, so it counts for less.")
            else:
                support.append(f"Shared {ID_LABELS[col]} `{ra[col]}` (appears in {f} reports)")
    if match == 0:
        against.append("No shared identifier: this link rests only on wording and timing.")

    words = shared_terms(a["X"], a["terms"], i, j)
    if text_s >= 0.30:
        txt = f"Narratives are {text_s:.0%} similar."
        if words:
            txt += " Distinctive shared wording: " + ", ".join(f"'{w}'" for w in words) + "."
        support.append(txt)
    elif text_s < 0.15:
        against.append(f"Narratives look quite different ({text_s:.0%} similar).")
    else:
        notes.append(f"Narrative similarity is moderate ({text_s:.0%}).")

    if flags & e.FLAG_GAP:
        against.append(f"The reports are {gap} days apart, a large gap.")
    elif gap <= 30:
        support.append(f"Reported close in time ({gap} days apart)")
    else:
        notes.append(f"The reports are {gap} days apart.")

    if flags & e.FLAG_CATEGORY:
        against.append(f"Different scam categories ({ra['category']} vs {rb['category']}).")
    else:
        support.append(f"Same scam category ({ra['category']})")

    if flags & e.FLAG_CITY:
        notes.append(f"Different cities ({ra['city']} and {rb['city']}). Online fraud crosses cities, so this is information only.")
    else:
        notes.append(f"Same city ({ra['city']}).")

    penalties = {}
    if flags & e.FLAG_CATEGORY:
        penalties["different scam category"] = -p["pen_cat"]
    if flags & e.FLAG_GAP:
        penalties["large time gap"] = -p["pen_gap"]
    if flags & e.FLAG_COMMON:
        penalties["very common identifier"] = -p["pen_common"]
    tier = tier_of(score, flags)
    return dict(i=i, j=j, score=score, tier=tier, reason=tier_reason(tier, score, flags),
                text=text_s, id=id_s, time=time_s, gap_days=gap,
                parts=dict(text=p["w_text"] * text_s, identifiers=p["w_id"] * id_s, time=p["w_time"] * time_s),
                penalties=penalties, support=support, against=against, notes=notes)


# ----------------------------------------------------------------------------
# Campaigns (chaining)
# ----------------------------------------------------------------------------
def build_clusters(a, edge_threshold=CLUSTER_EDGE, accepted=(), rejected=()):
    """
    Chains links into campaigns: if A~B and B~C, then A, B, C are one campaign.
    A reviewer can force a link in (accepted) or out (rejected); pass pairs of report IDs.
    Returns (graph, cluster_of) where cluster_of[i] = campaign number (1, 2, ...), or -1 if the report is alone.
    """
    n, out = a["feat"]["n"], a["out"]
    ids = a["reports"]["report_id"].to_numpy()
    pos = {r: k for k, r in enumerate(ids)}
    rej = {frozenset(p) for p in rejected}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for k in np.where(out["score"] >= edge_threshold)[0]:
        i, j = int(a["I"][k]), int(a["J"][k])
        if frozenset((ids[i], ids[j])) in rej:
            continue
        G.add_edge(i, j, score=float(out["score"][k]), flags=int(out["flags"][k]))
    for pair in accepted:
        r1, r2 = tuple(pair)
        if r1 in pos and r2 in pos and frozenset(pair) not in rej:
            i, j = sorted((pos[r1], pos[r2]))
            k = pair_k(n, i, j)
            G.add_edge(i, j, score=float(out["score"][k]), flags=int(out["flags"][k]), accepted=True)
    cluster_of = np.full(n, -1, dtype=int)
    comps = sorted((c for c in nx.connected_components(G) if len(c) >= 2), key=lambda c: (-len(c), min(c)))
    for cid, comp in enumerate(comps, 1):
        for v in comp:
            cluster_of[v] = cid
    return G, cluster_of


def cluster_pair_prediction(a, edge_threshold=CLUSTER_EDGE):
    """For every pair: are the two reports in the same campaign? (used for evaluation)"""
    _, cl = build_clusters(a, edge_threshold=edge_threshold)
    return (cl[a["I"]] == cl[a["J"]]) & (cl[a["I"]] >= 0)


# ----------------------------------------------------------------------------
# Evaluation of tiers and chaining (unseen test data)
# ----------------------------------------------------------------------------
def campaign_stats(a, gang, y, thr):
    """Pair-level precision/recall/F1 of 'same campaign', plus purity and sizes."""
    import evaluate as ev
    G, cl = build_clusters(a, edge_threshold=thr)
    pred = (cl[a["I"]] == cl[a["J"]]) & (cl[a["I"]] >= 0)
    p, r, f = ev.prf(pred, y)
    comps = [np.where(cl == c)[0] for c in range(1, cl.max() + 1)]
    pure = np.mean([len(set(gang[c])) == 1 and gang[c][0] != "NONE" for c in comps]) if comps else 0.0
    shared = e.baseline_identifier_scores(a["feat"]) > 0
    return dict(precision=p, recall=r, f1=f, purity=float(pure), campaigns=len(comps),
                largest=max((len(c) for c in comps), default=0), recall_hard=float(pred[y & ~shared].mean()))


def main():
    import evaluate as ev
    from generate_data import generate
    res = os.path.join(HERE, "results")
    os.makedirs(res, exist_ok=True)
    params = load_params()
    pair_thr = float(pd.read_csv(os.path.join(res, "metrics_test.csv")).query("method == @ev.LINKLENS")["threshold"].iloc[0])

    def prepare(seed):
        reps, gang = e.frames_from_rows(generate(seed, 40, 160))
        return analyze(reps, params), gang, e.true_pair_labels(gang)

    # 1) choose the chaining threshold on DEV data: lowest threshold with campaign precision >= 0.99
    dev_a, dev_g, dev_y = prepare(ev.DEV_SEED)
    chosen = next(float(t) for t in np.round(np.arange(0.30, 0.70, 0.01), 2)
                  if campaign_stats(dev_a, dev_g, dev_y, t)["precision"] >= 0.99)
    print(f"Chaining threshold chosen on dev data: {chosen}  (CLUSTER_EDGE in this file = {CLUSTER_EDGE})")
    if abs(chosen - CLUSTER_EDGE) > 1e-9:
        print("  WARNING: update CLUSTER_EDGE to match")

    # 2) report on the unseen TEST datasets
    tier_rows, seed_rows, sweep_rows = [], [], []
    sweep_grid = [0.30, 0.33, 0.35, 0.37, 0.38, 0.39, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60, 0.70]
    for seed in ev.TEST_SEEDS:
        a, g, y = prepare(seed)
        sc, fl = a["out"]["score"], a["out"]["flags"]
        tiers = tiers_for(sc, fl)
        for t in ("strong", "uncertain", "weak"):
            m = tiers == t
            tier_rows.append(dict(seed=seed, tier=t, links=int(m.sum()), truly_linked=int((y & m).sum())))
        pp, pr, pf = ev.prf(sc >= pair_thr, y)
        shared = e.baseline_identifier_scores(a["feat"]) > 0
        c = campaign_stats(a, g, y, CLUSTER_EDGE)
        seed_rows.append(dict(seed=seed, pair_precision=pp, pair_recall=pr, pair_f1=pf,
                              cluster_precision=c["precision"], cluster_recall=c["recall"], cluster_f1=c["f1"],
                              pair_recall_hard=float((sc >= pair_thr)[y & ~shared].mean()),
                              cluster_recall_hard=c["recall_hard"], purity=c["purity"],
                              campaigns=c["campaigns"], largest_campaign=c["largest"]))
        for thr in sweep_grid:
            s = campaign_stats(a, g, y, thr)
            sweep_rows.append(dict(seed=seed, edge_threshold=thr, **s))

    tiers = pd.DataFrame(tier_rows).groupby("tier").agg(
        links=("links", "mean"), truly_linked=("truly_linked", "mean")).reindex(["strong", "uncertain", "weak"])
    tiers["share_truly_linked"] = tiers["truly_linked"] / tiers["links"]
    tiers.round(3).to_csv(os.path.join(res, "tier_check.csv"))

    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(os.path.join(res, "clusters_per_seed.csv"), index=False)
    summ = per_seed.drop(columns="seed").agg(["mean", "std"]).T
    summ.to_csv(os.path.join(res, "clusters.csv"))

    sweep = pd.DataFrame(sweep_rows).drop(columns="seed").groupby("edge_threshold").mean().reset_index()
    sweep.round(3).to_csv(os.path.join(res, "chaining_sweep.csv"), index=False)

    m = summ["mean"]
    L = ["# Tiers and campaign chaining", "",
         "Averages over the 5 unseen test datasets. The chaining threshold was chosen on dev data only.", "",
         "## Are the tier labels honest?", "",
         "| Tier | Suggested links per dataset | Share that are truly linked |", "|---|---|---|"]
    for t, r in tiers.iterrows():
        L.append(f"| {t} | {r['links']:.0f} | {r['share_truly_linked']:.1%} |")
    L += ["", "## Pair by pair vs chained into campaigns", "",
          "| Method | Precision | Recall | F1 | Recall on hard links |", "|---|---|---|---|---|",
          f"| Pairs scoring above the tuned threshold ({pair_thr:.2f}) | {m.pair_precision:.2f} | {m.pair_recall:.2f} | {m.pair_f1:.2f} | {m.pair_recall_hard:.2f} |",
          f"| Campaigns (links chained at {CLUSTER_EDGE:.2f}) | {m.cluster_precision:.2f} | {m.cluster_recall:.2f} | {m.cluster_f1:.2f} | {m.cluster_recall_hard:.2f} |",
          "", f"Campaigns found per dataset: {m.campaigns:.1f} (true gangs: 40). Campaigns containing a single true gang: {m.purity:.0%}.",
          "", "## What happens as the chaining threshold changes", "",
          "| Threshold | Precision | Recall | F1 | Pure campaigns | Largest campaign |", "|---|---|---|---|---|---|"]
    for _, r in sweep.iterrows():
        L.append(f"| {r.edge_threshold:.2f} | {r.precision:.2f} | {r.recall:.2f} | {r.f1:.2f} | {r.purity:.0%} | {r.largest:.0f} |")
    with open(os.path.join(res, "chaining_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
