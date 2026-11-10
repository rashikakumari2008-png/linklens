"""
app.py  -  the LinkLens web app (Streamlit)

Run:  streamlit run app.py
Pages: Overview | Suggested links | Campaign network | Evaluation | Review log
"""
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

import engine as e
import linkgraph as lg

st.set_page_config(page_title="LinkLens", page_icon="🔎", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
LOG = os.path.join(HERE, "review_log.csv")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CAT_COLORS = {"kyc": BLUE, "refund": ORANGE, "job": AQUA, "loan": YELLOW}
TIER_ICON = {"strong": "🟢 strong", "uncertain": "🟡 uncertain", "weak": "⚪ weak"}
INK, INK2 = "#0b0b0b", "#52514e"


# ----------------------------------------------------------------------------
# Data (cached so the app stays fast)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Scoring all pairs of reports with the C engine...")
def get_analysis():
    if not os.path.exists(os.path.join(DATA, "reports.csv")):
        import generate_data
        generate_data.save(generate_data.generate(42, 40, 160), DATA)
    reports, gang = e.load_reports(DATA)
    return lg.analyze(reports), gang


@st.cache_data
def get_links(_a, min_score):
    return lg.suggested_links(_a, min_score)


def load_log():
    if os.path.exists(LOG):
        return pd.read_csv(LOG, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=["time", "report_a", "report_b", "score", "tier", "decision", "note"])


def save_decision(row, decision, note):
    log = load_log()
    log = log[~((log.report_a == row["report_a"]) & (log.report_b == row["report_b"]))]   # newest decision wins
    new = dict(time=datetime.now().strftime("%Y-%m-%d %H:%M"), report_a=row["report_a"], report_b=row["report_b"],
               score=f"{row['score']:.3f}", tier=row["tier"], decision=decision, note=note)
    pd.concat([log, pd.DataFrame([new])]).to_csv(LOG, index=False)


def current_decision(row):
    log = load_log()
    m = log[(log.report_a == row["report_a"]) & (log.report_b == row["report_b"])]
    return m.iloc[-1] if len(m) else None


a, gang = get_analysis()
reports = a["reports"]
n = len(reports)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🔎 LinkLens")
st.sidebar.caption("Explainable crime linkage & intelligence explorer")
page = st.sidebar.radio("Page", ["Overview", "Suggested links", "Campaign network", "Evaluation", "Review log"])
st.sidebar.info("All reports are **fictional**. Suggested links are leads for human review, never accusations.")
st.sidebar.caption(f"Scoring engine: {a['engine']}")


# ----------------------------------------------------------------------------
# Page: Overview
# ----------------------------------------------------------------------------
def page_overview():
    st.title("LinkLens")
    st.subheader("Finding hidden connections between fraud reports - and showing why")
    st.write(
        "Separate fraud reports often share phone numbers, payment IDs or methods, but spotting this by hand is slow. "
        "LinkLens scores every pair of reports using **story similarity**, **time**, and **shared identifiers**, "
        "shows the **evidence for and against** each suggested link, groups linked reports into probable "
        "**campaigns**, and lets a human reviewer accept or reject each lead.")
    links = get_links(a, lg.TIER_UNCERTAIN)
    _, cl = lg.build_clusters(a)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reports", n)
    c2.metric("Pairs compared", f"{len(a['I']):,}")
    c3.metric("Suggested links", len(links), help="pairs scoring at least the 'uncertain' cut-off")
    c4.metric("Probable campaigns", int(cl.max()) if cl.max() > 0 else 0)

    st.markdown("#### How a pair is scored")
    p = a["params"]
    st.markdown(
        f"`score = {p['w_text']} x story similarity + {p['w_id']} x shared identifiers + {p['w_time']} x time closeness - penalties`\n\n"
        "- **Penalties** apply for conflicting signals: a different scam category, a very large time gap, "
        "or an identifier that appears in many unrelated reports (it may be recycled or popular).\n"
        f"- **Tiers:** 🟢 strong (score ≥ {lg.TIER_STRONG:.2f} and nothing conflicts) - 🟡 uncertain (≥ {lg.TIER_UNCERTAIN:.2f}, "
        "or a high score with a conflicting signal: needs a human look) - ⚪ weak (below, hidden).\n"
        "- The scoring runs in **C**; Python handles text analysis, the graph and this app.")

    st.markdown("#### Headline results (5 unseen synthetic test sets)")
    path = os.path.join(RESULTS, "metrics_test.csv")
    if os.path.exists(path):
        m = pd.read_csv(path)
        view = m[["method", "precision", "recall", "f1", "ap"]].copy()
        view.columns = ["Method", "Precision", "Recall", "F1", "Avg. precision"]
        st.dataframe(view.round(2), hide_index=True, width="stretch")
    cpath, tpath = os.path.join(RESULTS, "clusters.csv"), os.path.join(RESULTS, "tier_check.csv")
    if os.path.exists(cpath) and os.path.exists(tpath):
        c = pd.read_csv(cpath, index_col=0)["mean"]
        t = pd.read_csv(tpath, index_col=0)
        st.caption(
            f"The confidence labels hold up: {t.loc['strong', 'share_truly_linked']:.0%} of 'strong' links were truly "
            f"linked, against {t.loc['uncertain', 'share_truly_linked']:.0%} of 'uncertain' ones. Chaining links into "
            f"campaigns lifts precision from {c['pair_precision']:.2f} to {c['cluster_precision']:.2f} "
            f"(F1 {c['pair_f1']:.2f} to {c['cluster_f1']:.2f}), but it does not help the hardest links: recall on links "
            f"with no shared identifier stays low ({c['pair_recall_hard']:.2f} pair by pair, {c['cluster_recall_hard']:.2f} chained).")
    st.warning("Limitation: the data is synthetic, so these results show the method works in controlled conditions, "
               "not on real crime data.")


# ----------------------------------------------------------------------------
# Page: Suggested links
# ----------------------------------------------------------------------------
def score_chart(ev):
    labels = ["Story similarity", "Shared identifiers", "Time closeness"] + list(ev["penalties"])
    values = [ev["parts"]["text"], ev["parts"]["identifiers"], ev["parts"]["time"]] + list(ev["penalties"].values())
    colors = [BLUE] * 3 + [ORANGE] * len(ev["penalties"])
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.38 * len(labels)))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55)
    for y, v in enumerate(values[::-1]):
        ax.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK)
    ax.axvline(0, color=INK2, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Score {ev['score']:.2f} = sum of the bars", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.tick_params(labelsize=9, colors=INK2)
    ax.set_xlim(min(-0.12, min(values) - 0.08), max(values) + 0.12)
    fig.tight_layout()
    return fig


def show_pair(row):
    ev = lg.evidence(a, int(row["i"]), int(row["j"]))
    ra, rb = reports.iloc[ev["i"]], reports.iloc[ev["j"]]
    st.markdown(f"### {ra.report_id}  ↔  {rb.report_id}   -   {TIER_ICON[ev['tier']]}  (score {ev['score']:.2f})")
    left, right = st.columns(2)
    for col, r in ((left, ra), (right, rb)):
        with col:
            st.markdown(f"**{r.report_id}** - {r.date} - {r.city} - *{r.category}*")
            st.info(r.narrative)
            ids = [f"{c.replace('_id', '').upper()}: `{r[c]}`" for c in e.ID_COLS if r[c]]
            st.caption("Identifiers: " + (" - ".join(ids) if ids else "none given"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Evidence for the link**")
        for s in ev["support"] or ["Nothing strong"]:
            st.markdown(f"- {s}")
    with c2:
        st.markdown("**⚠️ Conflicting or missing signals**")
        for s in ev["against"] or ["None found"]:
            st.markdown(f"- {s}")
    for note in ev["notes"]:
        st.caption(note)
    st.pyplot(score_chart(ev), width="content")
    if ev["tier"] == "uncertain":
        st.warning(ev["reason"] + " Treat it as a lead and check the evidence before accepting.")
    else:
        st.caption(ev["reason"])

    st.markdown("**Reviewer decision**")
    cur = current_decision(row)
    if cur is not None:
        st.success(f"Current decision: **{cur['decision']}** ({cur['time']})" + (f" - {cur['note']}" if cur["note"] else ""))
    note = st.text_input("Optional note", key=f"note_{row['pair_index']}")
    b1, b2, b3, _ = st.columns([1, 1, 1, 4])
    if b1.button("✅ Accept", key=f"acc_{row['pair_index']}"):
        save_decision(row, "accepted", note)
        st.rerun()
    if b2.button("❌ Reject", key=f"rej_{row['pair_index']}"):
        save_decision(row, "rejected", note)
        st.rerun()
    if b3.button("❓ Unsure", key=f"uns_{row['pair_index']}"):
        save_decision(row, "unsure", note)
        st.rerun()


def page_links():
    st.title("Suggested links")
    f1, f2, f3 = st.columns([2, 2, 2])
    tiers = f1.multiselect("Confidence", ["strong", "uncertain"], default=["strong", "uncertain"])
    cats = f2.multiselect("Scam category", sorted(reports.category.unique()), default=sorted(reports.category.unique()))
    q = f3.text_input("Find a report ID (e.g. R0123)")
    links = get_links(a, lg.TIER_UNCERTAIN)
    cat_of = reports.category.to_numpy()
    mask = links.tier.isin(tiers) & (pd.Series(cat_of[links.i.to_numpy()]).isin(cats).to_numpy()
                                     | pd.Series(cat_of[links.j.to_numpy()]).isin(cats).to_numpy())
    if q.strip():
        mask &= (links.report_a.str.contains(q.strip(), case=False) | links.report_b.str.contains(q.strip(), case=False))
    links = links[mask].reset_index(drop=True)
    st.caption(f"{len(links)} links match. Pick one to see the evidence trail.")
    if links.empty:
        st.info("No links match these filters.")
        return

    view = links[["report_a", "report_b", "score", "tier", "text", "identifiers", "time", "conflict"]].copy()
    view["tier"] = view["tier"].map(TIER_ICON)
    view.columns = ["Report A", "Report B", "Score", "Confidence", "Story", "Identifiers", "Time", "Conflicts"]
    st.dataframe(view.round(2), hide_index=True, width="stretch", height=260)

    options = [f"{r.report_a} ↔ {r.report_b}  -  {r.score:.2f}  -  {r.tier}" for r in links.itertuples()]
    choice = st.selectbox("Open a link", range(len(options)), format_func=lambda k: options[k])
    show_pair(links.iloc[choice])


# ----------------------------------------------------------------------------
# Page: Campaign network
# ----------------------------------------------------------------------------
def layout_components(G, comps):
    """Spring layout inside each campaign, campaigns placed on a grid so they never overlap."""
    pos, cols = {}, int(np.ceil(np.sqrt(len(comps))))
    for k, comp in enumerate(comps):
        sub = G.subgraph(comp)
        p = nx.spring_layout(sub, seed=3, k=0.9 / np.sqrt(max(len(comp), 1)), weight="score") if len(comp) > 1 else {next(iter(comp)): np.zeros(2)}
        ox, oy = (k % cols) * 3.2, -(k // cols) * 3.2
        for v, xy in p.items():
            pos[v] = (xy[0] + ox, xy[1] + oy)
    return pos


def draw_network(G, comps, show_truth, gang_ids):
    pos = layout_components(G, comps)
    nodes = [v for c in comps for v in c]
    size = 7 if len(comps) == 1 else 2.2 * np.ceil(np.sqrt(len(comps)))
    fig, ax = plt.subplots(figsize=(size * 1.25 if len(comps) > 1 else 8, size if len(comps) > 1 else 6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    def is_strong(d):
        return d.get("accepted", False) or lg.tier_of(d["score"], d["flags"]) == "strong"
    strong = [(u, v) for u, v, d in G.edges(data=True) if u in pos and v in pos and is_strong(d)]
    weak = [(u, v) for u, v, d in G.edges(data=True) if u in pos and v in pos and not is_strong(d)]
    nx.draw_networkx_edges(G, pos, edgelist=strong, width=2.4, edge_color="#52514e", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=weak, width=1.2, edge_color="#52514e", style="dashed", ax=ax)
    colors = [CAT_COLORS.get(reports.category.iloc[v], "#999999") for v in nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=260 if len(comps) == 1 else 90,
                           edgecolors="white", linewidths=1.5, ax=ax)
    if len(comps) <= 3:
        nx.draw_networkx_labels(G, pos, labels={v: reports.report_id.iloc[v][1:] for v in nodes}, font_size=7, ax=ax)
    if show_truth:
        for comp in comps:
            gangs = pd.Series([gang_ids[v] for v in comp])
            top = gangs.value_counts()
            purity = top.iloc[0] / len(comp)
            xs = [pos[v][0] for v in comp]
            ys = [pos[v][1] for v in comp]
            ax.text(np.mean(xs), max(ys) + 0.25, f"{top.index[0]}  {purity:.0%} pure", ha="center", fontsize=7,
                    color="#008300" if purity == 1 else "#e34948")
    ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=k) for k, c in CAT_COLORS.items()]
    handles += [plt.Line2D([0], [0], color="#52514e", linewidth=2.4, label="strong link"),
                plt.Line2D([0], [0], color="#52514e", linewidth=1.2, linestyle="--", label="uncertain link")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8, ncol=3 if len(comps) > 1 else 1)
    fig.tight_layout()
    return fig


def page_network():
    st.title("Campaign network")
    st.write("Reports linked (directly or through other reports) are grouped into a probable **campaign**. "
             "Solid lines are strong links, dashed lines are uncertain ones. Colour shows the scam category.")
    edge = st.slider("Link threshold for grouping", 0.30, 0.70, float(lg.CLUSTER_EDGE), 0.01,
                     help="Lower = bigger groups, more recall but more false links. Higher = smaller, safer groups.")
    log = load_log()
    acc = [(r.report_a, r.report_b) for r in log[log.decision == "accepted"].itertuples()]
    rej = [(r.report_a, r.report_b) for r in log[log.decision == "rejected"].itertuples()]
    G, cl = lg.build_clusters(a, edge_threshold=edge, accepted=acc, rejected=rej)
    if acc or rej:
        st.caption(f"Your review decisions are applied: {len(acc)} accepted link(s) forced in, {len(rej)} rejected link(s) removed.")
    comps = sorted((c for c in nx.connected_components(G) if len(c) >= 2), key=len, reverse=True)
    if not comps:
        st.info("No campaigns at this threshold.")
        return
    biggest = len(comps[0])
    st.caption(f"{len(comps)} probable campaigns. Largest has {biggest} reports." +
               ("  ⚠️ Very large group - some links in it may be false; raise the threshold." if biggest > 15 else ""))
    truth = st.toggle("Reveal hidden answer key (demo only)", value=False,
                      help="Shows the true gang for each campaign and how 'pure' the grouping is.")
    pick = st.selectbox("Show", ["All campaigns"] + [f"Campaign {k + 1} ({len(c)} reports)" for k, c in enumerate(comps)])
    chosen = comps if pick == "All campaigns" else [comps[int(pick.split()[1]) - 1]]
    st.pyplot(draw_network(G, chosen, truth, gang), width=("stretch" if pick == "All campaigns" else "content"))

    st.markdown("#### Reports in " + ("the campaigns" if pick == "All campaigns" else pick))
    rows = []
    for c in chosen:
        k = int(cl[next(iter(c))])
        for v in sorted(c):
            r = reports.iloc[v]
            row = {"Campaign": k, "Report": r.report_id, "Date": r.date, "City": r.city, "Category": r.category,
                   "Phone": r.phone, "UPI": r.upi_id, "URL": r.url}
            if truth:
                row["True gang"] = gang[v]
            rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=300)


# ----------------------------------------------------------------------------
# Page: Evaluation
# ----------------------------------------------------------------------------
def page_eval():
    st.title("Evaluation")
    st.write("Weights and thresholds were chosen on one dev dataset. Every number below comes from **5 other synthetic "
             "datasets** that were never used for tuning (± = spread across them).")
    summary = os.path.join(RESULTS, "results_summary.md")
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as f:
            st.markdown(f.read().split("\n", 1)[1])
    csum = os.path.join(RESULTS, "chaining_summary.md")
    if os.path.exists(csum):
        with open(csum, encoding="utf-8") as f:
            st.markdown(f.read().replace("# Tiers and campaign chaining", "### Tiers and campaign chaining", 1)
                        .replace("\n## ", "\n#### "))
    spath = os.path.join(RESULTS, "chaining_sweep.csv")
    if os.path.exists(spath):
        st.markdown("#### Chaining threshold: precision, recall and F1")
        st.line_chart(pd.read_csv(spath).set_index("edge_threshold")[["precision", "recall", "f1"]])
        st.caption("Below about 0.38 the groups start merging different gangs (precision collapses). "
                   "The default of 0.40 stays safely above that cliff.")
    for img, cap in [("threshold_sweep.png", "How results change as the threshold moves"),
                     ("ablation.png", "Which signals matter"), ("speed.png", "C engine vs Python")]:
        path = os.path.join(RESULTS, img)
        if os.path.exists(path):
            st.image(path, caption=cap)
    st.caption("To recompute everything: `python evaluate.py` then `python linkgraph.py`.")


# ----------------------------------------------------------------------------
# Page: Review log
# ----------------------------------------------------------------------------
def page_log():
    st.title("Review log")
    log = load_log()
    if log.empty:
        st.info("No decisions yet. Open a link on the 'Suggested links' page and accept, reject or mark it unsure.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Accepted", int((log.decision == "accepted").sum()))
    c2.metric("Rejected", int((log.decision == "rejected").sum()))
    c3.metric("Unsure", int((log.decision == "unsure").sum()))
    st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
    st.download_button("Download log (CSV)", log.to_csv(index=False), "review_log.csv", "text/csv")


{"Overview": page_overview, "Suggested links": page_links, "Campaign network": page_network,
 "Evaluation": page_eval, "Review log": page_log}[page]()
