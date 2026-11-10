# LinkLens

**Explainable Crime Linkage & Intelligence Explorer** - an explainable tool that suggests links between fictional cyber-fraud reports, shows the evidence (and the conflicting signals), groups linked reports into probable campaigns, and lets a human reviewer decide.

> All data is synthetic. No real people, numbers or accounts are used. Output is a lead for human review, never an accusation.

## Stack
Python, C (pairwise scoring engine), pandas, scikit-learn, NetworkX, Streamlit

## Project status
- [x] Step 1: synthetic data generator (`generate_data.py`)
- [x] Step 2: baselines - TF-IDF text similarity, shared-identifier rule (`engine.py`)
- [x] Step 3: C scoring engine (`scoring.c`, called from Python with ctypes)
- [x] Step 4: evaluation - precision / recall / F1, ablation, threshold sweep, speed (`evaluate.py`)
- [x] Step 5: evidence trail, confidence tiers, campaign graph, Streamlit app (`linkgraph.py`, `app.py`)
- [x] Step 6: reviewer workflow (accept / reject / unsure, saved to a log and applied to the campaigns)

## How LinkLens scores a pair of reports
For every pair of reports the C engine computes:
1. **Text similarity** - TF-IDF cosine similarity of the two narratives (numbers removed).
2. **Identifier score** - shared phone / UPI ID / URL. Rarer identifiers count more; an identifier seen in many reports counts less (it may be recycled or popular).
3. **Time score** - decays smoothly as the reports get further apart in time.
4. **Conflict flags** - different scam category, large time gap, very common identifier, different city (information only).

Final score = weighted sum of 1-3 minus penalties for flags. Every number is kept, so each suggested link can show its evidence.

## What the app shows
- **Suggested links** - each link has an evidence trail: what supports it (shared identifiers, distinctive shared wording, closeness in time), what argues against it (large gap, recycled identifier, different category), and a bar chart showing how the score adds up.
- **Confidence tiers** - strong (score >= 0.40 and nothing conflicts), uncertain (>= 0.33, or a high score with a conflicting signal), weak (hidden).
- **Campaign network** - reports linked directly or through other reports (chaining) are grouped into probable campaigns. A slider changes the chaining threshold live, and a demo switch reveals the hidden answer key.
- **Reviewer workflow** - accept, reject or mark each link unsure, with a note. Decisions are saved to `review_log.csv` and applied to the campaign graph.
- **Evaluation** - all tables and charts below.

## Results (average of 5 unseen synthetic test datasets)
Weights and thresholds are chosen on one separate dev dataset only.

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Baseline A: text similarity only | 0.18 | 0.28 | 0.22 |
| Baseline B: shared identifier only | 0.86 | 0.34 | 0.49 |
| LinkLens, pair by pair | 0.62 | 0.47 | 0.53 |
| LinkLens, chained into campaigns | 0.99 | 0.45 | 0.62 |

- **Tiers are honest:** 99% of "strong" links were truly linked, against 33% of "uncertain" ones.
- **Campaigns:** 39 found per dataset (40 true gangs); 96% of them contain a single true gang.
- **Ablation:** removing identifiers hurts most (F1 0.53 to 0.32).
- **Speed:** the C engine is about 50x faster than plain Python loops and 3-4x faster than NumPy.
- Full tables and charts: `results/`.

## How the evaluation stays honest
- Weights and thresholds are chosen on one **dev** dataset (seed 42, the data in `data/`).
- All reported results come from **5 other datasets** (seeds 101-105) never used for tuning.
- Hard cases are built in: gangs that share no identifier, recycled phone numbers, popular numbers, look-alike scams, typos.
- C, plain-Python and NumPy versions of the scoring are checked to give identical scores.
- The chaining threshold (0.40) is the lowest value where campaigns are at least 99% precise on dev data.

## Run
```
pip install -r requirements.txt
python generate_data.py     # writes data/reports.csv and data/ground_truth.csv
python evaluate.py          # pair-level evaluation -> results/
python linkgraph.py         # tiers and chaining evaluation -> results/
streamlit run app.py        # the web app (opens in the browser)
```
The C file is compiled automatically on the first run (needs `gcc`). If `gcc` is missing, the app falls back to a NumPy version of the same scoring and says so in the sidebar.

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Makes the synthetic reports and the hidden answer key |
| `scoring.c` | The pairwise scoring engine in C |
| `engine.py` | Loads data, TF-IDF, baselines, calls the C engine (plus Python and NumPy versions for checking and speed comparison) |
| `evaluate.py` | Tuning, test results, ablation, threshold sweep, speed test, charts |
| `linkgraph.py` | Suggested links, tiers, evidence trail, campaign chaining, and their evaluation |
| `app.py` | The Streamlit web app |
| `results/` | Output of `evaluate.py` and `linkgraph.py` |

## Limitations
- Data is synthetic, so results show the method works under controlled conditions, not on real crime data.
- Links with no shared identifier remain the hardest: only about 19% are found pair by pair, and chaining does not improve that (about 17%). Text similarity alone is too generic for them.
- The chaining threshold has a sharp cliff: below about 0.38 the groups start merging different gangs (precision falls from 0.99 to 0.5 at 0.35). See the sweep on the Evaluation page.
- The conflict penalties did not measurably improve F1 in the ablation; they matter for the evidence trail and the tier rules rather than raw accuracy.
- Text similarity is a simple bag-of-words TF-IDF.
- Future work: validate on anonymised real data with an agency partnership; try sentence embeddings for the hard links.
