# LinkLens

**Explainable Crime Linkage & Intelligence Explorer** - an explainable tool that suggests links between fictional cyber-fraud reports, shows the evidence (and the conflicting signals), and lets a human reviewer decide.

> All data is synthetic. No real people, numbers or accounts are used. Output is a lead for human review, never an accusation.

## Stack
Python, C (pairwise scoring engine), pandas, scikit-learn, NetworkX, Streamlit

## Project status
- [x] Step 1: synthetic data generator (`generate_data.py`)
- [x] Step 2: baselines - TF-IDF text similarity, shared-identifier rule (`engine.py`)
- [x] Step 3: C scoring engine (`scoring.c`, called from Python with ctypes)
- [x] Step 4: evaluation - precision / recall / F1, ablation, threshold sweep, speed (`evaluate.py`)
- [ ] Step 5: network graph and Streamlit app
- [ ] Step 6: reviewer workflow

## How LinkLens scores a pair of reports
For every pair of reports the C engine computes:
1. **Text similarity** - TF-IDF cosine similarity of the two narratives (numbers removed).
2. **Identifier score** - shared phone / UPI ID / URL. Rarer identifiers count more; an identifier seen in many reports counts less (it may be recycled or popular).
3. **Time score** - decays smoothly as the reports get further apart in time.
4. **Conflict flags** - different scam category, large time gap, very common identifier, different city (information only).

Final score = weighted sum of 1-3 minus penalties for flags. Every number is kept, so each suggested link can show its evidence.

## How the evaluation stays honest
- Weights and thresholds are chosen on one **dev** dataset (seed 42, the data in `data/`).
- All reported results come from **5 other datasets** (seeds 101-105) never used for tuning.
- Hard cases are built in: gangs that share no identifier, recycled phone numbers, popular numbers, look-alike scams, typos.
- C, plain-Python and NumPy versions of the scoring are checked to give identical scores.

## Run
```
pip install -r requirements.txt
python generate_data.py     # writes data/reports.csv and data/ground_truth.csv
python evaluate.py          # writes results/ (tables, charts, results_summary.md)
```
The C file is compiled automatically on the first run (needs `gcc`).

### Run in Google Colab (free, no laptop setup)
Paste these into Colab cells and run them:
```
!git clone https://github.com/rashikakumari2008-png/linklens.git
%cd linklens
!pip install -q -r requirements.txt
!python evaluate.py
```

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Makes the synthetic reports and the hidden answer key |
| `scoring.c` | The pairwise scoring engine in C |
| `engine.py` | Loads data, TF-IDF, baselines, calls the C engine (plus Python and NumPy versions for checking and speed comparison) |
| `evaluate.py` | Tuning, test results, ablation, threshold sweep, speed test, charts |
| `results/` | Output of `evaluate.py` |

## Limitations
- Data is synthetic, so results show the method works under controlled conditions, not on real crime data.
- Text similarity is a simple bag-of-words TF-IDF; links with no shared identifier remain the hardest to find.
- Future work: validate on anonymised real data with an agency partnership.
