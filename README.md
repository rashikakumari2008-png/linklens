# LinkLens

**Explainable Crime Linkage & Intelligence Explorer** - an explainable tool that suggests links between fictional cyber-fraud reports, shows the evidence (and the conflicting signals), and lets a human reviewer decide.

> All data is synthetic. No real people, numbers or accounts are used. Output is a lead for human review, never an accusation.

## Stack
Python, C (pairwise scoring engine), pandas, scikit-learn, NetworkX, Streamlit

## Project status
- [x] Step 1: synthetic data generator (`generate_data.py`)
- [ ] Step 2: TF-IDF baseline
- [ ] Step 3: C scoring engine
- [ ] Step 4: evaluation (precision / recall / F1, ablation, threshold sweep, speed)
- [ ] Step 5: network graph and Streamlit app
- [ ] Step 6: reviewer workflow

## Run
```
pip install -r requirements.txt
python generate_data.py
```
This writes `data/reports.csv` (what an investigator sees) and `data/ground_truth.csv` (hidden answer key, used only for evaluation).
