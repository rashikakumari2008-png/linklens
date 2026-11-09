"""
engine.py  -  the LinkLens matching engine

Three versions of the SAME scoring logic, so we can prove they agree and compare speed:
  score_c       C code (scoring.c) called through ctypes     <- the one LinkLens uses
  score_python  plain Python loops (slow, easy to read)
  score_numpy   vectorised NumPy (a fair "fast Python" competitor)

Also here: loading data, building features, TF-IDF text similarity, and the baselines.
"""
import ctypes
import math
import os
import re
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
ID_COLS = ["phone", "upi_id", "url"]

# flag bits (same as in scoring.c)
FLAG_CATEGORY, FLAG_GAP, FLAG_COMMON, FLAG_CITY = 1, 2, 4, 8
FLAG_NAMES = {FLAG_CATEGORY: "different scam category",
              FLAG_GAP: "large time gap",
              FLAG_COMMON: "identifier is very common (may be recycled)",
              FLAG_CITY: "different city (information only)"}

# The order matters: it must match the p[] layout in scoring.c
PARAM_ORDER = ["w_text", "w_id", "w_time", "tau",
               "type_phone", "type_upi", "type_url",
               "pen_cat", "pen_gap", "pen_common",
               "gap_days", "common_freq"]

DEFAULT_PARAMS = dict(w_text=0.4, w_id=0.4, w_time=0.2, tau=45.0,
                      type_phone=0.8, type_upi=0.9, type_url=0.9,
                      pen_cat=0.3, pen_gap=0.1, pen_common=0.1,
                      gap_days=120, common_freq=8)


def params_array(p):
    return np.array([p[k] for k in PARAM_ORDER], dtype=np.float64)


# ----------------------------------------------------------------------------
# Loading data and building features
# ----------------------------------------------------------------------------
def load_reports(data_dir="data"):
    """Reads reports.csv and ground_truth.csv (everything as text so phone numbers keep their zeros)."""
    reports = pd.read_csv(os.path.join(data_dir, "reports.csv"), dtype=str, keep_default_na=False)
    truth = pd.read_csv(os.path.join(data_dir, "ground_truth.csv"), dtype=str, keep_default_na=False)
    return reports, truth["gang_id"].to_numpy()


def frames_from_rows(rows):
    """Turns the rows made by generate_data.generate() into the same tables load_reports() gives."""
    reports = pd.DataFrame({
        "report_id": [f"R{i:04d}" for i in range(1, len(rows) + 1)],
        "date": [r[0].isoformat() for r in rows],
        "city": [r[1] for r in rows],
        "category": [r[2] for r in rows],
        "narrative": [r[3] for r in rows],
        "phone": [r[4] for r in rows],
        "upi_id": [r[5] for r in rows],
        "url": [r[6] for r in rows],
    })
    return reports, np.array([r[7] for r in rows])


def build_features(reports):
    """Turns the table into plain integer arrays that the C code can read."""
    n = len(reports)
    day = reports["date"].map(lambda s: date.fromisoformat(s).toordinal()).to_numpy()
    day = (day - day.min()).astype(np.int32)
    cat = pd.factorize(reports["category"])[0].astype(np.int32)
    city = pd.factorize(reports["city"])[0].astype(np.int32)

    ids = np.full((n, 3), -1, dtype=np.int32)      # which identifier value (as a number), -1 = none
    freq = np.zeros((n, 3), dtype=np.int32)        # how many reports share that value
    for t, col in enumerate(ID_COLS):
        codes = {}
        for i, v in enumerate(reports[col].to_numpy()):
            if v:
                ids[i, t] = codes.setdefault(v, len(codes))
        if codes:
            counts = np.bincount(ids[ids[:, t] >= 0, t])
            has = ids[:, t] >= 0
            freq[has, t] = counts[ids[has, t]]
    return dict(n=n, day=day, cat=cat, city=city,
                ids=np.ascontiguousarray(ids), freq=np.ascontiguousarray(freq))


def _clean(text):
    """Removes numbers (amounts) so they do not distort the text similarity."""
    return re.sub(r"[0-9,]+", " ", text.lower())


def text_similarity(narratives):
    """TF-IDF cosine similarity between every pair of narratives (n x n matrix)."""
    vec = TfidfVectorizer(preprocessor=_clean, ngram_range=(1, 2), sublinear_tf=True)
    x = vec.fit_transform(narratives)
    return np.ascontiguousarray((x @ x.T).toarray(), dtype=np.float64)


def pair_index(n):
    return np.triu_indices(n, 1)


def true_pair_labels(gang):
    """True/False for every pair: do the two reports really belong to the same gang?"""
    iu = pair_index(len(gang))
    return (gang[iu[0]] == gang[iu[1]]) & (gang[iu[0]] != "NONE")


# ----------------------------------------------------------------------------
# Version 1: C engine (through ctypes)
# ----------------------------------------------------------------------------
_LIB = None


def load_c_library():
    """Compiles scoring.c with gcc the first time (needs gcc) and loads it."""
    global _LIB
    if _LIB is not None:
        return _LIB
    src = os.path.join(HERE, "scoring.c")
    ext = ".dll" if sys.platform.startswith("win") else ".so"
    lib_path = os.path.join(HERE, "libscoring" + ext)
    if not os.path.exists(lib_path) or os.path.getmtime(lib_path) < os.path.getmtime(src):
        cmd = ["gcc", "-O2", "-shared", "-o", lib_path, src, "-lm"]
        if not sys.platform.startswith("win"):
            cmd.insert(3, "-fPIC")
        subprocess.check_call(cmd)
    lib = ctypes.CDLL(lib_path)
    ip = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    lib.score_pairs.argtypes = [ctypes.c_int, ip, ip, ip, ip, ip, dp, dp, dp, dp, ip, ip, dp]
    lib.score_pairs.restype = None
    _LIB = lib
    return lib


def score_c(feat, text_sim, params):
    n = feat["n"]
    m = n * (n - 1) // 2
    out = dict(id=np.zeros(m), time=np.zeros(m), match=np.zeros(m, dtype=np.int32),
               flags=np.zeros(m, dtype=np.int32), score=np.zeros(m))
    load_c_library().score_pairs(n, feat["day"], feat["cat"], feat["city"], feat["ids"],
                                 feat["freq"], text_sim, params_array(params),
                                 out["id"], out["time"], out["match"], out["flags"], out["score"])
    return out


# ----------------------------------------------------------------------------
# Version 2: plain Python loops (same logic, slow)
# ----------------------------------------------------------------------------
def score_python(feat, text_sim, params):
    p = params
    n = feat["n"]
    day, cat, city = feat["day"].tolist(), feat["cat"].tolist(), feat["city"].tolist()
    ids, freq = feat["ids"].tolist(), feat["freq"].tolist()
    tw = [p["type_phone"], p["type_upi"], p["type_url"]]
    m = n * (n - 1) // 2
    out = dict(id=[0.0] * m, time=[0.0] * m, match=[0] * m, flags=[0] * m, score=[0.0] * m)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            keep, match, common = 1.0, 0, False
            for t in range(3):
                a = ids[i][t]
                if a >= 0 and a == ids[j][t]:
                    f = freq[i][t]
                    keep *= 1.0 - tw[t] / (1.0 + math.log(f - 1.0))
                    match |= 1 << t
                    if f >= int(p["common_freq"]):
                        common = True
            id_s = 1.0 - keep
            gap = abs(day[i] - day[j])
            time_s = math.exp(-gap / p["tau"])
            flags = 0
            if cat[i] != cat[j]:
                flags |= 1
            if gap > int(p["gap_days"]):
                flags |= 2
            if common:
                flags |= 4
            if city[i] != city[j]:
                flags |= 8
            s = p["w_text"] * text_sim[i, j] + p["w_id"] * id_s + p["w_time"] * time_s
            if flags & 1:
                s -= p["pen_cat"]
            if flags & 2:
                s -= p["pen_gap"]
            if flags & 4:
                s -= p["pen_common"]
            out["id"][k], out["time"][k], out["match"][k] = id_s, time_s, match
            out["flags"][k], out["score"][k] = flags, max(s, 0.0)
            k += 1
    return {key: np.array(v) for key, v in out.items()}


# ----------------------------------------------------------------------------
# Version 3: vectorised NumPy (same logic, fast Python)
# ----------------------------------------------------------------------------
def score_numpy(feat, text_sim, params):
    p = params
    n = feat["n"]
    iu = pair_index(n)
    I, J = iu
    keep = np.ones(len(I))
    match = np.zeros(len(I), dtype=np.int32)
    common = np.zeros(len(I), dtype=bool)
    tw = [p["type_phone"], p["type_upi"], p["type_url"]]
    for t in range(3):
        a, b = feat["ids"][I, t], feat["ids"][J, t]
        hit = (a >= 0) & (a == b)
        f = np.where(hit, feat["freq"][I, t], 2).astype(float)
        keep = np.where(hit, keep * (1.0 - tw[t] / (1.0 + np.log(f - 1.0))), keep)
        match |= (hit.astype(np.int32) << t)
        common |= hit & (f >= int(p["common_freq"]))
    id_s = 1.0 - keep
    gap = np.abs(feat["day"][I] - feat["day"][J])
    time_s = np.exp(-gap / p["tau"])
    flags = ((feat["cat"][I] != feat["cat"][J]).astype(np.int32) * 1
             + (gap > int(p["gap_days"])).astype(np.int32) * 2
             + common.astype(np.int32) * 4
             + (feat["city"][I] != feat["city"][J]).astype(np.int32) * 8)
    s = p["w_text"] * text_sim[I, J] + p["w_id"] * id_s + p["w_time"] * time_s
    s = s - p["pen_cat"] * (flags & 1) - p["pen_gap"] * ((flags >> 1) & 1) - p["pen_common"] * ((flags >> 2) & 1)
    return dict(id=id_s, time=time_s, match=match, flags=flags, score=np.maximum(s, 0.0))


# ----------------------------------------------------------------------------
# Baselines (the methods LinkLens must beat)
# ----------------------------------------------------------------------------
def baseline_text_scores(text_sim):
    """Baseline A: how alike do the stories look? (nothing else)"""
    iu = pair_index(len(text_sim))
    return text_sim[iu]


def baseline_identifier_scores(feat):
    """Baseline B: link two reports if they share ANY identifier (1.0) else 0.0."""
    I, J = pair_index(feat["n"])
    hit = np.zeros(len(I), dtype=bool)
    for t in range(3):
        a, b = feat["ids"][I, t], feat["ids"][J, t]
        hit |= (a >= 0) & (a == b)
    return hit.astype(float)
