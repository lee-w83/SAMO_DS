import argparse, json, os, re
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict

# -----------------------------
# Config (edit if you like)
# -----------------------------
CONF_THR   = 0.25   # confidence threshold for accepting a label per sentence
KW_BOOST   = 0.15   # how much to boost scores when a keyword is present
TOP_K      = 3      # consider top-k labels per sentence
MODEL_NAME = "bhadresh-savani/bert-base-go-emotion"

# Map fine-grained GoEmotions → macro buckets
MACRO_MAP = {
    "admiration":"positive","amusement":"positive","approval":"positive","caring":"positive","desire":"positive",
    "excitement":"positive","gratitude":"positive","joy":"positive","love":"positive","optimism":"positive",
    "pride":"positive","relief":"positive",
    "anger":"negative","annoyance":"negative","disappointment":"negative","disapproval":"negative","disgust":"negative",
    "embarrassment":"negative","fear":"negative","grief":"negative","nervousness":"negative","remorse":"negative","sadness":"negative",
    "curiosity":"neutral","confusion":"neutral","realization":"neutral","surprise":"neutral","neutral":"neutral"
}

# Simple keyword clusters (emotion names must match GoEmotions label names)
KW = {
    "sadness":      ["lonely","alone","worthless","cry","down","hopeless","depressed","empty"],
    "nervousness":  ["anxious","panic","worry","worried","overthinking","nervous"],
    "anger":        ["angry","furious","mad","rage","irritated","annoyed"],
    "joy":          ["happy","grateful","excited","proud","joy","delighted"],
    "fear":         ["afraid","scared","terrified","frightened"],
    "relief":       ["relieved","better now","calmer","at ease"],
    "pride":        ["proud","accomplished"],
}

# -----------------------------
# Lazy imports for heavy libs
# -----------------------------
nlp = None          # spaCy object (or None → fallback splitter)
emo_pipe = None     # HF pipeline object

def load_spacy():
    """Load spaCy sentence splitter, else fallback to regex."""
    global nlp
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None  # fallback used

def split_sentences(text: str):
    """Return a list of sentences for a journal entry."""
    load_spacy()
    text = (text or "").strip()
    if not text:
        return []
    if nlp is None:
        # naive fallback
        return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        # fallback if spaCy yields one big chunk
        return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    return sents

# -----------------------------
# HF pipeline (modern-safe)
# -----------------------------
def ensure_pipe():
    """Create the emotion pipeline once, using CPU/GPU/MPS if available."""
    global emo_pipe
    if emo_pipe is None:
        import torch
        from transformers import pipeline
        # device: -1 = CPU; >=0 → GPU/MPS
        has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        device_id = 0 if (torch.cuda.is_available() or has_mps) else -1
        emo_pipe = pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=None,          # modern replacement for return_all_scores=True
            truncation=True,
            device=device_id
        )

def _normalize_hf_output(out):
    """
    Normalize HF outputs to a flat list[dict].
    Possibilities:
      - list[dict]
      - list[list[dict]]  (newer behavior with top_k=None)
      - dict              (if top_k=1 or other edge cases)
    """
    if isinstance(out, dict):
        return [out]
    if isinstance(out, list):
        if len(out) > 0 and isinstance(out[0], list):
            return out[0]  # unwrap one level
        return out
    raise ValueError(f"Unexpected pipeline output type: {type(out)}")

def predict_scores(sentence: str) -> dict:
    """Return {label: score} for a single sentence."""
    ensure_pipe()
    out = emo_pipe(sentence)         # may be list[list[dict]] or list[dict]
    out = _normalize_hf_output(out)
    return {d["label"]: float(d["score"]) for d in out}

# -----------------------------
# Per-sentence + aggregation
# -----------------------------
def keyword_boost(sentence: str, scores: dict) -> dict:
    """Boost certain labels when keywords appear; clip to [0,1]."""
    s = sentence.lower()
    boosted = dict(scores)
    for emo, words in KW.items():
        if any(w in s for w in words):
            if emo in boosted:
                boosted[emo] += KW_BOOST
    for k in boosted:
        boosted[k] = min(1.0, max(0.0, boosted[k]))
    return boosted

def per_sentence_labels(sentence: str):
    """Return a list of (label, score) accepted for this sentence."""
    scores = predict_scores(sentence)
    scores = keyword_boost(sentence, scores)
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    accepted = [(lab, sc) for lab, sc in items[:TOP_K] if sc >= CONF_THR]
    # ensure coverage
    if not accepted and items:
        accepted = [items[0]]
    return accepted

def aggregate_entry(sentences):
    """Aggregate sentence-level labels to entry-level final_emotion & details."""
    emo_counts = Counter()
    emo_conf = defaultdict(list)
    for s in sentences:
        for lab, sc in per_sentence_labels(s):
            emo_counts[lab] += 1
            emo_conf[lab].append(sc)
    if not emo_counts:
        return {"final_emotion":"neutral","final_macro":"neutral","avg_confidence":0.0,"details":{}}
    final_emotion = max(emo_counts.items(), key=lambda x: x[1])[0]
    final_macro = MACRO_MAP.get(final_emotion, "neutral")
    avg_conf = sum(sum(v) for v in emo_conf.values()) / max(1, sum(len(v) for v in emo_conf.values()))
    details = {
        e: {"count": int(emo_counts[e]), "avg_conf": round(sum(emo_conf[e]) / len(emo_conf[e]), 3)}
        for e in emo_counts
    }
    return {
        "final_emotion": final_emotion,
        "final_macro": final_macro,
        "avg_confidence": round(avg_conf, 3),
        "details": details
    }

def analyze_entry(text: str):
    """Full analysis for a single journal entry."""
    sents = split_sentences(text)
    agg = aggregate_entry(sents)
    return {"original_text": text, "sentences": sents, **agg}

# -----------------------------
# Memory Lane & evaluation
# -----------------------------
def ensure_dirs():
    Path("artifacts/plots").mkdir(parents=True, exist_ok=True)
    Path("artifacts/json").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

def memory_lane(df_user: pd.DataFrame, user_id: str):
    """Aggregate weekly macro counts and streaks for a single user."""
    df = df_user.copy()
    df["week_start"] = df["timestamp"].dt.to_period("W-MON").dt.start_time
    counts = pd.crosstab(df["week_start"], df["final_macro"]).reset_index()
    for col in ["positive","negative","neutral"]:
        if col not in counts.columns: counts[col] = 0
    counts["total"] = counts["positive"] + counts["negative"] + counts["neutral"]
    counts["positive_ratio"] = counts["positive"] / counts["total"].replace({0:1})

    def dom(row):
        triples = {"positive":row["positive"],"negative":row["negative"],"neutral":"neutral"}
        # Fix: ensure numeric comparison for dominant macro
        triples = {"positive": row["positive"], "negative": row["negative"], "neutral": row["neutral"]}
        return max(triples.items(), key=lambda x: x[1])[0] if row["total"] > 0 else "neutral"

    counts["dominant_macro"] = counts.apply(dom, axis=1)

    # positive streaks
    weeks_sorted = counts.sort_values("week_start")
    doms = weeks_sorted["dominant_macro"].tolist()
    best = cur = 0
    for d in doms:
        if d == "positive":
            cur += 1; best = max(best, cur)
        else:
            cur = 0

    buckets = []
    for _, r in weeks_sorted.iterrows():
        buckets.append({
            "week_start": r["week_start"].strftime("%Y-%m-%d"),
            "macro_counts": {
                "positive": int(r["positive"]),
                "negative": int(r["negative"]),
                "neutral": int(r["neutral"])
            },
            "dominant_macro": r["dominant_macro"],
            "positive_ratio": round(float(r["positive_ratio"]), 3)
        })
    ml = {"user_id": user_id, "buckets": buckets, "streaks": {"positive": {"current": cur, "best": best}}}
    out = Path(f"artifacts/json/memory_lane_{user_id}.json")
    out.write_text(json.dumps(ml, indent=2))
    return ml

def maybe_eval(df):
    """
    Optional evaluation: expects a column 'true_labels' with semicolon-separated emotions.
    Computes micro/macro F1 as multi-label binarized classification.
    """
    if "true_labels" not in df.columns:
        return None
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score, classification_report

    all_labels = sorted(set(l for d in df["details"] for l in d.keys()))
    if not all_labels:
        return None

    mlb = MultiLabelBinarizer(classes=all_labels)
    y_true = df["true_labels"].fillna("").apply(lambda s: [t.strip() for t in s.split(";") if t.strip()]).tolist()
    y_pred = df["details"].apply(lambda det: [k for k, v in det.items() if v["count"] >= 1]).tolist()

    Y_true = mlb.fit_transform(y_true)
    Y_pred = mlb.transform(y_pred)

    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    report = classification_report(Y_true, Y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)

    metrics = {"f1_micro": round(micro, 3), "f1_macro": round(macro, 3)}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(
        json.dumps({"metrics": metrics, "labels": mlb.classes_.tolist(), "report": report}, indent=2)
    )
    return metrics

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/raw/journals.csv",
                        help="CSV with columns: user_id, timestamp, text[, true_labels]")
    args = parser.parse_args()

    ensure_dirs()

    # Load input
    df = pd.read_csv(args.input_csv)

    # Normalize column names
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "date": rename_map[c] = "timestamp"
        if lc in ("journal_entry","entry","content","text"): rename_map[c] = "text"
        if lc in ("user","userid","user_id"): rename_map[c] = "user_id"
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns: raise ValueError("timestamp column required")
    if "text" not in df.columns: raise ValueError("text column required")
    if "user_id" not in df.columns: df["user_id"] = "u1"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp","text"]).copy()

    # Inference
    results = df["text"].apply(analyze_entry)
    df["final_emotion"]  = results.apply(lambda r: r["final_emotion"])
    df["final_macro"]    = results.apply(lambda r: r["final_macro"])
    df["avg_confidence"] = results.apply(lambda r: r["avg_confidence"])
    df["details"]        = results.apply(lambda r: r["details"])

    # Save processed CSV
    out_csv = Path("data/processed/journals_with_emotions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Memory lane per user
    for uid, dfg in df.groupby("user_id"):
        memory_lane(dfg, uid)

    # Optional evaluation
    metrics = maybe_eval(df)
    if metrics:
        print("Evaluation:", metrics)

    print("Done. Saved:")
    print(" -", out_csv)
    print(" - artifacts/json/memory_lane_<user>.json")
    if metrics: print(" - artifacts/metrics.json")

if __name__ == "__main__":
    main()
