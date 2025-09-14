import argparse, json, os, re
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict

# --- config (edit here) ---
CONF_THR = 0.25         # confidence threshold for accepting a label
KW_BOOST = 0.15         # how much to boost keywords if confidence is low
TOP_K = 3               # how many labels to consider per sentence
MODEL_NAME = "bhadresh-savani/bert-base-go-emotion"

MACRO_MAP = {
    "admiration":"positive","amusement":"positive","approval":"positive","caring":"positive","desire":"positive",
    "excitement":"positive","gratitude":"positive","joy":"positive","love":"positive","optimism":"positive",
    "pride":"positive","relief":"positive",
    "anger":"negative","annoyance":"negative","disappointment":"negative","disapproval":"negative","disgust":"negative",
    "embarrassment":"negative","fear":"negative","grief":"negative","nervousness":"negative","remorse":"negative","sadness":"negative",
    "curiosity":"neutral","confusion":"neutral","realization":"neutral","surprise":"neutral","neutral":"neutral"
}

# simple keyword clusters (extend if you have better lists)
KW = {
    "sadness": ["lonely","alone","worthless","cry","down","hopeless","depressed","empty"],
    "anxiety": ["anxious","panic","worry","worried","overthinking","nervous"],
    "anger": ["angry","furious","mad","rage","irritated","annoyed"],
    "joy": ["happy","grateful","excited","proud","joy","delighted"],
    "fear": ["afraid","scared","terrified","frightened"],
}

# --- lazy imports for heavy libs (faster CLI parse) ---
nlp = None
pipe = None

def load_spacy():
    global nlp
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # fallback: naive sentence split
            nlp = None

def split_sentences(text: str):
    load_spacy()
    text = (text or "").strip()
    if not text:
        return []
    if nlp is None:
        # naive split
        return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        # fallback if spacy returns one big chunk
        return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    return sents

def load_model():
    global pipe
    if pipe is None:
        from transformers import pipeline
        pipe = pipeline("text-classification", model=MODEL_NAME, return_all_scores=True, truncation=True)

def predict_scores(text: str):
    load_model()
    out = pipe(text, top_k=None)[0]  # list of dicts: {'label': 'joy', 'score': 0.87}
    return {d['label']: float(d['score']) for d in out}

def keyword_boost(sentence: str, scores: dict):
    s = sentence.lower()
    boosted = dict(scores)
    for emo, words in KW.items():
        if any(w in s for w in words):
            # if this emotion exists in the model labels, boost it
            if emo in boosted:
                boosted[emo] += KW_BOOST
    # clip to [0,1]
    for k in boosted:
        if boosted[k] > 1.0: boosted[k] = 1.0
        if boosted[k] < 0.0: boosted[k] = 0.0
    return boosted

def per_sentence_labels(sentence: str):
    scores = predict_scores(sentence)
    scores = keyword_boost(sentence, scores)
    # take top-k above threshold
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    accepted = [(lab, sc) for lab, sc in items[:TOP_K] if sc >= CONF_THR]
    # if nothing above threshold, keep the top-1 anyway for coverage
    if not accepted and items:
        accepted = [items[0]]
    return accepted  # list of (label, score)

def aggregate_entry(sentences):
    emo_counts = Counter()
    emo_conf = defaultdict(list)
    for s in sentences:
        for lab, sc in per_sentence_labels(s):
            emo_counts[lab] += 1
            emo_conf[lab].append(sc)
    if not emo_counts:
        return {"final_emotion":"neutral","final_macro":"neutral","avg_confidence":0.0,"details":{}}
    final_emotion = max(emo_counts.items(), key=lambda x: x[1])[0]
    # macro
    final_macro = MACRO_MAP.get(final_emotion, "neutral")
    # avg conf
    avg_conf = sum([sum(v) for v in emo_conf.values()]) / max(1, sum([len(v) for v in emo_conf.values()]))
    # details
    details = {e: {"count": int(emo_counts[e]), "avg_conf": round(sum(emo_conf[e])/len(emo_conf[e]), 3)} for e in emo_counts}
    return {
        "final_emotion": final_emotion,
        "final_macro": final_macro,
        "avg_confidence": round(avg_conf,3),
        "details": details
    }

def analyze_entry(text: str):
    sents = split_sentences(text)
    agg = aggregate_entry(sents)
    return {
        "original_text": text,
        "sentences": sents,
        **agg
    }

def ensure_dirs():
    Path("artifacts/plots").mkdir(parents=True, exist_ok=True)
    Path("artifacts/json").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

def memory_lane(df_user: pd.DataFrame, user_id: str):
    # expects columns: timestamp (datetime), final_macro (str)
    df = df_user.copy()
    df["week_start"] = df["timestamp"].dt.to_period("W-MON").dt.start_time
    counts = pd.crosstab(df["week_start"], df["final_macro"]).reset_index()
    for col in ["positive","negative","neutral"]:
        if col not in counts.columns: counts[col] = 0
    counts["total"] = counts["positive"]+counts["negative"]+counts["neutral"]
    counts["positive_ratio"] = counts["positive"] / counts["total"].replace({0:1})
    # dominant
    def dom(row):
        triples = {"positive":row["positive"],"negative":row["negative"],"neutral":row["neutral"]}
        return max(triples.items(), key=lambda x: x[1])[0] if row["total"]>0 else "neutral"
    counts["dominant_macro"] = counts.apply(dom, axis=1)

    # streaks (positive)
    weeks_sorted = counts.sort_values("week_start")
    doms = weeks_sorted["dominant_macro"].tolist()
    best = cur = 0
    for d in doms:
        if d=="positive":
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
            "positive_ratio": round(float(r["positive_ratio"]),3)
        })
    ml = {"user_id": user_id, "buckets": buckets, "streaks": {"positive": {"current": cur, "best": best}}}
    out = Path(f"artifacts/json/memory_lane_{user_id}.json")
    out.write_text(json.dumps(ml, indent=2))
    return ml

def maybe_eval(df):
    """
    Optional evaluation: expects a column 'true_labels' with semicolon-separated emotions
    We compute micro/macro F1 on a multi-label binarized basis.
    """
    if "true_labels" not in df.columns:
        return None
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score, classification_report

    # predicted: collect from details counts (multi-label: take any label with count >=1)
    all_labels = sorted(set(l for d in df["details"] for l in d.keys()))
    mlb = MultiLabelBinarizer(classes=all_labels)

    y_true = df["true_labels"].fillna("").apply(lambda s: [t.strip() for t in s.split(";") if t.strip()]).tolist()
    y_pred = df["details"].apply(lambda det: [k for k,v in det.items() if v["count"]>=1]).tolist()

    Y_true = mlb.fit_transform(y_true)
    Y_pred = mlb.transform(y_pred)

    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    report = classification_report(Y_true, Y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)

    metrics = {"f1_micro": round(micro,3), "f1_macro": round(macro,3)}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(json.dumps({"metrics":metrics, "labels": mlb.classes_.tolist()}, indent=2))
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/raw/journals.csv",
                        help="CSV with columns: user_id, timestamp, text[, true_labels]")
    args = parser.parse_args()

    ensure_dirs()

    df = pd.read_csv(args.input_csv)
    # normalize columns
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "date": rename_map[c] = "timestamp"
        if lc in ("journal_entry","entry","content","text"): rename_map[c] = "text"
        if lc in ("user","userid","user_id"): rename_map[c] = "user_id"
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns: raise ValueError("timestamp column required")
    if "text" not in df.columns: raise ValueError("text column required")
    if "user_id" not in df.columns: df["user_id"] = "u1"  # fallback
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp","text"]).copy()

    # inference
    results = df["text"].apply(analyze_entry)
    df["final_emotion"] = results.apply(lambda r: r["final_emotion"])
    df["final_macro"]   = results.apply(lambda r: r["final_macro"])
    df["avg_confidence"]= results.apply(lambda r: r["avg_confidence"])
    df["details"]       = results.apply(lambda r: r["details"])

    # save processed
    out_csv = Path("data/processed/journals_with_emotions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # memory lane per user
    for uid, dfg in df.groupby("user_id"):
        memory_lane(dfg, uid)

    # optional evaluation if true labels present
    metrics = maybe_eval(df)
    if metrics:
        print("Evaluation:", metrics)

    print("Done. Saved:")
    print(" -", out_csv)
    print(" - artifacts/json/memory_lane_<user>.json")
    if metrics: print(" - artifacts/metrics.json")

if __name__ == "__main__":
    main()
