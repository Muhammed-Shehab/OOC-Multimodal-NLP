import numpy as np, pandas as pd, torch, spacy, networkx as nx, streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import (CLIPModel, CLIPProcessor, BlipForConditionalGeneration,
                          BlipProcessor, AutoTokenizer, AutoModelForSequenceClassification)

st.set_page_config(page_title="NLP Out-of-Context Detector", page_icon="📰", layout="wide")
device = "mps" if torch.backends.mps.is_available() else "cpu"

@st.cache_resource(show_spinner="Loading models (first time takes 1-2 minutes)...")
def load_all():
    clip_n = "openai/clip-vit-base-patch32"
    blip_n = "Salesforce/blip-image-captioning-base"
    nli_n = "cross-encoder/nli-distilroberta-base"
    m = {}
    m["clip"] = CLIPModel.from_pretrained(clip_n).to(device).eval()
    m["clip_p"] = CLIPProcessor.from_pretrained(clip_n)
    m["blip"] = BlipForConditionalGeneration.from_pretrained(blip_n).to(device).eval()
    m["blip_p"] = BlipProcessor.from_pretrained(blip_n)
    m["sbert"] = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    m["nli_t"] = AutoTokenizer.from_pretrained(nli_n)
    m["nli"] = AutoModelForSequenceClassification.from_pretrained(nli_n).to(device).eval()
    try:
        m["nlp"] = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        m["nlp"] = spacy.load("en_core_web_sm")
    return m

@st.cache_data
def load_thresholds():
    df = pd.read_csv("results/step_b_scores.csv")
    y = df["label"].values
    def best(col):
        s = df[col].values
        cands = np.quantile(s, np.linspace(0, 1, 200))
        return float(max(cands, key=lambda c: ((s < c).astype(int) == y).mean()))
    tfidf = TfidfVectorizer(stop_words="english").fit(list(df["caption"]) + list(df["blip"]))
    return {"clip": best("CLIP similarity"), "sbert": best("Sentence-BERT cosine"),
            "nli": best("NLI entail minus contradict"), "tfidf": best("TF-IDF cosine")}, tfidf

M = load_all()
THR, TFIDF = load_thresholds()
nlp, sbert = M["nlp"], M["sbert"]

def describe(image):
    inp = M["blip_p"](images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = M["blip"].generate(**inp, max_new_tokens=30, num_beams=3)
    return M["blip_p"].batch_decode(ids, skip_special_tokens=True)[0].strip()

def textrank(sents, k=2):
    emb = sbert.encode(sents, normalize_embeddings=True)
    sim = np.clip(emb @ emb.T, 0, None)
    np.fill_diagonal(sim, 0)
    pr = nx.pagerank(nx.from_numpy_array(sim))
    top = sorted(pr, key=pr.get, reverse=True)[:k]
    return [sents[i] for i in sorted(top)], pr

def nli_probs(premise, hypothesis):
    enc = M["nli_t"](premise, hypothesis, return_tensors="pt", truncation=True,
                     max_length=128).to(device)
    with torch.no_grad():
        p = M["nli"](**enc).logits.softmax(-1)[0].cpu().numpy()
    labels = {int(k): v.lower() for k, v in M["nli"].config.id2label.items()}
    return {labels[i]: float(p[i]) for i in range(len(p))}

def clip_sim(image, text):
    inp = M["clip_p"](text=[text], images=image, return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(device)
    with torch.no_grad():
        out = M["clip"](**inp)
    a = out.image_embeds[0].cpu().numpy(); b = out.text_embeds[0].cpu().numpy()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

st.title("📰 Out-of-Context Detector")
st.write("Upload an image and enter a headline or a full paragraph.")

left, right = st.columns([1, 2])
with left:
    file = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    if file:
        st.image(Image.open(file).convert("RGB"), use_container_width=True)
with right:
    text = st.text_area("Headline or article paragraph", height=180,
                        placeholder="A dog runs across a grassy field.")
    go = st.button("Analyse", type="primary")

if go:
    if file is None or not text.strip():
        st.warning("Please upload an image and enter some text.")
        st.stop()
    image = Image.open(file).convert("RGB")
    text = text.strip()

    # Stage 1: image -> text
    desc = describe(image)
    st.subheader("1. Image description (BLIP captioning)")
    st.info(desc)

    # Stage 2: sentence segmentation + TextRank
    sents = [s.text.strip() for s in nlp(text).sents if len(s.text.strip()) > 3]
    st.subheader("2. Sentence segmentation and TextRank")
    st.write(f"Found **{len(sents)}** sentence(s).")
    if len(sents) > 2:
        chosen, pr = textrank(sents, 2)
        st.dataframe(pd.DataFrame({"Sentence": sents, "TextRank score": [round(pr[i], 3) for i in range(len(sents))],
                                   "Selected": [s in chosen for s in sents]}),
                     hide_index=True, use_container_width=True)
    else:
        chosen = sents or [text]
        st.caption("Short text: all sentences are used.")
    used = " ".join(chosen)

    # Stage 3: similarity scores
    e = sbert.encode([used, desc], normalize_embeddings=True)
    s_sbert = float(e[0] @ e[1])
    t = TFIDF.transform([used, desc])
    s_tfidf = float(t[0].multiply(t[1]).sum())
    s_clip = clip_sim(image, used)

    st.subheader("3. Similarity scores")
    c1, c2, c3 = st.columns(3)
    c1.metric("TF-IDF (lexical)", f"{s_tfidf:.3f}", f"{s_tfidf - THR['tfidf']:+.3f} vs thr")
    c2.metric("Sentence-BERT (semantic)", f"{s_sbert:.3f}", f"{s_sbert - THR['sbert']:+.3f} vs thr")
    c3.metric("CLIP (image-text)", f"{s_clip:.3f}", f"{s_clip - THR['clip']:+.3f} vs thr")
    st.caption("Positive difference = consistent. Thresholds were learned on your test scores from Step B.")

    # Stage 4: NLI
    probs = nli_probs(desc, used)
    ent = next(v for k, v in probs.items() if "entail" in k)
    con = next(v for k, v in probs.items() if "contra" in k)
    neu = next(v for k, v in probs.items() if "neutral" in k)
    st.subheader("4. Textual entailment (NLI)")
    st.caption("Premise = image description. Hypothesis = your text.")
    n1, n2, n3 = st.columns(3)
    n1.metric("Entailment", f"{ent:.0%}"); n2.metric("Neutral", f"{neu:.0%}"); n3.metric("Contradiction", f"{con:.0%}")

    # Stage 5: NER
    doc = nlp(used)
    st.subheader("5. Named entities in your text")
    if doc.ents:
        st.dataframe(pd.DataFrame([(e_.text, e_.label_) for e_ in doc.ents],
                                  columns=["Entity", "Type"]), hide_index=True)
        if any(e_.label_ in ("GPE", "LOC", "FAC") for e_ in doc.ents):
            st.warning("A place name was found, but none of these models can verify it against the "
                       "image. A swapped location (e.g. Kerala vs Punjab) usually goes undetected.")
    else:
        st.write("No named entities found.")

    # Stage 6: verdict
    flags = {"CLIP": s_clip < THR["clip"], "Sentence-BERT": s_sbert < THR["sbert"],
             "NLI": (ent - con) < THR["nli"]}
    n_flag = sum(flags.values())
    st.subheader("6. Verdict")
    if n_flag >= 2:
        st.error(f"⚠️ Likely OUT OF CONTEXT ({n_flag} of 3 methods flagged: "
                 f"{', '.join(k for k, v in flags.items() if v)})")
    else:
        st.success(f"✅ Likely PRISTINE ({3 - n_flag} of 3 methods found the text consistent)")
    reasons = []
    if con > 0.5:
        reasons.append(f"the image description contradicts the text ({con:.0%})")
    if s_sbert < THR["sbert"]:
        reasons.append("the text is semantically far from the image description")
    if s_clip < THR["clip"]:
        reasons.append("CLIP finds low image-text similarity")
    if reasons:
        st.write("**Why:** " + "; ".join(reasons) + ".")
    st.caption("Limitation: these models detect topic mismatch, not false facts such as a wrong place, date or name.")
