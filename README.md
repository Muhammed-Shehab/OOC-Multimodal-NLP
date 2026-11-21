# Out-of-Context Image-Caption Detection with Multimodal NLP

A Streamlit app that checks whether a news-style caption or paragraph faithfully describes an image. It is a multimodal NLP project: the image is compared with the text directly (CLIP) and also converted to text (BLIP) so that standard NLP methods can compare text with text.

**Live demo:** https://ooc-multimodal-nlp-yf5u3de2tvbwvuamot3bqx.streamlit.app/

## What the app does

Upload an image and enter a headline or a paragraph. The app runs six stages:

| Stage | What you see | NLP concept |
|---|---|---|
| 1 | A text description of the image (BLIP) | Image captioning |
| 2 | Sentences found in your text, with the ones TextRank selected | Sentence segmentation, TextRank |
| 3 | TF-IDF, Sentence-BERT and CLIP similarity scores | Lexical vs semantic similarity |
| 4 | Entailment, neutral and contradiction probabilities | Natural language inference (NLI) |
| 5 | Named entities in your text | Named entity recognition (spaCy) |
| 6 | A verdict (pristine or out of context) with a short reason | Explainability |

**Pristine** means the caption faithfully describes the image. **Out of context** means it does not.

## Models used

| Purpose | Model |
|---|---|
| Image-text similarity | `openai/clip-vit-base-patch32` |
| Image captioning | `Salesforce/blip-image-captioning-base` |
| Sentence embeddings | `all-MiniLM-L6-v2` (Sentence-BERT) |
| Entailment | `cross-encoder/nli-distilroberta-base` |
| Tokenization, sentences, NER, POS | spaCy `en_core_web_sm` |

All models are pretrained and used without fine-tuning. The project's contribution is the pipeline, the comparison of methods, and the error analysis.

## Results

Evaluated on 1,000 test pairs (500 images, each with a true and a mismatched caption), split by image so no image appears in both training and test.

| Method | ROC-AUC |
|---|---|
| CLIP similarity | 0.997 |
| Sentence-BERT cosine | 0.965 |
| NLI (entailment minus contradiction) | 0.943 |
| TF-IDF cosine | 0.877 |
| Noun-lemma overlap | 0.875 |

Findings:

- Semantic methods beat lexical ones (Sentence-BERT 0.965 vs TF-IDF 0.877).
- For NLI, contradiction is a stronger signal than entailment.
- CLIP is best because it sees the image directly. The text pipeline inherits BLIP's description errors.
- Paragraph-level scoring beats the first sentence alone (about 0.97 vs 0.957).
- Ablation: removing nouns costs about 14.5 points, while shuffling word order costs about 0.2. The models capture topic, not syntax or facts.

## Dataset

A 2,000-image subset of Flickr8k (human-written captions). Pristine pairs are an image with its own caption. Out-of-context pairs are **synthetic**: the same image with a caption taken from a different image. Random mismatches are easier than real misinformation, so the scores above are higher than they would be on news data.

## Limitations

- Mismatches are random captions, not subtle manipulation (right topic, wrong event).
- Flickr captions are descriptive, not news-style text.
- Paragraphs in the paragraph experiment are five redundant captions of one scene.
- BLIP errors propagate into every text-based method.
- The models detect topic mismatch, not false facts such as a swapped place name or date. A flood photo labelled with the wrong state is usually judged pristine.
- Demo thresholds were fit on the test-set scores, so demo accuracy is slightly optimistic.
- The free CPU server makes each check take several seconds.

## Run locally

Python 3.11 is recommended.

```bash
git clone https://github.com/YOUR-USERNAME/ooc-nlp-detector.git
cd ooc-nlp-detector
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the models (about 2 GB) and takes a few minutes.

## Repository structure

```
app.py                      Streamlit application
requirements.txt            Python dependencies (includes the spaCy model)
results/step_b_scores.csv   Scores used to set the decision thresholds
README.md                   This file
```

## Deployment notes

Deployed on Streamlit Community Cloud. The spaCy model is installed through `requirements.txt` because the server does not allow downloading packages while the app is running. The free tier has limited memory (about 2.7 GB) and puts idle apps to sleep, so the first load after a pause can take a minute or two.

## Future work

- Evaluate on real news data (COSMOS, NewsCLIPpings).
- Entity linking and claim verification against a news database.
- Reverse image search to find an image's original context.
- Chunking and summarization for full articles.

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021.
- Li et al., *BLIP: Bootstrapping Language-Image Pre-training*, 2022.
- Reimers and Gurevych, *Sentence-BERT*, 2019.
- Hodosh et al., *Framing Image Description as a Ranking Task* (Flickr8k), 2013.
- Luo et al., *NewsCLIPpings*, 2021.
- Aneja et al., *COSMOS: Catching Out-of-Context Misinformation with Self-Supervised Learning*, 2021.

## Author

Muhammed Shehab
