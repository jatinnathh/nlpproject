# Robust Hindi Fake News Detection via Pre-trained MuRIL Fine-Tuning and Leakage-Free Preprocessing

**Course / Context:** Natural Language Processing Laboratory Project  
**Domain:** Indic NLP, Vernacular Misinformation Detection, Deep Transfer Learning  
**Date:** October 2026  

---

## Executive Summary

The proliferation of online misinformation across regional and Indic languages presents a formidable challenge to digital content moderation. While automated fact-checking pipelines have reached high maturity for English, low-to-medium resource languages like Hindi face acute vulnerabilities due to morphological complexity, code-mixing, dialectal variation, and scarce curated benchmarks.

This project delivers an end-to-end NLP pipeline for automated Hindi fake news detection:
1. **Curated & Audited Benchmark:** We conducted a systematic data audit, eliminating duplicate posts, resolving conflicting annotations, removing degenerate texts, and preventing inter-split data leakage, producing a clean corpus of **16,898 instances** split stratified into 70% Train (11,828), 15% Validation (2,535), and 15% Test (2,535).
2. **Empirical Benchmarking:** We developed a classical baseline (**TF-IDF unigram/bigram + regularized Logistic Regression**) and fine-tuned a state-of-the-art Indic transformer encoder (**Google's MuRIL — Multilingual Representations for Indian Languages**).
3. **Imbalance-Aware Training:** We implemented an inverse frequency class-weighted Cross-Entropy loss ($w_{\text{real}} \approx 0.416, w_{\text{fake}} \approx 0.584$) alongside early stopping monitored via validation Macro-F1.
4. **Significant Performance Gains:** On the independent held-out test set ($N=2,535$), **MuRIL achieves 89.03% Accuracy and 0.8800 Macro-F1**, outperforming the baseline by **+10.41% Accuracy** and **+10.94% Macro-F1**. Most critically, MuRIL achieved **93.0% precision** and **79.0% recall** on detecting fake news articles.

---

## 1. Problem Statement & Motivation

Misinformation distributed via encrypted messaging platforms (e.g., WhatsApp) and social media feeds in India is predominantly composed in vernacular languages. Traditional English-centric moderation techniques often translate Hindi text to English before classification, which introduces translation errors, loses regional cultural subtext, and adds latency. 

Key challenges in native Hindi fake news detection include:
- **Devanagari Orthography & Morphology:** Rich inflectional forms degrade bag-of-words tokenizers.
- **Transliteration & Code-Mixing:** Frequent inclusion of English terms and Romanized Hindi (Hinglish).
- **Label Inconsistency in Public Corpora:** Conflicting labels assigned to identical viral messages across web scraping runs.
- **Class Imbalance:** Real news stories naturally outnumber verified debunks, causing naive classifiers to exhibit high false-negative rates on fake articles.

---

## 2. Dataset Audit & Preprocessing Protocol

### 2.1 Audit Findings in Raw Data
Before modeling, exploratory data analysis across the raw corpus identified three major anomalies:
- **Exact & Near Duplicates:** Viral rumors appeared dozens of times with trivial whitespace differences.
- **Conflicting Labels:** Identical text claims were tagged as `0` (Real) in one batch and `1` (Fake) in another.
- **Degenerate Utterances:** Ultra-short snippets ($<5$ words) lacking any factual assertion to verify.

### 2.2 Cleaning & Leakage Prevention Pipeline
- **Conflict Resolution:** All duplicate texts carrying contradictory labels were permanently discarded to prevent ambiguous training signals.
- **Deduplication:** Repeated instances were collapsed into single representative records.
- **Length Thresholding:** Texts under 5 words were filtered out.
- **Cross-Split Verification:** Character n-gram similarity checks were run to guarantee zero overlap across partitions.

### 2.3 Partition Statistics

| Split | Real (0) | Fake (1) | Total Instances | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 6,907 | 4,921 | **11,828** | 70.0% |
| **Validation** | 1,480 | 1,055 | **2,535** | 15.0% |
| **Test (Held-Out)** | 1,480 | 1,055 | **2,535** | 15.0% |
| **Total Cleaned** | **9,867** | **7,031** | **16,898** | **100.0%** |

### 2.4 Tokenization & Length Profiling
Using the `google/muril-base-cased` subword tokenizer:
- 90th percentile of tokenized lengths fell within 220 subword tokens.
- Setting `max_length = 256` covers **>93.4%** of articles completely without truncation.
- Long texts are truncated at 256 tokens, preserving the headline and lead paragraph where claims reside.

---

## 3. Methodology & Model Architectures

```
[Raw Hindi News Corpus] 
          │
          ▼
[Audit & Cleaning: Deduplication + Conflict Removal + Length Filtering]
          │
          ▼
[Leakage-Free Stratified Split: Train (11,828) / Val (2,535) / Test (2,535)]
          │
    ┌─────┴─────────────────────────────────┐
    ▼                                       ▼
[Baseline Path]                     [Transformer Path]
TF-IDF (1-2 ngrams, max 10k)        MuRIL Tokenizer (Devanagari subwords)
Custom Hindi Stopword Removal               │
    │                                       ▼
Logistic Regression (L2, max_iter=1000)  google/muril-base-cased
    │                                 + Classification Head (Dropout + Linear)
    │                                       │
    │                                       ▼
    │                               Weighted Cross-Entropy Loss
    │                               + AdamW (lr=2e-5, wd=0.01)
    │                               + Early Stopping (Macro-F1, patience=2)
    │                                       │
    └──────────────────┬────────────────────┘
                       ▼
           [Test Evaluation (N=2,535)]
           [Detailed Error Analysis]
```

### 3.1 Baseline: TF-IDF + Logistic Regression
- **Feature Extraction:** Subword-aware unigrams and bigrams ($n \in \{1, 2\}$), max 10,000 features.
- **Hindi Stopwords Pruned:** 13 frequent grammatical particles (`के`, `में`, `की`, `है`, `और`, `को`, `से`, `हैं`, `कि`, `का`, `लिए`, `पर`, `यह`).
- **Classifier:** $L_2$-regularized Logistic Regression solved with L-BFGS.

### 3.2 Transformer: MuRIL (`google/muril-base-cased`)
- **Backbone:** 12 layers, 768 hidden dimensions, 12 attention heads.
- **Pre-training:** Pre-trained on 17 Indian languages and English across monolingual text and translated/transliterated pairs, yielding superior Devanagari representations compared to mBERT or XLM-RoBERTa.
- **Classification Head:** Linear projection on top of the pooled `[CLS]` token representation with dropout $p = 0.1$.
- **Class-Weighted Loss:**
  $$\mathcal{L} = - \sum_{i=1}^B w_{y_i} \log \hat{y}_{i, y_i}$$
  where $w_0 \approx 0.416$ (Real) and $w_1 \approx 0.584$ (Fake).
- **Optimization:** AdamW ($\text{lr} = 2 \times 10^{-5}$, weight decay $= 0.01$), batch size $= 16$, early stopping on validation Macro-F1.

---

## 4. Experimental Results

Both models were evaluated on the exact same, independent held-out test split of **2,535 instances** (1,480 Real, 1,055 Fake).

### 4.1 Overall Performance Comparison

| Metric | TF-IDF + Logistic Regression | Fine-Tuned MuRIL | Delta / Gain |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **78.62%** | **89.03%** | **+10.41%** |
| **Macro Precision** | 0.7932 | **0.9000** | **+0.1068** |
| **Macro Recall** | 0.7640 | **0.8800** | **+0.1160** |
| **Macro-F1** | **0.7706** | **0.8800** | **+0.1094** |
| **Weighted-F1** | 0.7806 | **0.8900** | **+0.1094** |

### 4.2 Class-Wise Performance Breakdown

| Model | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | Real (0) | 0.7733 | 0.8966 | 0.8304 | 1,480 |
| | Fake (1) | 0.8132 | 0.6313 | 0.7108 | 1,055 |
| **MuRIL Fine-Tuned** | **Real (0)** | **0.8700** | **0.9600** | **0.9100** | 1,480 |
| | **Fake (1)** | **0.9300** | **0.7900** | **0.8600** | 1,055 |

### 4.3 Confusion Matrix Analysis

#### Baseline (TF-IDF + Logistic Regression)
```
                  Predicted Real (0)    Predicted Fake (1)
Actual Real (0)          1,327                  153
Actual Fake (1)            389                  666
```

#### MuRIL (Fine-Tuned Transformer)
```
                  Predicted Real (0)    Predicted Fake (1)
Actual Real (0)          1,420                   60
Actual Fake (1)            218                  837
```

#### Key Takeaways:
1. **False Positives (Real misflagged as Fake):** Dropped from 153 to 60 (**60.8% reduction**). Critical for avoiding censorship of legitimate journalism.
2. **False Negatives (Fake slipping past):** Dropped from 389 to 218 (**44.0% reduction**). The transformer successfully flagged 171 deceptive articles missed by the baseline.
3. **High Fake Precision (93.0%):** When MuRIL flags an article as fake news, it is correct in 93 out of 100 cases.

---

## 5. Qualitative Error Analysis

Inspection of high-confidence misclassifications revealed four distinct error patterns:

1. **Ground-Truth Noise in Web-Scraped Sources:** A small fraction of errors represents mislabeled original source data where real news releases were tagged as fake by unreliable online aggregators.
2. **Journalistic Style Mimicry:** Advanced propaganda articles often mirror authentic press release framing (formal salutations, quotes, statistics), which presents a challenge when relying strictly on linguistic cues without external knowledge retrieval.
3. **Satire & Sarcasm:** Humorous or satirical posts without overt markers are occasionally misclassified as deceptive claims.
4. **Code-Mixed / Hinglish Named Entities:** Texts containing heavy mixing of English loanwords in Roman script paired with Devanagari verbs create fragmented subwords, reducing embedding confidence.

---

## 6. Recommendations & Future Scope

1. **Retrieval-Augmented Verification (RAG):** Connect the classifier to a live verified database (e.g., PIB Fact Check or Vishwas News API) to cross-verify specific empirical claims.
2. **Multimodal Extension:** Extend the pipeline to analyze viral forward images, video thumbnails, and accompanying captions.
3. **Ensemble Modeling:** Combine MuRIL with XLM-RoBERTa and IndicBERT to leverage complementary representations.

---

## 7. How to Compile / View the LaTeX Report

The full publication-ready LaTeX paper is saved at [`report/report.tex`](file:///Users/rs/Desktop/nlpLabProject/report/report.tex).

To compile into PDF:
- **Overleaf:** Upload `report/report.tex` directly to [Overleaf](https://www.overleaf.com) and click **Recompile**.
- **CLI (if TeX is installed):**
  ```bash
  cd report
  pdflatex report.tex && pdflatex report.tex
  ```
