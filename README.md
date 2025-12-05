# ElmosYar_app

Professor Analytics and Recommender System



## Ethical Considerations and Bias

* The dataset may overrepresent highly motivated or dissatisfied students.
* Automated sentiment may misinterpret sarcasm or culturally specific expressions.
* Dashboard will avoid definitive judgments about instructors and include usage disclaimers.
* Recommendations are informational, not prescriptive.



## Overview

ElmosYar_app is an end-to-end data mining project designed to extract structured information from raw Telegram channel messages, analyze student feedback about professors and courses, and provide an interactive analytical dashboard along with an explainable recommender system.

The project includes a full pipeline: data ingestion, parsing, cleaning, NLP preprocessing, feature extraction, modeling, and dashboard deployment.



## Goals

* Convert unstructured Telegram feedback messages into structured, reliable data.
* Provide search, filtering, and comparison capabilities for students and faculty.
* Analyze text to extract sentiment, keywords, and topic patterns.
* Train both supervised and unsupervised models for prediction and clustering.
* Build an explainable course/professor recommender.
* Deliver a reproducible project with clear documentation and modular code.



## Data Source

The dataset consists of Telegram channel JSON exports containing course feedback messages.
Each JSON item typically includes message metadata and a single combined text field containing:

* professor name (raw)
* course name (raw)
* department (if present)
* multiple numeric ratings
* grading style descriptions
* attendance information
* free-text comments
* date or term indicator

All messages are parsed into a unified, structured format using a custom parser.


## Repository Structure

```
ElmosYar_app/
├─ data/
│  ├─ raw/
│  ├─ processed/
├─ notebooks/
│  ├─ 01_EDA.ipynb
│  ├─ 02_parsing.ipynb
│  ├─ 03_cleaning_standardization.ipynb
│  ├─ 04_nlp_feature_engineering.ipynb
│  ├─ 05_models_supervised.ipynb
│  ├─ 06_models_unsupervised.ipynb
│  └─ 07_recommender.ipynb
├─ src/
│  ├─ parser.py
│  ├─ normalizer.py
│  ├─ nlp_utils.py
│  ├─ models.py
│  └─ recommender.py
├─ app/
│  └─ streamlit_app.py
├─ requirements.txt
└─ README.md
```



## Development Pipeline

### Step 1: Load and Exploratory Data Analysis

* Read raw JSON into Pandas.
* Inspect message count, missing fields, rating distributions, professor/course frequency.
* Produce EDA visualizations.

Purpose: understand structure and limitations of the dataset before parsing.



### Step 2: Parsing

* Implement robust rule-based parsing using regex and deterministic patterns.
* Extract: professor name, course name, ratings, grading style, attendance, comment text, date/term.
* Save all failures using a `parse_error` flag for review.
* Maintain unit tests for edge cases.

Purpose: convert free-form messages into structured, machine-readable fields.



### Step 3: Cleaning and Standardization

* Normalize Persian text (Unicode cleaning, whitespace normalization, character harmonization).
* Canonicalize professor names using fuzzy matching (RapidFuzz / Levenshtein distance).
* Standardize course names.
* Map raw grading and attendance descriptions to a controlled label set.
* Produce canonical IDs for professors.

Purpose: ensure consistency across all records to avoid fragmentation in analysis.



### Step 4: NLP and Feature Engineering

Operations include:

* Persian text preprocessing (normalization, tokenization, stopword removal).
* Sentiment scoring (baseline rule-based method; transformer-based methods).
* TF-IDF features and keyword extraction.
* Topic modeling using LDA or BERTopic.
* embeddings using sentence-transformers.

Purpose: extract meaningful signals from comment text for modeling and the recommender.



## Transformer-Based Experiments 

If time allows, the project will include experiments with multiple transformer models. Only one is required for baseline performance, but the following models may be evaluated depending on resource availability:

### Candidate Transformer Models

1. ParsBERT (Hooshvare Lab)
2. XLM-RoBERTa (multilingual)
3. mBERT
4. Persian-Sentiment-BERT models (community trained)
5. Sentence-Transformer variants for embedding-based similarity

### Experimental Goals

* Compare sentiment classification performance between rule-based and transformer-based methods.
* Evaluate embedding quality for clustering and recommender similarity scoring.
* Document performance tradeoffs (accuracy vs. compute cost).

The project structure already supports plugging in different transformer backends with minimal code changes.



## Step 5: Supervised Modeling

Example tasks:

* Predict aggregated rating from text and metadata (regression).
* Classify comments into positive, neutral, negative (classification).

Models:

* Logistic Regression
* Random Forest
* SVM

Metrics:

* For classification: accuracy, F1
* For regression: MAE, RMSE
* Perform cross-validation and report results.

Purpose: provide predictive modeling insights and benchmarks.



## Step 6: Unsupervised Modeling

Cluster professors based on:

* average sentiment
* average numeric ratings
* grading style distribution
* topic vectors or text embeddings

Methods:

* KMeans
* Agglomerative clustering

Visualize clusters using PCA or UMAP.

Purpose: identify instructor groups and characteristic teaching/assessment styles.



## Step 7: Recommender System

A simple explainable recommender will be implemented with two approaches:

### Content-Based

* Build a feature vector for each professor (ratings, sentiment, topics, embeddings).
* Compute similarity using cosine similarity.
* Recommend professors similar to the user’s preferred criteria.

### Rule-Based (Fallback)

* Apply filtering based on user-selected criteria such as grading leniency, exam difficulty, teaching quality.

Optional Hybrid: combine rule filters with similarity ranking.

Purpose: provide transparent, interpretable suggestions to students.



## Step 8: Dashboard (Streamlit)

The dashboard includes:

### Overview

* Dataset summary
* Rating and sentiment distributions

### Search and Filter

* Search by professor or course
* Filter by department, term, rating range, grading style

### Professor Profile

* Rating breakdown
* Sentiment summary
* Word cloud
* Historical trends

### Compare

* Side-by-side comparison of selected professors

### Recommender

* User preference inputs
* Ranked recommendations with explanation



## Tools and Justification

### Core

* Python: data processing and reproducibility
* Pandas, NumPy: structured data handling
* regex: parsing noisy text
* RapidFuzz: name deduplication
* Scikit-learn: classical ML modeling
* Streamlit: lightweight and fast dashboard development

### NLP

* Hazm, Shekar, or similar: Persian text preprocessing
* Transformers: higher-quality semantic modeling
* BERTopic or LDA: topic extraction
* Sentence-transformers: embedding-based similarity



## Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



## Usage

### Parsing

```
python src/parser.py --input data/raw/messages.json --output data/processed/parsed.csv
```

### Feature Building

```
python src/nlp_utils.py --input data/processed/parsed.csv --output data/processed/features.parquet
```

### Modeling

```
python src/models.py --features data/processed/features.parquet --out models/
```

### Dashboard

```
streamlit run app/streamlit_app.py
```



## Design Decisions

The report will include a dedicated section detailing major design choices, such as:

* Rule-based parsing vs. ML parsing
* Rule-based sentiment vs. transformer-based sentiment
* Streamlit vs. Dash
* Justification for metric selections




## Timeline

1. EDA and parser draft
2. Cleaning and canonicalization
3. NLP and feature engineering
4. Supervised and unsupervised models
5. Recommender
6. Dashboard
7. Final report and cleanup


