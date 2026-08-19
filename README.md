# Fantasy Premier League (FPL) Analytics & Prediction

This repository contains an end-to-end Fantasy Premier League (FPL) analytics project, spanning data cleaning, feature engineering, a deep learning points-prediction model, model interpretability, and a knowledge-graph-based trivia question-answering system.

## Table of Contents

- [Overview](#overview)
- [Part 1: Player Points Prediction](#part-1-player-points-prediction)
  - [Dataset](#dataset)
  - [Data Cleaning](#data-cleaning)
  - [Feature Engineering](#feature-engineering)
  - [Target Variables](#target-variables)
  - [Model Architecture](#model-architecture)
  - [Training Configuration](#training-configuration)
  - [Model Performance](#model-performance)
  - [Model Iterations](#model-iterations)
  - [Interpretability (SHAP & LIME)](#interpretability-shap--lime)
  - [Inference Pipeline](#inference-pipeline)
- [Part 2: FPL Trivia Question Answering (Graph-RAG)](#part-2-fpl-trivia-question-answering-graph-rag)
  - [System Architecture](#system-architecture)
  - [Intent Classification & Entity Extraction](#intent-classification--entity-extraction)
  - [Graph Retrieval Layer](#graph-retrieval-layer)
  - [Failed Approaches](#failed-approaches)
  - [LLM Generation Layer](#llm-generation-layer)
  - [UI](#ui)
- [Project Structure](#project-structure)

## Overview

FPL is a fantasy football game where managers build squads of real Premier League players and earn points based on real-world match performance. This project applies data science and machine learning to two related tasks:

1. **Predicting a player's upcoming Gameweek points** using a Feed-Forward Neural Network (FFNN) trained on historical performance data.
2. **Answering natural-language FPL trivia questions** using a hybrid Knowledge Graph + LLM (Graph-RAG) system.

## Part 1: Player Points Prediction

### Dataset

- **Rows:** 96,169
- **Columns:** 37
- Feature categories: identifiers, player role, match context, performance metrics, fantasy scoring, and manager behavior.
- Only one column (`team_x`) had significant missing data (20.64%).

### Data Cleaning

- **Removed columns:** `ict_index` (redundant with its components), `kickoff_time`, `round` (duplicate of `GW`), `selected`, `transfers_balance`, `transfers_out` (popularity features, out of scope / leakage risk).
- **Missing `team_x` recovery:** Inferred from the opponent team within each `(season, gameweek, fixture)` group, avoiding any data loss.
- **Duplicate checks:** Verified no exact or partial (player + gameweek) duplicates remained.
- **Position standardization:** Merged `GK` and `GKP` into a single `GK` category.
- **Data type validation:** Confirmed correct numeric, boolean, and string/null types across all columns.

### Feature Engineering

**Form**
```
form = average(total_points over previous 4 gameweeks) / 10
```
- 4-gameweek window balances recency vs. stability.
- Divided by 10 to normalize into a ~0–2 range for neural network training.
- Computed per player/season, using only prior weeks to avoid leakage.

**Eight additional engineered features:**

| Feature | Description |
|---|---|
| `weighted_form` | Weighted moving average of last 4 GWs (weights 0.1, 0.2, 0.3, 0.4) |
| `goals_last_4` | Total goals in the last 4 matches |
| `assists_last_4` | Total assists in the last 4 matches |
| `minutes_avg_last_4` | Average minutes played over the last 4 matches |
| `minutes_stability` | Inverse of the std. deviation of minutes over the last 4 matches |
| `cs_rate_last_4` | Clean sheet rate over the last 4 matches |
| `blank_rate_last_4` | Fraction of the last 4 matches with zero points |
| `bonus_avg_last_4` | Average bonus points over the last 4 matches |

### Target Variables

- `upcoming_total_points` — points scored in the next Gameweek (one-step forward shift per player/season).
- `upcoming_blank` — binary flag, 1 if the player scores 0 points next Gameweek.
- Final rows after target creation: **87,473** (2,782 rows dropped — final GW per player/season with no future label).
- Blank rate: **54.40%**.

### Feature Set

- **Base features:** `creativity`, `influence`, `threat`
- **Engineered features:** `weighted_form`, `goals_last_4`, `assists_last_4`, `minutes_avg_last_4`, `minutes_stability`, `cs_rate_last_4`, `blank_rate_last_4`, `bonus_avg_last_4`
- **Context features:** `position` (label-encoded), `was_home`, `upcoming_blank`

Categorical `position` was label-encoded (GK=0, DEF=1, MID=2, FWD=3). Missing engineered feature values were filled with 0.

**Train/Validation/Test split:** 70 / 15 / 15 (61,231 / 13,121 / 13,121 samples), random seed 42, shuffled before splitting.

**Scaling:** `StandardScaler` fit on the training set only, then applied to validation and test sets.

### Model Architecture

Feed-Forward Neural Network (FFNN), chosen for its ability to model non-linear interactions between form, playing time, and match context.

```
Input → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
      → Dense(64, ReLU)  → BatchNorm → Dropout(0.2)
      → Dense(32, ReLU)  → BatchNorm → Dropout(0.2)
      → Dense(1, linear)   # regression output
```

- Total parameters: 13,057 (12,673 trainable / 384 non-trainable)

### Training Configuration

- **Optimizer:** Adam (learning rate = 0.001)
- **Loss:** Mean Squared Error (MSE)
- **Metric:** Mean Absolute Error (MAE)
- **Batch size:** 64, up to 100 epochs
- **Callbacks:**
  - `EarlyStopping` (patience=20, restores best weights)
  - `ReduceLROnPlateau` (factor=0.5, patience=10, min_lr=0.00001)

### Model Performance

Final test set metrics:

| Metric | Value |
|---|---|
| MAE | 0.948 |
| MSE | 3.493 |
| RMSE | 1.869 |
| R² | 0.421 |

The model explains ~42% of the variance in next-Gameweek points, with average predictions within about 1 point of actual outcomes.

### Model Iterations

| Model | Architecture | MAE | RMSE | R² | Notes |
|---|---|---|---|---|---|
| 1 | 64→32→16→8 | 1.378 | 2.552 | -0.005 | Overfitted |
| 2 | 32→16→8 | 1.35–1.4 | 2.5 | ≈0.00 | Underfitted |
| 3 | 32→16→8 | 1.588 | 2.451 | 0.073 | Balanced, modest improvement |
| **4 (Final)** | **128→64→32 + BatchNorm/Dropout** | **0.948** | **1.869** | **0.421** | Best performance & interpretability |

### Interpretability (SHAP & LIME)

- **SHAP** (global + local): Across all positions (FWD, MID, DEF, GK), `upcoming_blank` is consistently the dominant feature — availability overwhelms all other signals. Secondary contributors include `minutes_avg_last_4`, `goals_last_4`, `threat`, and `weighted_form`.
- **LIME**: Confirms the same pattern per-instance — players flagged as unavailable (`upcoming_blank`) receive strongly suppressed predictions regardless of recent form.

### Inference Pipeline

A single reusable function `predict_full_pipeline()` takes:
1. A current-Gameweek row (ICT stats, position, player name, home/away).
2. A 4-row history of the player's last four Gameweeks (points, goals, assists, minutes, clean sheets, bonus).

It performs cleaning → feature engineering (weighted form, minutes stability, cumulative stats, rate-based metrics) → scaling → prediction, and returns a human-readable report with predicted points and key contributing factors.

## Part 2: FPL Trivia Question Answering (Graph-RAG)

A separate system that answers natural-language FPL statistical questions using a hybrid knowledge-graph + vector-retrieval + LLM pipeline.

### System Architecture

```
User Query → Intent Classification & Entity Extraction
           → Graph/Vector Retrieval (Neo4j)
           → LLM Generation
           → Answer
```

- **Input layer:** Intent classification, entity extraction (Regex + spaCy), 10 supported query types
- **Retrieval layer:** Cypher queries (baseline, exact) + vector search (MiniLM/MPNet, semantic) combined via a hybrid strategy
- **Generation layer:** 3 candidate LLMs, context integration, answer synthesis

### Intent Classification & Entity Extraction

- **Method:** Regex-based rule matching with priority ordering; confidence = 90% for a recognized intent, 30% for `UNKNOWN`.
- **Supported intents:** top goal scorer, top assist provider, highest points player, player fixtures, player vs. player, best bonus performer, most yellow cards, highest ICT index, team clean sheets, gameweek top scorer.
- **Entity extraction:** Season, player name (full/last-name matching against a known-players list), team name, gameweek, and metric — extracted via regex, optionally enhanced with spaCy NER (`PERSON`/`ORG`) merged into player matches.

### Graph Retrieval Layer

Built on a **Neo4j** knowledge graph (nodes: Player, Team, Fixture, Gameweek), using a two-tier hybrid retrieval strategy:

- **Baseline (Cypher):** Structured, deterministic queries for exact aggregations (`SUM`, `AVG`, `ORDER BY`/`LIMIT`) across 10 templated query types (player stats, leaderboards, comparisons, position leaders, fixture history, disciplinary records, clean sheets, ICT rankings, transfer trends).
- **Enhanced (Embeddings):** Semantic similarity search using precomputed feature-based embeddings stored on Player nodes (`embedding_minilm` [384-dim], `embedding_mpnet` [768-dim]).
- **Fusion:** Baseline (top 10) + embedding (top 5) results are merged into a single context string and passed to the LLM.

**Final embedding design (27-dim "Advanced Model"):**
- 6 aggregate metrics, 4 per-game metrics, 6 advanced stats (ICT, influence, creativity, threat, minutes, fixtures), 10 one-hot intent flags, 1 comparison-detection dimension.
- Achieved **92% retrieval accuracy** with 0% dimension-mismatch errors and ~5% LLM hallucination rate (down from 35–60% accuracy and 35–40% hallucination in earlier attempts).

### Failed Approaches

Documented for reference:

1. **Node embeddings (sentence-transformer descriptions on Player nodes):** Failed due to dimension mismatches, loss of numeric context (retrieved player names without stats), and semantic-vs-quantitative mismatch (similarity ≠ exact counts).
2. **Basic 14-dimensional query embeddings:** Too sparse and binary-encoded to distinguish complex or multi-metric queries (e.g., comparisons vs. simple lookups), no temporal/comparison support, and poor retrieval accuracy (0.4–0.6 cosine similarity, ~40% false positives).

### LLM Generation Layer

Three candidate models were evaluated using a `persona / context / task` prompt structure ("You are an FPL Trivia Expert..."):

| Model | Type | Notes |
|---|---|---|
| Microsoft Phi-3-mini (4K context) | Small, efficient, instruction-tuned | Fast, low-resource |
| Google Flan-T5 (Base) | Instruction-finetuned encoder-decoder | Compact, structured tasks |
| Google Gemma-2B | Conversational, instruction-tuned decoder | Most natural tone |

Across both embedding strategies (MiniLM & MPNet), all three models scored similarly on relevance, accuracy, completeness, clarity, and naturalness (mostly 4–5/5), but **Gemma-2B had the fastest response time** (~0.54–0.62s), making it the preferred choice for production deployment.

### UI

A **Streamlit** front end exposes the full pipeline:
- Free-text or dropdown question input
- A dedicated panel showing retrieved KG context (baseline Cypher results + embedding-based semantic results) for transparency
- A separate panel for the final LLM-generated answer
- Support for multiple consecutive questions within a session without losing prior context

## Project Structure

```
.
├── data/                     # Raw and processed FPL datasets
├── notebooks/                # EDA, feature engineering, model training
├── models/                   # Trained FFNN model artifacts
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model.py              # FFNN architecture & training
│   ├── inference.py          # predict_full_pipeline()
│   └── interpretability.py   # SHAP / LIME analysis
├── trivia_rag/
│   ├── intent_classifier.py
│   ├── entity_extraction.py
│   ├── graph_retrieval.py    # Cypher + embedding retrieval
│   ├── llm_generation.py
│   └── app.py                # Streamlit UI
└── README.md
```


## Key Takeaways

- Careful feature engineering (form, weighted form, rolling stats) and leakage-safe target construction were critical to a realistic prediction setup.
- The final FFNN (128→64→32 with BatchNorm/Dropout, EarlyStopping, ReduceLROnPlateau) achieved the best trade-off between bias and variance (R² = 0.421), a 4x improvement over earlier baselines.
- `upcoming_blank` (fixture availability) is the single most important predictive signal across all player positions.
- For the trivia system, a **hybrid Cypher + embedding retrieval** strategy substantially outperformed either exact-match or pure semantic-embedding retrieval alone, and richer, purpose-built feature embeddings outperformed generic sentence-transformer node embeddings.
