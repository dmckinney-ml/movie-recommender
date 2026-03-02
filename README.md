# 🎬 Movie Recommender

A scalable, end-to-end movie recommendation ML pipeline built on Google Cloud Platform. The system combines collaborative filtering via deep learning embeddings with XGBoost ranking to deliver personalized movie recommendations.

## Table of Contents

- [Introduction](#introduction)
- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Locally (Notebook)](#training-locally-notebook)
  - [Running the Pipeline](#running-the-pipeline)
  - [Docker Components](#docker-components)
- [Model Details](#model-details)
  - [Two-Tower Neural Network](#two-tower-neural-network)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [XGBoost Ranker](#xgboost-ranker)
  - [FAISS Index](#faiss-index)
- [Pipeline Components](#pipeline-components)
  - [Dataflow Preprocessing](#dataflow-preprocessing)
  - [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

The Movie Recommender is a production-grade recommendation system that leverages a two-stage approach:

1. **Retrieval**: A multi-task neural network generates user and movie embeddings, queried efficiently at inference via a FAISS index.
2. **Ranking**: An XGBoost model re-ranks candidate movies using rich feature vectors derived from the embeddings.

Trained on the **[MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)** dataset — a stable benchmark containing **32 million ratings** and **2 million tag applications** applied to **87,585 movies** by **200,948 users**. Ratings data is stored in BigQuery and preprocessed via Apache Beam on Cloud Dataflow before training.

The system is orchestrated as a Kubeflow pipeline on Google Cloud's Vertex AI, with preprocessing handled by Apache Beam on Dataflow and training containerized via Docker.

---

## Architecture Overview

```
BigQuery (Raw Data)
        │
        ▼
┌────────────────────┐
│  Apache Beam /     │
│  Cloud Dataflow    │  ← Preprocessing Component
│  (Preprocessing)   │
└────────────────────┘
        │
        ▼
BigQuery (Preprocessed Data)
        │
        ▼
┌────────────────────┐
│  Training          │
│  Component         │  ← Vertex AI Custom Training Job
│                    │
│  ┌──────────────┐  │
│  │ Two-Tower NN │  │  ← Keras + TensorFlow Recommenders
│  │ (Keras Tuner)│  │
│  └──────┬───────┘  │
│         │ embeddings│
│  ┌──────▼───────┐  │
│  │ XGBoost      │  │  ← Ranking Model (Optuna Tuned)
│  │ Ranker       │  │
│  └──────────────┘  │
└────────────────────┘
        │
        ▼
  GCS (Model Artifacts)
  ├── Keras weights (TF SavedModel format)
  ├── XGBoost model (.json)
  └── FAISS index (.index)
```

---

## Features

- **Multi-task Learning**: Jointly optimizes rating prediction (RMSE) and item retrieval (top-K accuracy) in a single neural network.
- **Hyperparameter Tuning**:
  - Keras model tuned via **Hyperband** (TPE) and **Bayesian Optimization** using `keras-tuner`.
  - XGBoost ranker tuned via **Optuna** (25 trials).
- **Efficient Retrieval**: Movie embeddings indexed with **FAISS** for fast approximate nearest-neighbor search at inference.
- **Scalable Preprocessing**: Data pipeline built with **Apache Beam**, running on **Google Cloud Dataflow**.
- **Containerized Components**: Each pipeline stage packaged as a Docker image, pushed to **Google Artifact Registry**.
- **Vertex AI Orchestration**: End-to-end pipeline defined with **Kubeflow Pipelines (KFP)** and executed on **Vertex AI Pipelines**.
- **Genre-aware Recommendations**: One-hot encoded genre features fed directly into the neural network and XGBoost feature vectors.

---

## Tech Stack

| Layer                    | Technology                                                |
| ------------------------ | --------------------------------------------------------- |
| Deep Learning Framework  | TensorFlow 2.15.1 / tf_keras (legacy Keras)               |
| Recommendation Framework | TensorFlow Recommenders 0.7.3 (TFRS)                      |
| Hyperparameter Tuning    | Keras Tuner (Hyperband), Optuna                           |
| Ranking Model            | XGBoost                                                   |
| Vector Search            | FAISS                                                     |
| Data Processing          | Apache Beam, Pandas, NumPy                                |
| Cloud Platform           | Google Cloud Platform (GCP)                               |
| Dataset                  | MovieLens 32M (32M ratings, 87,585 movies, 200,948 users) |
| Data Warehouse           | BigQuery                                                  |
| Pipeline Orchestration   | Kubeflow Pipelines / Vertex AI                            |
| Containerization         | Docker                                                    |
| Artifact Storage         | GCS, Google Artifact Registry                             |
| Language                 | Python 3.10                                               |

---

## Project Structure

```
movie_recommender/
├── training.ipynb                        # Exploratory training notebook
├── best_hps.npy                          # Best Keras Tuner hyperparameters
├── best_params_xgb.npy                   # Best XGBoost hyperparameters
├── pyproject.toml                        # Project dependencies (Poetry)
├── requirements-local.txt                # Local (macOS) dependencies incl. tensorflow-metal
├── requirements-gcp.txt                  # GCP/Linux dependencies (no tensorflow-metal)
├── trained_model/
│   ├── tpe/                              # Hyperband tuner artifacts
│   │   ├── checkpoints/
│   │   ├── faiss/
│   │   ├── tensorboard/
│   │   └── weights/                      # TF SavedModel format
│   ├── xgb/models/                       # Saved XGBoost models (.json)
│   └── params/                           # Saved best hyperparameters
├── tuning/
│   └── tpe/                              # Local Hyperband trial logs
│                                         # (GCS: gs://movie-data-1/tuning)
└── recommender/
    └── pipeline/
        ├── create_pipeline.py            # KFP pipeline definition & submission
        ├── movie-pipeline.yaml           # Compiled pipeline YAML
        └── components/
            ├── dataflow_preprocessing/   # Preprocessing pipeline component
            │   ├── main.py
            │   ├── Dockerfile
            │   ├── preprocessing.yaml
            │   ├── Makefile
            │   └── scripts/
            │       ├── env.sh
            │       ├── build.sh
            │       ├── push.sh
            │       ├── run.sh
            │       └── login.sh
            └── training/                 # Training pipeline component
                ├── main.py
                ├── train.py
                ├── models.py
                ├── dataset.py
                ├── Dockerfile
                ├── training.yaml
                ├── requirements.txt
                ├── Makefile
                └── scripts/
                    ├── env.sh
                    ├── build.sh
                    ├── push.sh
                    ├── run.sh
                    └── login.sh
```

---

## Installation

### Prerequisites

- Python 3.10–3.12.2
- [Poetry](https://python-poetry.org/) (recommended) or `pip`
- [Docker](https://www.docker.com/)
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) authenticated with a GCP project
- A GCP project with the following APIs enabled:
  - BigQuery
  - Dataflow
  - Vertex AI
  - Artifact Registry
  - Cloud Storage

### Local Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/movie_recommender.git
   cd movie_recommender
   ```

2. **Install dependencies:**

   For local development (macOS / Apple Silicon, includes `tensorflow-metal` and Jupyter):

   ```bash
   pip install -r requirements-local.txt
   ```

   For GCP / Linux (no `tensorflow-metal`, no Jupyter):

   ```bash
   pip install -r requirements-gcp.txt
   ```

   > **Note:** Set `TF_USE_LEGACY_KERAS=1` before running. This project requires `tf_keras==2.15.1` — do **not** install standalone `keras`.

3. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

---

## Usage

### Training Locally (Notebook)

The `training.ipynb` notebook walks through the full training workflow interactively:

1. **Load data** from BigQuery into TensorFlow datasets.
2. **User-stratified split** (80/20 by `user_id`) — test users are entirely unseen during neural net training (cold-start).
3. **Build and tune** the multi-task neural network using Keras Tuner (Hyperband), with trial checkpoints written to GCS.
4. **Extract embeddings** from the trained model for both users and movies (batched, `float32` cast for FAISS).
5. **Create XGBoost feature vectors** by concatenating user embeddings, movie embeddings, and genre one-hot encodings.
6. **Cold-start XGBoost split** (80/20 by `user_id`) — val users unseen during XGBoost training.
7. **Tune and train** the XGBoost ranking model using Optuna (50 trials, per-user NDCG objective).
8. **Index embeddings** with FAISS for fast retrieval.
9. **Evaluate** with per-user NDCG, Precision@K, Recall@K, and Accuracy@K at multiple thresholds.

```bash
jupyter notebook training.ipynb
```

> **GCP training:** Hyperband tuning is resource-intensive. For best results, run on a GCP VM (`n1-highmem-8` + NVIDIA T4):
>
> ```bash
> gcloud compute scp training.ipynb tuning-vm:~ --zone=us-central1-a --project=YOUR_PROJECT_ID
> gcloud compute ssh tuning-vm --zone=us-central1-a --project=YOUR_PROJECT_ID -- -L 8888:localhost:8888
> ```

### Running the Pipeline

Compile and submit the Vertex AI pipeline:

```bash
cd recommender/pipeline
python create_pipeline.py
```

This will:

1. Compile `movie-pipeline.yaml` from the KFP pipeline definition.
2. Submit the job to Vertex AI Pipelines with caching enabled.

### Docker Components

Each component can be built, pushed, and run independently. From a component directory (e.g., `recommender/pipeline/components/training/`):

```bash
# Build the Docker image
make build

# Push to Google Artifact Registry
make push

# Run the container locally (mounts gcloud credentials)
make run

# Open an interactive shell in the container
make login

# Print resolved environment variables
make env
```

---

## Model Details

### Two-Tower Neural Network

The core retrieval model (`RecommendationHyperModel`) is a multi-task model built with TensorFlow Recommenders:

- **User Tower**: Integer lookup → Embedding layer (L2 regularized)
- **Movie Tower**: String lookup → Embedding layer (L2 regularized)
- **Genre Tower**: Dense projection of one-hot genre vector (L2 regularized)
- **Shared Head**: Concatenated embeddings → Dense(units_1, ReLU) → **BatchNorm** → Dropout → Dense(units_2, ReLU) → **BatchNorm** → Dropout → Dense(1) for rating prediction
- **Optimizer**: Adam with Exponential Decay LR schedule. Uses `tf.keras.optimizers.legacy.Adam` on macOS (Metal backend) and `tf.keras.optimizers.Adam` on Linux/GCP (CUDA).
- **Losses**:
  - Rating task: Mean Squared Error (`rating_weight=2.0`)
  - Retrieval task: Factorized Top-K (`retrieval_weight=0.5`, `temperature=0.1`)
- **Tuner objective**: minimize `val_root_mean_squared_error`
- **Trial storage**: `gs://movie-data-1/tuning` (GCS)

**Tunable Hyperparameters:**

| Hyperparameter        | Search Range          |
| --------------------- | --------------------- |
| `embedding_dimension` | 32–256 (step 32)      |
| `l2_reg`              | 1e-5–1e-2 (log scale) |
| `units_1`             | 128–512 (step 64)     |
| `dropout_1`           | 0.3–0.6 (step 0.1)    |
| `units_2`             | 64–256 (step 32)      |
| `dropout_2`           | 0.3–0.6 (step 0.1)    |
| `learning_rate`       | 1e-4–1e-2 (log scale) |
| `decay_steps`         | 500–1000 (step 100)   |
| `decay_rate`          | 0.8–0.9 (step 0.05)   |

### Hyperparameter Tuning

| Strategy        | Library       | Trials/Epochs           | Directory                        |
| --------------- | ------------- | ----------------------- | -------------------------------- |
| Hyperband (TPE) | `keras-tuner` | max 12 epochs, factor=3 | `gs://movie-data-1/tuning` (GCS) |

Tuning is run on a GCP VM (`n1-highmem-8` + NVIDIA T4) to avoid Apple Silicon memory constraints. Trial checkpoints are written directly to GCS.

### XGBoost Ranker

After retrieving candidates via FAISS, the XGBoost model re-ranks them using `rank:pairwise` with NDCG as the evaluation metric.

**Feature vector** per (user, movie) pair:

- User embedding (from Two-Tower model)
- Movie embedding (from Two-Tower model)
- Genre one-hot encoding (19 genres)

**Data split:** User-level cold-start split (80/20 by `user_id`) — val users are entirely unseen during XGBoost training, mirroring the neural net split.

**Training:** `num_boost_round=500`, early stopping with `patience=20` and `min_delta=1e-4`. Best iteration: ~139, Best RMSE: `~0.878` (cold-start).

**Best Hyperparameters (Optuna, 50 trials):**

| Hyperparameter     | Value  |
| ------------------ | ------ |
| `eta`              | 0.0727 |
| `max_depth`        | 9      |
| `min_child_weight` | 8      |
| `subsample`        | 0.7098 |
| `colsample_bytree` | 0.6422 |
| `gamma`            | 0.1992 |
| `lambda`           | 0.9450 |

### FAISS Index

After training, movie embeddings are extracted in batches (`batch_size=512`) and indexed into a FAISS `IndexFlatL2` for exact nearest-neighbor search. Embeddings are cast to `float32` C-contiguous arrays before indexing (required by FAISS). At inference:

1. The user embedding is computed by the user tower (cast to `float32`).
2. FAISS returns the top-K closest movie embeddings.
3. Candidate movies are re-ranked by the XGBoost model.

---

## Pipeline Components

### Dataflow Preprocessing

**Location:** `recommender/pipeline/components/dataflow_preprocessing/`

Reads raw `movies` and `ratings` tables from BigQuery, performs:

- Join of ratings with movie metadata (title, year extraction via regex).
- One-hot encoding of 19 genres.
- Filtering to movies released after 2020.
- Writes preprocessed data back to BigQuery.

Runs on **Google Cloud Dataflow** using the Apache Beam Python SDK.

### Training

**Location:** `recommender/pipeline/components/training/`

Orchestrates the full training workflow:

1. Loads preprocessed data from BigQuery (`dataset.py`).
2. Builds and tunes the Keras hypermodel (`models.py`).
3. Extracts embeddings and constructs XGBoost features.
4. Tunes and trains the XGBoost ranker (`models.py`).
5. Saves all artifacts (weights, model JSON, FAISS index) to GCS.

Entry point: `main.py` → `train.py:train()`

---

## Evaluation

All evaluation uses a **cold-start user split** — test/val users are entirely unseen during training of both stages.

### Neural Network (cold-start test users)

| Metric                     | Value | Notes                                    |
| -------------------------- | ----- | ---------------------------------------- |
| Ranking RMSE               | 0.956 | Expected range for cold-start: 0.90–1.05 |
| Retrieval Top-1 Accuracy   | 0.2%  | ~157× better than random (1/~14k titles) |
| Retrieval Top-5 Accuracy   | 1.1%  |                                          |
| Retrieval Top-10 Accuracy  | 2.0%  |                                          |
| Retrieval Top-50 Accuracy  | 7.5%  | Practical retrieval window               |
| Retrieval Top-100 Accuracy | 12.2% | ~25–40% estimated at top-300 to top-500  |

Retrieval accuracy is a strict exact-match metric (exact held-out movie in top-K from full corpus). Low absolute numbers are expected and acceptable — the retrieval stage only needs to supply diverse candidates to XGBoost.

### XGBoost Re-ranker (cold-start val users)

Evaluated with per-user NDCG and strict precision/recall (no fallback for users with no relevant items):

| Metric                        | Value  |
| ----------------------------- | ------ |
| NDCG Score (per-user average) | 0.9624 |
| Precision@10 (threshold=3.5)  | 0.7875 |
| Recall@10 (threshold=3.5)     | 0.8607 |
| Accuracy@10 (threshold=3.5)   | 0.9980 |
| Precision@10 (threshold=4.0)  | 0.6665 |
| Recall@10 (threshold=4.0)     | 0.8684 |
| Accuracy@10 (threshold=4.0)   | 0.9944 |
| Precision@10 (threshold=4.5)  | 0.4889 |
| Recall@10 (threshold=4.5)     | 0.8754 |
| Accuracy@10 (threshold=4.5)   | 0.9764 |

> **Note:** XGBoost val users are cold to XGBoost but warm to the neural net (their embeddings were learned during neural net training). True end-to-end cold-start would require users with no neural net training history.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
