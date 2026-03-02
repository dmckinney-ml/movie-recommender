# Lessons Learned — Movie Recommender Project

## Overview

This document captures key engineering decisions, bugs encountered, and solutions applied during the development of the hybrid movie recommendation system (MovieLens 32M, TensorFlow Recommenders + XGBoost + FAISS).

---

## 1. Memory Management on Apple Silicon (M-Series)

### Problem

The Hyperband tuning loop caused kernel OOM crashes on an M4 Mac. Memory accumulated across trials until the Metal backend ran out of unified memory.

### Root Cause

- `.cache()` was applied to TF datasets. Each Hyperband trial held on to the cached dataset in RAM rather than releasing memory between trials.
- `tf.keras.optimizers.Adam` on the Metal backend had subtle incompatibilities that contributed to instability.

### Fix

- Removed `.cache()` from all `tf.data` pipelines. `prefetch(tf.data.AUTOTUNE)` alone is sufficient — it reads ahead without pinning the entire dataset in memory.
- Switched to `tf.keras.optimizers.legacy.Adam` on macOS (platform detected via `platform.system() == 'Darwin'`).

### Lesson

> Never `.cache()` large datasets during HPO. Each tuner trial inherits whatever memory the previous trial left behind. Prefetch, don't cache.

---

## 2. Platform-Aware Optimizer

### Problem

`tf.keras.optimizers.legacy.Adam` does not exist on Linux/GCP. Hardcoding it broke the GCP training job.

### Fix

```python
import platform
if platform.system() == 'Darwin':
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Lesson

> Optimizer compatibility is platform-specific in TF 2.15. Abstract it at the top level; don't bury it inside model-building functions.

---

## 3. Row-Level Train/Test Split Causes Embedding Leakage

### Problem

An initial row-level shuffle + `take` / `skip` split allowed the same user to appear in both training and test partitions. Because the user embedding is learned during training, any user present in both partitions has their embedding "pre-heated", which can inflate retrieval top-K accuracy by up to 10 percentage points.

### Fix

Replaced with a **user-stratified cold-start split**:

```python
all_user_ids_unique = np.unique(ratings_bq['user_id'].values)
np.random.shuffle(all_user_ids_unique)
split_idx = int(len(all_user_ids_unique) * 0.8)
train_user_ids_set = set(all_user_ids_unique[:split_idx].tolist())
test_user_ids_set  = set(all_user_ids_unique[split_idx:].tolist())
assert not (train_user_ids_set & test_user_ids_set), "User leakage detected!"
```

### Lesson

> For recommender systems, always split by user identity, not by row. A row-level split is a data-leakage bug, not just a methodology preference.

---

## 4. Cross-Stage XGBoost Leakage

### Problem

The XGBoost feature DataFrame was built from `combined_dataset.as_numpy_iterator()` — the **full** interaction dataset. This let XGBoost train on interactions that the neural network never saw (the neural net's held-out test users), introducing cross-stage leakage that artificially inflated XGBoost validation scores.

### Fix

Build the XGBoost DataFrame from `train_combined_dataset` only (the neural net's training partition):

```python
xgb_df = pd.DataFrame(train_combined_dataset.as_numpy_iterator())
```

### Lesson

> In a multi-stage pipeline, each stage must respect the upstream data partition boundaries. Stage 2 (XGBoost) cannot train on data that Stage 1 (neural net) held out.

---

## 5. Global vs Per-User NDCG

### Problem

Computing NDCG globally (`ndcg_score([y_val], [y_pred_scaled])`) treats the entire validation set as a single ranked list. Because most items are not relevant for any given user, the algorithm trivially gets ~0.988 NDCG. This gives a false sense of ranker quality.

### Fix

Compute NDCG per user, then average:

```python
user_ndcg_scores = []
for uid, user_data in val_df.groupby('user_id'):
    true_ratings = user_data['rating'].values
    pred_ratings = user_data['predicted_rating'].values
    if len(true_ratings) > 1:
        user_ndcg_scores.append(ndcg_score([true_ratings], [pred_ratings]))
ndcg = np.mean(user_ndcg_scores)
```

### Lesson

> Global ranking metrics for recommendation tasks are meaningless. Always group by user (query) and average the per-query metric.

---

## 6. Embedding Extraction OOM with `.predict()`

### Problem

`model.predict(unique_user_ids)` and `model.predict(unique_titles)` on 200k users / 87k movies caused OOM on both the M4 Mac and the T4 GPU because `.predict()` allocates the full output tensor before returning.

### Fix

Batched inference with explicit chunk size:

```python
batch_size = 512
titles_arr = np.array(unique_titles)
movie_embeddings = np.vstack([
    model.movie_model(tf.constant(titles_arr[i:i+batch_size], dtype=tf.string)).numpy()
    for i in range(0, len(titles_arr), batch_size)
]).astype(np.float32)
```

### Lesson

> Use direct model calls in chunks rather than `.predict()` when the output tensor is too large to fit in memory at once. `.predict()` is convenient but not memory-safe for large catalogues.

---

## 7. Wrong Function Signatures Between Pipeline Components

### Problem

The notebook, `dataset.py`, and `train.py` diverged silently. `train.py` called `create_xgb_data(xgb_df)` expecting `(dtrain, dval, y_val)` but `dataset.py` was being updated to return `(dtrain, dval, val_df)`. This would have caused a silent variable-name bug that only appears at the XGBoost evaluation step.

### Fix

- Standardised return signature to `(dtrain, dval, val_df)` everywhere.
- Updated all call sites in `train.py` in the same PR.

### Lesson

> Keep pipeline component I/O contracts explicit and tested. When you change a function's return value, grep for every call site before committing.

---

## 8. Checkpoint Monitor Missing `val_` Prefix

### Problem

The `ModelCheckpoint` callback was monitoring `'factorized_top_k/top_5_categorical_accuracy'` (training set metric). Keras validation metrics are prefixed with `val_`, so the callback was tracking the wrong metric and potentially saving suboptimal checkpoints.

### Fix

```python
monitor='val_factorized_top_k/top_5_categorical_accuracy'
```

### Lesson

> Always confirm that callback `monitor` strings match the exact key in `history.history`. Print `history.history.keys()` after the first epoch if unsure.

---

## 9. Keras Compatibility — `TF_USE_LEGACY_KERAS`

### Problem

`keras_tuner` and `tensorflow_recommenders` on TF 2.15 default to standalone Keras 3 when it is installed. Keras 3 has breaking API changes that affect both libraries. On GCP (clean environment), imports silently mixed Keras 2 and Keras 3 objects, causing cryptic errors deep in model construction.

### Fix

```python
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # must be set BEFORE importing TensorFlow
```

Pin `tf_keras==2.15.1` in requirements.

### Lesson

> Set `TF_USE_LEGACY_KERAS=1` as the very first statement in any script or notebook cell that uses TFRS or keras-tuner with TF 2.15. Import order matters — TF reads the env variable at import time.

---

## 10. Always-Applied Feature Hstack Crash

### Problem

In `create_xgb_data`, the code always did:

```python
X_train = np.hstack([X_train, train_df[remaining_cols].values])
```

When `remaining_cols` was empty, this was a no-op — but when it contained non-numeric columns it would crash silently or produce a mismatched array. When there were no remaining columns at all, calling `.values` on an empty DataFrame produced a zero-column array that hstack would mangle.

### Fix

Guard with an explicit check:

```python
remaining_cols = [c for c in train_df.columns if c not in ['user_id', 'title', 'rating', 'genres', 'features']]
if remaining_cols:
    X_train = np.hstack([X_train, train_df[remaining_cols].values])
    X_val   = np.hstack([X_val,   val_df[remaining_cols].values])
```

### Lesson

> Never unconditionally hstack optional feature columns. Wrap it in an existence check so the path is safe when no extra features exist.

---

## 11. GCP Environment Setup

### Key Decisions

- VM: `n1-highmem-8` + NVIDIA T4, `us-central1-a` for low latency to BigQuery `us-central1` region.
- Trials written directly to GCS (`gs://movie-data-1/tuning`) so Hyperband can resume after VM interruptions.
- Two separate requirements files: `requirements-local.txt` (adds `tensorflow-metal`, `jupyter`, `ipykernel`) and `requirements-gcp.txt` (no Metal, standard TF).

### Lesson

> Use GCS-backed tuner directories from day one. Local storage is lost when a preemptible VM is reclaimed. GCS trial persistence is free and saves hours of re-running.

---

## 12. Hyperband Objective Selection

### Problem

Early runs used `val_factorized_top_k/top_5_categorical_accuracy` as the Hyperband objective. Because the rating task is upweighted (`rating_weight=2.0`), the model was being evaluated on the retrieval auxiliary signal rather than the primary task, causing the tuner to over-index on retrieval while tolerating high rating RMSE.

### Fix

Changed tuner objective to `val_root_mean_squared_error` (direction: `min`) to align the HPO signal with the primary loss.

### Lesson

> The tuner objective must match the upweighted loss component. If rating prediction is 4× more important than retrieval in the training loss, the HPO objective should also optimise rating error.

---

## 13. Inference Pipeline — OOV Titles

### Problem

`create_rank_feature_vector` raised a `KeyError` when a movie title in the candidate set was not in `movie_embedding_dict`. This happens for movies added after training or titles with minor formatting differences.

### Fix

```python
if title not in movie_embedding_dict:
    movie_embedding = np.zeros_like(next(iter(movie_embedding_dict.values())))
else:
    movie_embedding = movie_embedding_dict[title]
```

### Lesson

> Always provide a safe fallback for OOV items in inference. A zero vector is a reasonable default for an embedding lookup miss — it will score low relative to known items, which is the correct degraded behaviour.

---

## Summary Table

| #   | Issue                           | Impact                     | Fix                                              |
| --- | ------------------------------- | -------------------------- | ------------------------------------------------ |
| 1   | `.cache()` during HPO           | Kernel OOM                 | Remove `.cache()`, keep `.prefetch()`            |
| 2   | `legacy.Adam` on Linux          | Import error               | Platform-aware optimizer dispatch                |
| 3   | Row-level train/test split      | Embedding leakage          | User-stratified cold-start split                 |
| 4   | XGBoost trained on full dataset | Cross-stage leakage        | Use training partition only                      |
| 5   | Global NDCG                     | Inflated ~0.99 NDCG        | Per-user NDCG average                            |
| 6   | `.predict()` on large arrays    | OOM                        | Batched direct model calls (chunk=512)           |
| 7   | Signature drift between files   | Silent runtime crash       | Standardise return values; update all call sites |
| 8   | Wrong checkpoint monitor        | Saves wrong model          | Add `val_` prefix                                |
| 9   | Keras 2/3 mixing                | Cryptic import errors      | `TF_USE_LEGACY_KERAS=1` before TF import         |
| 10  | Unconditional feature hstack    | Crash on empty columns     | Guard with `if remaining_cols`                   |
| 11  | Local tuner storage             | Lost trials on VM reclaim  | GCS-backed tuner directory                       |
| 12  | Wrong HPO objective             | Tuner optimises wrong task | Switch to `val_root_mean_squared_error`          |
| 13  | OOV titles at inference         | KeyError crash             | Zero-vector fallback                             |
