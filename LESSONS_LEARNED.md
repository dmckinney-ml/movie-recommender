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

- VM: `g2-standard-8` + NVIDIA L4, `us-central1-a` for low latency to BigQuery `us-central1` region.
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

| #   | Issue                                    | Impact                     | Fix                                                         |
| --- | ---------------------------------------- | -------------------------- | ----------------------------------------------------------- |
| 1   | `.cache()` during HPO                    | Kernel OOM                 | Remove `.cache()`, keep `.prefetch()`                       |
| 2   | `legacy.Adam` on Linux                   | Import error               | Platform-aware optimizer dispatch                           |
| 3   | Row-level train/test split               | Embedding leakage          | User-stratified cold-start split                            |
| 4   | XGBoost trained on full dataset          | Cross-stage leakage        | Use training partition only                                 |
| 5   | Global NDCG                              | Inflated ~0.99 NDCG        | Per-user NDCG average                                       |
| 6   | `.predict()` on large arrays             | OOM                        | Batched direct model calls (chunk=512)                      |
| 7   | Signature drift between files            | Silent runtime crash       | Standardise return values; update all call sites            |
| 8   | Wrong checkpoint monitor                 | Saves wrong model          | Add `val_` prefix                                           |
| 9   | Keras 2/3 mixing                         | Cryptic import errors      | `TF_USE_LEGACY_KERAS=1` before TF import                    |
| 10  | Unconditional feature hstack             | Crash on empty columns     | Guard with `if remaining_cols`                              |
| 11  | Local tuner storage                      | Lost trials on VM reclaim  | GCS-backed tuner directory                                  |
| 12  | Wrong HPO objective                      | Tuner optimises wrong task | Switch to `val_root_mean_squared_error`                     |
| 13  | OOV titles at inference                  | KeyError crash             | Zero-vector fallback                                        |
| 14  | Embedding collapse / hub effect          | Same top-10 for all users  | L2-norm embeddings + balanced task weights                  |
| 15  | TFRS 0.7.3 + mixed_float16               | Runtime dtype crash        | Remove mixed precision; TFRS is float32-only                |
| 16  | GCP tf-keras 2.20 conflict               | RecursionError on import   | Pin `tf-keras==2.15.1`; restart kernel after pip            |
| 17  | `tf.py_function` GPU bottleneck          | CPU-bound input pipeline   | Pre-join genres at DataFrame level; `from_tensor_slices`    |
| 18  | XGBoost `device` param < v2.0            | GPU silently unused        | Upgrade to XGBoost ≥2.0 or use `tree_method='gpu_hist'`     |
| 19  | EarlyStopping `metric_name` mismatch     | Callback never triggers    | `metric_name` must exactly match `eval_metric`              |
| 20  | `study.best_params` missing fixed params | Wrong training objective   | Always merge `objective`/`eval_metric` before `xgb.train()` |

---

## 14. Embedding Collapse — Same Recommendations for Every User

### Problem

After deployment, the top-10 recommendations were nearly identical regardless of which movies the user liked. Changing input movies drastically had almost no effect on the output.

### Root Causes (in order of impact)

**1. No L2 normalisation before FAISS**  
`movie_model` outputs raw (unnormalized) vectors. `IndexFlatIP` computes raw dot products, not cosine similarity. Movies that appear many more times in training receive more gradient updates and develop **larger embedding magnitudes**. A high-magnitude embedding always wins the inner product comparison regardless of query direction — these are the "hub" movies that surface in every user's top-k.

**2. Retrieval loss severely underweighted (`rating_weight=2.0, retrieval_weight=0.5`)**  
The contrastive retrieval loss, which is responsible for spreading embeddings apart in embedding space, received 4× less gradient weight than the rating MSE loss. The rating loss is indifferent to geometric diversity, so the model optimised almost entirely for rating prediction, allowing embeddings to collapse into a narrow cluster.

**3. Unnormalised embeddings passed to retrieval task in `compute_loss`**  
The retrieval task received raw embeddings with unbounded magnitudes. With `temperature=0.1`, the logit distribution is sharpened — but when magnitudes are unbounded the distribution is controlled by magnitude as much as direction, amplifying the popularity-bias collapse during training.

### Fixes

**Immediate (no retraining) — `export_hf_artifacts.ipynb`**  
L2-normalise all embeddings before building the FAISS index. This converts `IndexFlatIP` to true cosine similarity and removes the magnitude/popularity bias from the already-trained and deployed model.

```python
faiss.normalize_L2(movie_embeddings)   # in-place; also normalises the saved .npy
faiss.normalize_L2(cold_embeddings)
```

> **Important**: the inference repo must also L2-normalise query vectors before calling `index.search()`, otherwise the inner products are not cosine similarities.

**Training (requires retraining) — `training.ipynb` and `models.py`**

```python
# compute_loss: normalise before retrieval task
user_emb_norm  = tf.nn.l2_normalize(user_embeddings, axis=-1)
movie_emb_norm = tf.nn.l2_normalize(movie_embeddings, axis=-1)
retrieval_loss = self.retrieval_task(user_emb_norm, movie_emb_norm)

# Balanced task weights — retrieval loss now has equal gradient weight as rating
rating_weight=1.0, retrieval_weight=1.0

# Temperature calibrated for unit-norm cosine similarity space ([-1, 1])
retrieval_task = tfrs.tasks.Retrieval(metrics=metrics, temperature=0.05)

# Larger batch → more in-batch negatives per retrieval step (255 vs 127)
train_batch_size = 256
```

### Lesson

> Always L2-normalise embeddings before `IndexFlatIP`. Raw dot products conflate direction and magnitude — popular items with large-magnitude embeddings become universal hubs. Also keep retrieval and rating task weights balanced; a 4:1 imbalance in favour of rating MSE starves the contrastive loss of the gradient signal it needs to keep the embedding space diverse.

---

## 15. TFRS 0.7.3 Incompatible with Mixed Precision

### Problem

Enabling `tf.keras.mixed_precision.set_global_policy('mixed_float16')` on GCP (NVIDIA L4) caused `TypeError` during Hyperband tuning. Two separate crash sites:

1. **`FactorizedTopK` / `Streaming` accumulator**: Internal state tensors are allocated as `float32` by TFRS; when the global policy is `float16` those tensors receive gradients cast to `float16`, producing a dtype mismatch.
2. **In-batch matmul inside `Retrieval.call`**: The matmul `query @ candidates.T` fails when one operand is cast to `float16` and the other remains `float32` (hardcoded in TFRS).

### Fix

Remove mixed precision entirely. GPU training runs in `float32`:

```python
# Remove or never add:
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### Lesson

> TFRS 0.7.3 hardcodes `float32` in its metric accumulators and retrieval matmuls. Mixed precision is incompatible — do not enable `mixed_float16` with this version of TFRS. The speedup on `float32`-heavy recommender workloads is marginal anyway; the GPU throughput benefit of mixed precision is largest for convolutions and transformers, not embedding lookups.

---

## 16. GCP tf-keras Version Conflict → RecursionError on Import

### Problem

GCP Deep Learning VMs (as of early 2026) ship with `tf-keras==2.20.1` pre-installed in `/opt/conda`. Installing `tensorflow==2.15.1` on top does not downgrade `tf-keras`. When `TF_USE_LEGACY_KERAS=1` is set, TF 2.15.1 tries to import `tf_keras` (the 2.x legacy shim), but the installed 2.20.1 package has a changed internal structure that causes infinite import recursion:

```
RecursionError: maximum recursion depth exceeded while calling a Python object
```

### Fix

Explicitly pin `tf-keras` before any other install:

```
# requirements-gcp.txt
tf-keras==2.15.1
```

After `pip install -r requirements-gcp.txt`, **restart the kernel** before importing TensorFlow. The `%pip install` magic installs in the current process but the old module remains cached in `sys.modules` until restart.

### Lesson

> GCP Deep Learning VMs pre-install recent versions of framework accessories that can conflict with pinned framework versions. Always pin every transitive Keras dependency (`tf-keras`) in `requirements-gcp.txt`, and restart the kernel after installation rather than relying on importlib reload.

---

## 17. `tf.py_function` Starves the GPU Input Pipeline

### Problem

The genre lookup in `combine_datasets` was implemented as a `tf.py_function` that called back into Python per element to map integer genre IDs to one-hot vectors. Because `tf.py_function` holds the Python GIL, the tf.data prefetch thread was effectively serialised: only one element could be processed at a time regardless of `AUTOTUNE` or `num_parallel_calls`. On a NVIDIA L4 this produced ~40% GPU utilisation for the first experiments.

### Root Cause

`tf.py_function` opts out of the XLA/tf.data graph and acquires the GIL on every call, creating a CPU bottleneck that the data pipeline cannot pipeline around.

### Fix

Pre-join genres at the DataFrame level before constructing the TF dataset, then use pure `from_tensor_slices` with no Python callbacks:

```python
# Join genre one-hot into the ratings DataFrame before building TF datasets
ratings_bq = ratings_bq.merge(movies_df[['movieId', *genre_cols]], on='movieId', how='left')

# Build dataset without any .map() or tf.py_function
dataset = tf.data.Dataset.from_tensor_slices({
    'user_id':   user_ids,
    'title':     titles,
    'rating':    ratings,
    'genres':    genre_matrix,   # pre-computed float32 array, shape (N, 19)
})
```

### Lesson

> Avoid `tf.py_function` in tf.data pipelines whenever possible. It serialises the entire pipeline through the GIL. Pre-compute any Python-dependent features (genre lookups, string mappings, normalisation) into numpy arrays and feed them through `from_tensor_slices` instead. Reserve `.map()` for stateless, purely TF-graph operations.

---

## 18. XGBoost `device` Parameter Requires XGBoost ≥ 2.0

### Problem

Adding `device='cuda'` to XGBoost params triggered a warning in XGBoost 1.7.6:

```
Parameters: { "device" } are not used.
```

GPU training was silently falling back to CPU, and the XGBoost step took 8× longer than expected.

### Root Cause

The unified `device` parameter was introduced in XGBoost 2.0. In 1.7.x, GPU training requires the legacy parameter `tree_method='gpu_hist'`.

### Fix (option 1 — upgrade):

```
xgboost==2.1.4   # in requirements-gcp.txt
```

### Fix (option 2 — version-agnostic):

```python
import platform
if platform.system() != 'Darwin':
    params['tree_method'] = 'hist'    # works in 1.7.x and 2.x
    params['device'] = 'cuda'         # ignored by 1.7.x, used by 2.x
```

This dual-write is safe: `tree_method='hist'` in XGBoost 2.x simply means "use the unified hist algorithm" (which runs on GPU when `device='cuda'`).

### Lesson

> Check the XGBoost changelog when using GPU parameters — the API changed substantially between 1.7 and 2.0. If you must stay on 1.7.x, use `tree_method='gpu_hist'`; if you can upgrade, use `device='cuda'` with `tree_method='hist'`. The dual-write approach is safe across both versions.

---

## 19. XGBoost EarlyStopping `metric_name` Must Match `eval_metric`

### Problem

`train_xgb_model` in `models.py` (pipeline component) had:

```python
params = {'eval_metric': 'ndcg', ...}
callbacks = [xgb.callback.EarlyStopping(rounds=20, metric_name='rmse', ...)]
```

`metric_name='rmse'` does not appear in the eval log (`eval-ndcg`) so the callback watched a metric that never existed. XGBoost ran all 500 boosting rounds on every training run, never stopping early.

### Root Cause

`metric_name` must be the **exact string** that appears as the suffix of the eval log key. With `eval_metric='ndcg'`, the log key is `eval-ndcg`, so `metric_name` must be `'ndcg'`.

### Fix

```python
xgb.callback.EarlyStopping(
    rounds=20,
    metric_name='ndcg',   # must match eval_metric exactly
    data_name='eval',
    min_delta=1e-4,
)
```

### Lesson

> After changing `eval_metric`, always update every `EarlyStopping(metric_name=...)` call site. A mismatched `metric_name` silently disables early stopping — training continues to `num_boost_round` every run.

---

## 20. `study.best_params` Missing Fixed Params → Wrong Training Objective

### Problem

`tune_xgb_model` returns `study.best_params` — only the hyperparameters that Optuna searched over (`eta`, `max_depth`, etc.). It does **not** include fixed params like `objective` or `eval_metric`. When `train_xgb_model` passed `best_params` directly to `xgb.train`, XGBoost defaulted to `objective='reg:squarederror'` — a regression objective — rather than `rank:pairwise`. The model was trained with the wrong objective and the bug was invisible in the logs because XGBoost does not warn about missing `objective`.

### Root Cause

Optuna's `study.best_params` is a dict of only the `trial.suggest_*` parameters from the objective function. Fixed training parameters that are not part of the search space must be merged in separately.

### Fix

```python
# In train_xgb_model (models.py):
final_params = {
    'objective':   'rank:pairwise',  # always fixed
    'eval_metric': 'ndcg',           # always fixed
    **best_params,                   # Optuna-tuned values
}
xgb.train(final_params, ...)
```

The same merge should be applied in `tune_xgb_model`'s inner `objective` function to ensure tuning and final training use identical objectives.

### Lesson

> `study.best_params` returns only what Optuna searched — never assume it contains the full parameter dict needed for training. Always explicitly merge fixed params (objective, eval_metric, seed, etc.) before calling `xgb.train()`. A missing `objective` is silent but catastrophic: the model trains a completely different task than intended.
