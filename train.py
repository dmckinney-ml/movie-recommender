#!/usr/bin/env python3
"""
End-to-end training script for the movie recommender model.

Equivalent to training.ipynb but runnable headlessly on a VM:
    python train.py

Pipeline:
    1.  Load movies + ratings from BigQuery
    2.  Pre-join genres; user-stratified 80/20 train/test split
    3.  Build TF datasets
    4.  Load best hyperparameters from best_hps.npy and build the two-tower model
    5.  Train (EarlyStopping + TensorBoard + ModelCheckpoint)
    6.  Extract user/movie/genre embeddings
    7.  Prepare XGBoost training data; run Optuna HPO (25 trials)
    8.  Train final XGBoost ranker; save best_params_xgb.npy
    9.  Evaluate: per-user NDCG, Precision/Recall/Accuracy@10, NN retrieval metrics
    10. Save: Keras weights (SavedModel), XGBoost JSON, FAISS index
"""

import os
import glob
import site

# Must be set before TensorFlow is imported
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Dynamically discover and append all NVIDIA wheel libraries to LD_LIBRARY_PATH
nvidia_dirs = []
for _base in site.getsitepackages():
    _tf_dir = os.path.join(_base, "tensorflow")
    if os.path.isdir(_tf_dir):
        nvidia_dirs.append(_tf_dir)
    nvidia_dirs.extend(sorted(glob.glob(os.path.join(_base, "nvidia", "*", "lib"))))

_valid_dirs = [d for d in nvidia_dirs if os.path.isdir(d)]
_current_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(_valid_dirs) + (f":{_current_ld}" if _current_ld else "")

import datetime
import logging
import platform
import random
from typing import Dict, Text

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

import tensorflow as tf
import tensorflow_recommenders as tfrs

from keras_tuner import HyperModel, HyperParameters, Objective
from keras_tuner.tuners import Hyperband

import xgboost as xgb
import optuna

import gc

import faiss

from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "YOUR-PROJECT-ID"
DATASET_ID = "movie_data"

UNIQUE_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Drama", "Documentary", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_movies_bq() -> pd.DataFrame:
    try:
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT title, genres
        FROM `{PROJECT_ID}.{DATASET_ID}.preprocessed_movies`
        """
        return client.query(query).to_dataframe()
    except Exception as e:
        logger.error(f"Error loading movies from BigQuery: {e}")
        raise


def load_ratings_bq() -> pd.DataFrame:
    try:
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT user_id, title, rating
        FROM `{PROJECT_ID}.{DATASET_ID}.ratings_with_titles`
        """
        return client.query(query).to_dataframe()
    except Exception as e:
        logger.error(f"Error loading ratings from BigQuery: {e}")
        raise


# ── Model definition ──────────────────────────────────────────────────────────

class RecommendationModel(tfrs.Model):
    def __init__(
        self,
        user_model,
        movie_model,
        genre_model,
        rating_model,
        rating_task,
        retrieval_task,
        rating_weight: float = 1.0,
        retrieval_weight: float = 1.0,
    ):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.genre_model = genre_model
        self.rating_model = rating_model
        self.rating_task = rating_task
        self.retrieval_task = retrieval_task
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["title"])
        genre_embeddings = self.genre_model(features["genres"])
        rating_predictions = self.rating_model(
            [features["user_id"], features["title"], features["genres"]]
        )
        return user_embeddings, movie_embeddings, genre_embeddings, rating_predictions

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        ratings = features.pop("rating")
        user_embeddings, movie_embeddings, genre_embeddings, rating_predictions = self(features)
        rating_loss = self.rating_task(labels=ratings, predictions=rating_predictions)
        # L2-normalise before the retrieval task so the contrastive loss operates on
        # cosine similarity rather than raw dot products.
        user_emb_norm = tf.nn.l2_normalize(user_embeddings, axis=-1)
        movie_emb_norm = tf.nn.l2_normalize(movie_embeddings, axis=-1)
        retrieval_loss = self.retrieval_task(
            user_emb_norm, movie_emb_norm, compute_metrics=not training
        )
        return self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss


class RecommendationHyperModel(HyperModel):
    def __init__(
        self,
        unique_user_ids,
        unique_titles,
        num_genres: int,
        rating_weight: float = 1.0,
        retrieval_weight: float = 1.0,
        candidate_titles=None,
    ):
        self.unique_user_ids = unique_user_ids
        self.unique_titles = unique_titles
        self.num_genres = num_genres
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
        # Narrowing the candidate pool keeps FactorizedTopK eval tractable on
        # large catalogues — full ~87k vocab would make each epoch ~16x slower.
        self.candidate_titles = candidate_titles if candidate_titles is not None else unique_titles

    def build(self, hp):
        embedding_dimension = hp.Int("embedding_dimension", min_value=32, max_value=256, step=32)
        l2_reg = hp.Float("l2_reg", min_value=1e-5, max_value=1e-2, sampling="log")

        user_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="user_id")
        movie_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="title")
        genre_input = tf.keras.layers.Input(
            shape=(self.num_genres,), dtype=tf.float32, name="genres"
        )

        user_lookup = tf.keras.layers.IntegerLookup(
            vocabulary=self.unique_user_ids, mask_token=None
        )
        movie_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.unique_titles, mask_token=None
        )

        user_embedding = tf.keras.layers.Embedding(
            len(self.unique_user_ids) + 1,
            embedding_dimension,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        )(user_lookup(user_input))

        movie_embedding = tf.keras.layers.Embedding(
            len(self.unique_titles) + 1,
            embedding_dimension,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        )(movie_lookup(movie_input))

        genre_embedding = tf.keras.layers.Dense(
            embedding_dimension,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        )(genre_input)

        concatenated_embeddings = tf.concat(
            [user_embedding, movie_embedding, genre_embedding], axis=1
        )

        dense_1 = tf.keras.layers.Dense(
            hp.Int("units_1", min_value=128, max_value=512, step=64),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        )(concatenated_embeddings)
        dense_1 = tf.keras.layers.BatchNormalization()(dense_1)
        dropout_1 = tf.keras.layers.Dropout(
            hp.Float("dropout_1", min_value=0.3, max_value=0.6, step=0.1)
        )(dense_1)

        dense_2 = tf.keras.layers.Dense(
            hp.Int("units_2", min_value=64, max_value=256, step=32),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        )(dropout_1)
        dense_2 = tf.keras.layers.BatchNormalization()(dense_2)
        dropout_2 = tf.keras.layers.Dropout(
            hp.Float("dropout_2", min_value=0.3, max_value=0.6, step=0.1)
        )(dense_2)

        rating_output = tf.keras.layers.Dense(1)(dropout_2)

        user_model = tf.keras.Model(inputs=user_input, outputs=user_embedding)
        movie_model = tf.keras.Model(inputs=movie_input, outputs=movie_embedding)
        genre_model = tf.keras.Model(inputs=genre_input, outputs=genre_embedding)
        rating_model = tf.keras.Model(
            inputs=[user_input, movie_input, genre_input], outputs=rating_output
        )

        # Use only the rated-movie subset for FactorizedTopK candidates so eval
        # steps stay tractable when unique_titles grows to ~87k with the full catalog.
        metrics = tfrs.metrics.FactorizedTopK(
            candidates=(
                tf.data.Dataset.from_tensor_slices(self.candidate_titles)
                .batch(1024)
                .map(movie_model, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
                .prefetch(tf.data.AUTOTUNE)
            )
        )
        rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        # temperature=0.05 is calibrated for unit-normalised embeddings (cosine sim
        # in [-1, 1]) — equivalent to temperature=0.1 on unnormalised vectors.
        retrieval_task = tfrs.tasks.Retrieval(metrics=metrics, temperature=0.05)

        model = RecommendationModel(
            user_model, movie_model, genre_model, rating_model,
            rating_task, retrieval_task,
            self.rating_weight, self.retrieval_weight,
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
            ),
            decay_steps=hp.Int("decay_steps", min_value=500, max_value=1000, step=100),
            decay_rate=hp.Float("decay_rate", min_value=0.8, max_value=0.9, step=0.05),
            staircase=True,
        )

        # Use legacy Adam on Apple Silicon (Metal backend), standard Adam elsewhere.
        if platform.system() == "Darwin":
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer)
        return model


# ── Embedding / FAISS helpers ─────────────────────────────────────────────────

def extract_embeddings(model, unique_user_ids, unique_titles, batch_size: int = 512):
    """Extract float32 embeddings for all users and movies, batched to avoid OOM."""
    titles = np.array(unique_titles)
    movie_embeddings = np.vstack([
        model.movie_model(tf.constant(titles[i: i + batch_size], dtype=tf.string)).numpy()
        for i in range(0, len(titles), batch_size)
    ]).astype(np.float32)

    user_ids = np.array(unique_user_ids)
    user_embeddings = np.vstack([
        model.user_model(tf.constant(user_ids[i: i + batch_size], dtype=tf.int32)).numpy()
        for i in range(0, len(user_ids), batch_size)
    ]).astype(np.float32)

    return user_embeddings, movie_embeddings


def index_movie_embeddings(movie_embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS IndexFlatIP over the given movie embeddings."""
    if movie_embeddings.ndim != 2 or movie_embeddings.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D array, got shape {movie_embeddings.shape}")
    movie_embeddings = np.ascontiguousarray(movie_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(movie_embeddings.shape[1])
    index.add(movie_embeddings)
    return index


def recommend_movies(model, index: faiss.Index, unique_titles, user_id: int, k: int = 10):
    """Query the FAISS index for the top-k movies nearest to the user embedding."""
    k = min(k, index.ntotal)
    if k == 0:
        raise ValueError("FAISS index is empty — no movies to recommend.")
    user_embedding = np.ascontiguousarray(
        model.user_model(tf.constant([user_id], dtype=tf.int32)).numpy(),
        dtype=np.float32,
    )
    _, indices = index.search(user_embedding, k)
    return [
        t.decode("utf-8") if isinstance(t, bytes) else t
        for t in np.array(unique_titles)[indices[0]]
    ]


# ── Inference helpers (used post-training; not called during the training run) ─

def get_user_titles_rated(ratings_bq: pd.DataFrame, user_id: int) -> np.ndarray:
    rated = ratings_bq[ratings_bq["user_id"] == user_id]["title"].values
    if len(rated) == 0:
        logger.warning(f"No ratings found for user {user_id}.")
    return rated


def get_genre(movies_bq: pd.DataFrame, title: str) -> np.ndarray:
    result = movies_bq[movies_bq["title"] == title]["genres"].values
    if len(result) == 0:
        raise ValueError(f"Title '{title}' not found in movies_bq")
    return result[0]


def create_rank_feature_vector(
    user_embedding_dict: dict,
    movie_embedding_dict: dict,
    user_id: int,
    title,
    genres: np.ndarray,
) -> np.ndarray:
    if user_id not in user_embedding_dict:
        raise KeyError(f"User ID {user_id} not found in user_embedding_dict")
    title = title.decode("utf-8") if isinstance(title, bytes) else title
    movie_emb = movie_embedding_dict.get(
        title,
        np.zeros_like(next(iter(movie_embedding_dict.values()))),
    )
    return np.concatenate([user_embedding_dict[user_id], movie_emb, genres])


def rank_movies_with_xgb(
    user_id: int,
    movies_df: pd.DataFrame,
    bst: xgb.Booster,
    user_embedding_dict: dict,
    movie_embedding_dict: dict,
    ratings_bq: pd.DataFrame,
    k: int = 5,
    remove_all_rated: bool = True,
):
    if user_id not in user_embedding_dict:
        raise ValueError(f"User ID {user_id} not found in user_embedding_dict")
    already_rated = get_user_titles_rated(ratings_bq, user_id)
    logger.info(f"User {user_id} has rated {len(already_rated)} movies.")

    candidate_features = []
    for _, row in movies_df.iterrows():
        feature_vector = create_rank_feature_vector(
            user_embedding_dict, movie_embedding_dict,
            user_id, row["title"], row["genres"],
        )
        additional = row.drop(
            labels=[c for c in ["title", "genres"] if c in row.index]
        ).values
        candidate_features.append(np.concatenate([feature_vector, additional]))

    dtest = xgb.DMatrix(np.vstack(candidate_features))
    predicted = bst.predict(dtest)
    predicted_scaled = 0.5 + (predicted - predicted.min()) * 4.5 / (predicted.max() - predicted.min())

    titles = [
        t.decode("utf-8") if isinstance(t, bytes) else t
        for t in movies_df["title"].values
    ]
    ranked = list(zip(predicted_scaled, titles))

    if remove_all_rated:
        ranked = [m for m in ranked if m[1] not in already_rated]
        ranked.sort(reverse=True, key=lambda x: x[0])
        return ranked[:k]
    else:
        random.shuffle(ranked)
        filtered, already_rated_count = [], 0
        for movie in ranked:
            if movie[1] in already_rated:
                already_rated_count += 1
                if already_rated_count % 10 >= 3:
                    continue
            filtered.append(movie)
        filtered.sort(reverse=True, key=lambda x: x[0])
        return filtered[:k]


def rate_movies_with_hypermodel(hypermodel, user_id: int, titles, genres):
    predicted = []
    for title, genre in zip(titles, genres):
        _, _, _, rating = hypermodel({
            "user_id": np.array([user_id]),
            "title": np.array([title]),
            "genres": np.array([genre]),
        })
        predicted.append([title, rating.numpy()[0][0]])
    return predicted


def get_final_predictions(
    user_id: int,
    movies_df: pd.DataFrame,
    bst: xgb.Booster,
    hypermodel,
    user_embedding_dict: dict,
    movie_embedding_dict: dict,
    ratings_bq: pd.DataFrame,
    movies_bq: pd.DataFrame,
    k: int = 5,
    remove_all_rated: bool = True,
):
    candidates = rank_movies_with_xgb(
        user_id, movies_df, bst,
        user_embedding_dict, movie_embedding_dict, ratings_bq,
        k=k * 100, remove_all_rated=remove_all_rated,
    )
    if not candidates:
        raise ValueError(f"No candidate movies returned for user {user_id}.")
    _, movie_titles = zip(*candidates)
    movie_genres = [get_genre(movies_bq, t) for t in movie_titles]
    predictions = rate_movies_with_hypermodel(hypermodel, user_id, movie_titles, movie_genres)
    return sorted(predictions, key=lambda x: x[1], reverse=True)[:k]


# ── XGBoost evaluation helpers ────────────────────────────────────────────────

def _relevant_titles(user_data: pd.DataFrame, threshold: float) -> np.ndarray:
    return user_data[user_data["rating"] >= threshold]["title"].values


def precision_at_k_strict(val_df: pd.DataFrame, k: int, threshold: float = 4.0) -> float:
    scores = []
    for uid in val_df["user_id"].unique():
        ud = val_df[val_df["user_id"] == uid]
        relevant = _relevant_titles(ud, threshold)
        if len(relevant) == 0:
            continue
        top_k = ud.nsmallest(k, "rank")["title"].values
        scores.append(len(set(relevant) & set(top_k)) / min(k, len(ud)))
    return float(np.mean(scores))


def recall_at_k_strict(val_df: pd.DataFrame, k: int, threshold: float = 4.0) -> float:
    scores = []
    for uid in val_df["user_id"].unique():
        ud = val_df[val_df["user_id"] == uid]
        relevant = _relevant_titles(ud, threshold)
        if len(relevant) == 0:
            continue
        top_k = ud.nsmallest(k, "rank")["title"].values
        scores.append(len(set(relevant) & set(top_k)) / len(relevant))
    return float(np.mean(scores))


def accuracy_at_k_strict(val_df: pd.DataFrame, k: int, threshold: float = 4.0) -> float:
    correct, eligible = 0, 0
    for uid in val_df["user_id"].unique():
        ud = val_df[val_df["user_id"] == uid]
        relevant = _relevant_titles(ud, threshold)
        if len(relevant) == 0:
            continue
        eligible += 1
        top_k = ud.nsmallest(k, "rank")["title"].values
        if set(relevant) & set(top_k):
            correct += 1
    return correct / eligible if eligible > 0 else 0.0


# ── Main training pipeline ────────────────────────────────────────────────────

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info("Loading data from BigQuery...")
    movies_bq = load_movies_bq()
    ratings_bq = load_ratings_bq()

    # Pre-join genres at the DataFrame level — eliminates tf.py_function from the
    # pipeline so tf.data can decode tensor slices in parallel C++ worker threads.
    _genres_lookup = movies_bq.set_index("title")["genres"].to_dict()
    ratings_bq["genres"] = ratings_bq["title"].map(_genres_lookup).apply(
        lambda g: list(g) if isinstance(g, (list, np.ndarray)) else [0] * 19
    )
    logger.info(f"ratings_bq: {len(ratings_bq):,} rows, genres pre-joined")

    # ── 2. User-stratified train / test split ─────────────────────────────────
    tf.random.set_seed(42)
    np.random.seed(42)

    all_user_ids = np.unique(ratings_bq["user_id"].values)
    np.random.shuffle(all_user_ids)
    split_idx = int(len(all_user_ids) * 0.8)

    train_user_ids_set = set(all_user_ids[:split_idx].tolist())
    test_user_ids_set = set(all_user_ids[split_idx:].tolist())
    assert not (train_user_ids_set & test_user_ids_set), "User leakage detected!"

    train_ratings = ratings_bq[ratings_bq["user_id"].isin(train_user_ids_set)]
    test_ratings = ratings_bq[ratings_bq["user_id"].isin(test_user_ids_set)]

    logger.info(f"Train users: {len(train_user_ids_set):,} ({len(train_ratings):,} interactions)")
    logger.info(f"Test  users: {len(test_user_ids_set):,} ({len(test_ratings):,} interactions)")

    # ── 3. Build TF datasets ──────────────────────────────────────────────────
    def build_tf_dataset(ratings_df: pd.DataFrame) -> tf.data.Dataset:
        """Build a tf.data.Dataset with genres pre-joined — no Python callbacks."""
        genres_arr = np.vstack(ratings_df["genres"].values).astype(np.int32)
        return tf.data.Dataset.from_tensor_slices({
            "title": list(ratings_df["title"]),
            "user_id": list(ratings_df["user_id"]),
            "rating": ratings_df["rating"].values.astype(np.float32),
            "genres": genres_arr,
        })

    train_combined_dataset = build_tf_dataset(train_ratings)
    test_combined_dataset = build_tf_dataset(test_ratings)

    train_batch_size = 2048
    test_batch_size = 1024

    trainds = (
        train_combined_dataset
        .shuffle(100_000, seed=42)
        .batch(train_batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    testds = (
        test_combined_dataset
        .batch(test_batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── 4. Build vocabularies ─────────────────────────────────────────────────
    ratings_tf = tf.data.Dataset.from_tensor_slices({
        "title": list(ratings_bq["title"]),
        "user_id": list(ratings_bq["user_id"]),
    })
    movies_tf = tf.data.Dataset.from_tensor_slices({
        "title": list(movies_bq["title"]),
        "genres": list(movies_bq["genres"]),
    })

    unique_titles = np.unique(
        np.concatenate(list(movies_tf.batch(100_000).map(lambda x: x["title"])))
    )
    unique_user_ids = np.unique(
        np.concatenate(list(ratings_tf.batch(1_000_000).map(lambda x: x["user_id"])))
    )

    logger.info(f"Unique titles: {len(unique_titles):,} | Unique users: {len(unique_user_ids):,}")

    # ── 5. Build model from saved best hyperparameters ────────────────────────
    hps_dict = np.load("best_hps.npy", allow_pickle=True).item()
    hp = HyperParameters()
    for hp_key, hp_val in hps_dict.items():
        hp.Fixed(hp_key, hp_val)

    tuner_hypermodel = RecommendationHyperModel(
        unique_user_ids,
        unique_titles,
        len(UNIQUE_GENRES),
        rating_weight=1.0,
        retrieval_weight=1.0,
        candidate_titles=unique_titles,
    )
    tuned_model = tuner_hypermodel.build(hp)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    # Set CHECKPOINT_PATH to restore weights and skip training entirely.
    # Leave as None to train from scratch.
    # On the VM run: ls trained_model/tpe/checkpoints/ to find the timestamp.
    CHECKPOINT_PATH = "trained_model/tpe/checkpoints/20260405180521_cp.ckpt"  # e.g. "trained_model/tpe/checkpoints/20260405XXXXXX_cp.ckpt"

    if CHECKPOINT_PATH:
        logger.info(f"Restoring weights from checkpoint: {CHECKPOINT_PATH}")
        tuned_model.load_weights(CHECKPOINT_PATH).expect_partial()
        logger.info("Weights restored successfully — skipping training.")
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_factorized_top_k/top_50_categorical_accuracy",
            mode="max",
            patience=3,
            min_delta=0.001,
            restore_best_weights=True,
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"trained_model/tpe/checkpoints/{timestamp}_cp.ckpt",
            save_best_only=True,
            save_weights_only=True,
            monitor="val_factorized_top_k/top_50_categorical_accuracy",
            mode="max",
        )
        tensorboard = tf.keras.callbacks.TensorBoard(
            f"trained_model/tpe/tensorboard/{timestamp}_cp.ckpt",
            histogram_freq=1,
        )

        logger.info("Starting neural network training...")
        tuned_model.fit(
            trainds,
            epochs=12,
            validation_data=testds,
            callbacks=[checkpoint, tensorboard, early_stopping],
        )

    # ── 7. Extract embeddings ─────────────────────────────────────────────────
    logger.info("Extracting embeddings...")
    unique_genres_one_hot = np.eye(len(UNIQUE_GENRES), dtype=int).tolist()

    user_embs = tuned_model.user_model.predict(unique_user_ids)
    movie_embs = tuned_model.movie_model.predict(unique_titles)
    genre_embs = tuned_model.genre_model.predict(unique_genres_one_hot)

    user_embedding_dict = dict(zip(unique_user_ids, user_embs))
    movie_embedding_dict = {
        (t.decode("utf-8") if isinstance(t, bytes) else t): emb
        for t, emb in zip(unique_titles, movie_embs)
    }
    genre_embedding_dict = dict(enumerate(genre_embs))

    # ── 8. Prepare XGBoost data ───────────────────────────────────────────────
    logger.info("Preparing XGBoost feature matrix...")

    # Use train_ratings directly — avoids re-materialising the TF dataset into
    # a 25M-row Pandas DataFrame, eliminating the largest intermediate allocation.
    xgb_all_users = train_ratings["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(xgb_all_users)
    xgb_split = int(len(xgb_all_users) * 0.8)
    xgb_train_users = set(xgb_all_users[:xgb_split].tolist())
    xgb_val_users = set(xgb_all_users[xgb_split:].tolist())
    assert not (xgb_train_users & xgb_val_users), "XGB user leakage detected!"

    train_df = train_ratings[train_ratings["user_id"].isin(xgb_train_users)].reset_index(drop=True)
    val_df = train_ratings[train_ratings["user_id"].isin(xgb_val_users)].reset_index(drop=True)
    logger.info(f"XGB train users: {len(xgb_train_users):,} ({len(train_df):,} interactions)")
    logger.info(f"XGB val   users: {len(xgb_val_users):,} ({len(val_df):,} interactions)")

    # Vectorised feature matrix — NumPy fancy indexing instead of row-by-row
    # pandas .apply(), eliminating the intermediate column of per-row arrays.
    user_id_to_idx = {int(uid): i for i, uid in enumerate(unique_user_ids)}
    unique_titles_str = [
        t.decode("utf-8") if isinstance(t, bytes) else t for t in unique_titles
    ]
    title_to_idx = {t: i for i, t in enumerate(unique_titles_str)}

    def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
        user_indices = np.fromiter(
            (user_id_to_idx[int(uid)] for uid in df["user_id"]),
            dtype=np.int64, count=len(df),
        )
        movie_indices = np.fromiter(
            (title_to_idx[t.decode("utf-8") if isinstance(t, bytes) else t]
             for t in df["title"]),
            dtype=np.int64, count=len(df),
        )
        genres_arr = np.vstack(df["genres"].values).astype(np.float32)
        return np.hstack([
            user_embs[user_indices].astype(np.float32),
            movie_embs[movie_indices].astype(np.float32),
            genres_arr,
        ])

    logger.info("Building X_train...")
    X_train = _build_feature_matrix(train_df)
    y_train = train_df["rating"].values.astype(np.float32)

    group_train = train_df.groupby("user_id").size().tolist()
    assert sum(group_train) == X_train.shape[0]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_train)
    del X_train
    gc.collect()
    logger.info("dtrain built — X_train released.")

    logger.info("Building X_val...")
    X_val = _build_feature_matrix(val_df)
    y_val = val_df["rating"].values.astype(np.float32)

    group_val = val_df.groupby("user_id").size().tolist()
    assert sum(group_val) == X_val.shape[0]

    dval_xgb = xgb.DMatrix(X_val, label=y_val)
    dval_xgb.set_group(group_val)
    del X_val
    gc.collect()
    logger.info("dval_xgb built — X_val released.")

    # ── 9. Optuna HPO for XGBoost ─────────────────────────────────────────────
    logger.info("Running Optuna HPO for XGBoost (25 trials)...")

    def xgb_objective(trial):
        param = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            "eta": trial.suggest_float("eta", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "lambda": trial.suggest_float("lambda", 0.0, 1.0),
        }
        if platform.system() != "Darwin":
            param["device"] = "cuda"
            param["tree_method"] = "hist"

        bst_trial = xgb.train(
            param, dtrain, num_boost_round=100,
            evals=[(dval_xgb, "eval")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        y_pred = bst_trial.predict(dval_xgb)
        y_pred_scaled = 0.5 + (y_pred - y_pred.min()) * 4.5 / (y_pred.max() - y_pred.min())

        user_ndcg = []
        for uid in val_df["user_id"].unique():
            mask = val_df["user_id"] == uid
            true_r = val_df.loc[mask, "rating"].values
            pred_r = y_pred_scaled[mask]
            if len(true_r) > 1:
                user_ndcg.append(ndcg_score([true_r], [pred_r]))
        return float(np.mean(user_ndcg))

    study = optuna.create_study(direction="maximize")
    study.optimize(xgb_objective, n_trials=25)
    best_params = study.best_params
    logger.info(f"Best XGBoost parameters: {best_params}")

    # ── 10. Final XGBoost training ─────────────────────────────────────────────
    final_xgb_params = {"objective": "rank:pairwise", "eval_metric": "ndcg", **best_params}
    if platform.system() != "Darwin":
        final_xgb_params["device"] = "cuda"
        final_xgb_params["tree_method"] = "hist"

    bst = xgb.train(
        final_xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval_xgb, "eval")],
        early_stopping_rounds=20,
        verbose_eval=10,
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=20, metric_name="ndcg", data_name="eval", min_delta=1e-4
            )
        ],
    )
    logger.info(f"XGBoost: best iteration={bst.best_iteration}, best score={bst.best_score:.5f}")
    np.save("best_params_xgb.npy", best_params)

    # ── 11. Evaluate XGBoost ──────────────────────────────────────────────────
    logger.info("Evaluating XGBoost ranker...")
    y_pred = bst.predict(dval_xgb)
    y_pred_scaled = 0.5 + (y_pred - y_pred.min()) * 4.5 / (y_pred.max() - y_pred.min())

    val_df = val_df.copy()
    val_df["predicted_rating"] = y_pred_scaled
    val_df["rank"] = val_df.groupby("user_id")["predicted_rating"].rank(
        ascending=False, method="first"
    )

    user_ndcg_scores = []
    for uid, user_data in val_df.groupby("user_id"):
        true_r = user_data["rating"].values
        pred_r = user_data["predicted_rating"].values
        if len(true_r) > 1:
            user_ndcg_scores.append(ndcg_score([true_r], [pred_r]))
    ndcg_mean = float(np.mean(user_ndcg_scores))
    logger.info(
        f"NDCG (per-user average over {len(user_ndcg_scores)} users): {ndcg_mean:.4f}"
    )

    for threshold in [3.5, 4.0, 4.5]:
        p = precision_at_k_strict(val_df, 10, threshold)
        r = recall_at_k_strict(val_df, 10, threshold)
        a = accuracy_at_k_strict(val_df, 10, threshold)
        logger.info(
            f"Threshold {threshold}: Precision@10={p:.4f} | Recall@10={r:.4f} | Accuracy@10={a:.4f}"
        )

    # ── 12. Evaluate neural network ───────────────────────────────────────────
    logger.info("Evaluating neural network...")
    nn_metrics = tuned_model.evaluate(testds, return_dict=True)
    logger.info(f"Retrieval top-1   accuracy: {nn_metrics['factorized_top_k/top_1_categorical_accuracy']:.3f}")
    logger.info(f"Retrieval top-5   accuracy: {nn_metrics['factorized_top_k/top_5_categorical_accuracy']:.3f}")
    logger.info(f"Retrieval top-10  accuracy: {nn_metrics['factorized_top_k/top_10_categorical_accuracy']:.3f}")
    logger.info(f"Retrieval top-50  accuracy: {nn_metrics['factorized_top_k/top_50_categorical_accuracy']:.3f}")
    logger.info(f"Retrieval top-100 accuracy: {nn_metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}")
    logger.info(f"Ranking RMSE:               {nn_metrics['root_mean_squared_error']:.3f}")

    # ── 13. Save artifacts ────────────────────────────────────────────────────
    save_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger.info(f"Saving artifacts (timestamp: {save_timestamp})...")

    # Keras weights
    weights_dir = f"trained_model/tpe/weights/{save_timestamp}_weights"
    os.makedirs(weights_dir, exist_ok=True)
    tuned_model.save_weights(weights_dir)

    # Sub-models as SavedModel (full model.save() fails because FactorizedTopK
    # embeds a stateful tf.data pipeline that can't be serialized)
    for name, sub_model in {
        "user_model": tuned_model.user_model,
        "movie_model": tuned_model.movie_model,
        "genre_model": tuned_model.genre_model,
        "rating_model": tuned_model.rating_model,
    }.items():
        sub_path = os.path.join(weights_dir, name)
        sub_model.save(sub_path)
        logger.info(f"Saved {name} → {sub_path}")

    # XGBoost
    xgb_save_dir = "trained_model/xgb"
    os.makedirs(xgb_save_dir, exist_ok=True)
    xgb_path = f"{xgb_save_dir}/{save_timestamp}_xgb_model.json"
    bst.save_model(xgb_path)
    logger.info(f"Saved XGBoost model → {xgb_path}")

    # FAISS index
    logger.info("Building FAISS index...")
    _, movie_embeddings_faiss = extract_embeddings(tuned_model, unique_user_ids, unique_titles)
    faiss_index = index_movie_embeddings(movie_embeddings_faiss)
    faiss_save_dir = "trained_model/faiss"
    os.makedirs(faiss_save_dir, exist_ok=True)
    faiss_path = f"{faiss_save_dir}/{save_timestamp}_faiss.index"
    faiss.write_index(faiss_index, faiss_path)
    logger.info(f"Saved FAISS index → {faiss_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
