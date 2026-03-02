import numpy as np
import xgboost as xgb
from google.cloud import bigquery
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the BigQuery table and project details
# Replace these with your actual project and dataset details
PROJECT_ID = 'my-project'
DATASET_ID = 'movie_data'
TABLE_ID = 'preprocessed_data'

# Function to load movies from BigQuery
def load_movies_bq():
    try:
        logger.info("Loading movies from BigQuery...")
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT title, genres
        FROM `{PROJECT_ID}.{DATASET_ID}.preprocessed_movies`
        """
        query_job = client.query(query)
        return query_job.to_dataframe()
    except Exception as e:
        logger.error(f"Error loading movies from BigQuery: {e}")
        raise

# Function to load ratings from BigQuery
def load_ratings_bq():
    try:
        logger.info("Loading ratings from BigQuery...")
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT user_id, title, rating
        FROM `{PROJECT_ID}.{DATASET_ID}.ratings_with_titles`
        """
        query_job = client.query(query)
        return query_job.to_dataframe()
    except Exception as e:
        logger.error(f"Error loading ratings from BigQuery: {e}")
        raise

# Function to create a TensorFlow dataset
def create_tf_dataset(ratings_bq, combine_datasets_fn):
    """Build user-stratified train/test TF datasets from a ratings DataFrame.

    Splits by user_id (not by row) so no user's interactions straddle both
    partitions, preventing embedding-level data leakage. Also returns the
    unbatched training dataset so XGBoost features can be built from it.

    Returns:
        trainds:          Batched, shuffled training dataset.
        testds:           Batched test dataset.
        train_combined:   Unbatched training dataset (for XGBoost feature extraction).
        train_ratings_bq: Training partition DataFrame (for XGBoost user splits).
    """
    try:
        tf.random.set_seed(42)
        np.random.seed(42)

        # ── User-stratified split ─────────────────────────────────────────────
        all_user_ids = np.unique(ratings_bq['user_id'].values)
        np.random.shuffle(all_user_ids)
        split_idx = int(len(all_user_ids) * 0.8)

        train_user_ids = set(all_user_ids[:split_idx].tolist())
        test_user_ids  = set(all_user_ids[split_idx:].tolist())

        assert not (train_user_ids & test_user_ids), "User leakage detected between train and test sets!"

        train_ratings_bq = ratings_bq[ratings_bq['user_id'].isin(train_user_ids)]
        test_ratings_bq  = ratings_bq[ratings_bq['user_id'].isin(test_user_ids)]

        logger.info(f"Train users: {len(train_user_ids)} ({len(train_ratings_bq)} interactions)")
        logger.info(f"Test  users: {len(test_user_ids)} ({len(test_ratings_bq)} interactions)")

        def build_ds(ratings_df):
            d = {k: list(v) for k, v in ratings_df[['title', 'user_id', 'rating']].to_dict(orient='list').items()}
            ds = tf.data.Dataset.from_tensor_slices(d)
            ds = ds.map(lambda x: {"title": x["title"], "user_id": x["user_id"], "rating": x["rating"]})
            ds = ds.map(combine_datasets_fn)
            ds = ds.map(
                lambda x: {"title": x["title"], "user_id": x["user_id"], "genres": x["genres"], "rating": x["rating"]},
                num_parallel_calls=tf.data.AUTOTUNE
            )
            return ds

        train_combined = build_ds(train_ratings_bq)
        test_combined  = build_ds(test_ratings_bq)

        # NOTE: .cache() is intentionally omitted. On memory-constrained machines
        # caching accumulates RAM across Hyperband trials and can kill the process.
        logger.info("Batching training (128) and test (64) datasets...")
        trainds = train_combined.shuffle(100_000, seed=42).batch(128).prefetch(tf.data.AUTOTUNE)
        testds  = test_combined.batch(64).prefetch(tf.data.AUTOTUNE)

        return trainds, testds, train_combined, train_ratings_bq

    except Exception as e:
        logger.error(f"Error creating TensorFlow dataset: {e}")
        raise

def create_xgb_data(xgb_df):
    """Build XGBoost DMatrices using a user-stratified cold-start split.

    Splits by user_id so that validation users are entirely unseen during
    XGBoost training, mirroring the neural-net split discipline.

    Returns:
        dtrain: XGBoost DMatrix for training.
        dval:   XGBoost DMatrix for validation.
        val_df: Validation DataFrame (for per-user NDCG in tune_xgb_model).
    """
    try:
        np.random.seed(42)

        # ── User-stratified XGBoost split ─────────────────────────────────────
        xgb_all_user_ids = xgb_df['user_id'].unique()
        np.random.shuffle(xgb_all_user_ids)
        split_idx      = int(len(xgb_all_user_ids) * 0.8)
        xgb_train_users = set(xgb_all_user_ids[:split_idx].tolist())
        xgb_val_users   = set(xgb_all_user_ids[split_idx:].tolist())

        assert not (xgb_train_users & xgb_val_users), "XGB user leakage detected!"

        train_df = xgb_df[xgb_df['user_id'].isin(xgb_train_users)].copy()
        val_df   = xgb_df[xgb_df['user_id'].isin(xgb_val_users)].copy()

        logger.info(f"XGB train users: {len(xgb_train_users)} ({len(train_df)} interactions)")
        logger.info(f"XGB val   users: {len(xgb_val_users)} ({len(val_df)} interactions)")

        X_train = np.vstack(train_df['features'].values)
        y_train = train_df['rating'].values
        X_val   = np.vstack(val_df['features'].values)

        # Only hstack additional features if they actually exist
        remaining_cols = [c for c in train_df.columns if c not in ['user_id', 'title', 'rating', 'genres', 'features']]
        if remaining_cols:
            X_train = np.hstack([X_train, train_df[remaining_cols].values])
            X_val   = np.hstack([X_val,   val_df[remaining_cols].values])

        logger.info(f"Shape of X_train: {X_train.shape}")
        logger.info(f"Shape of X_val: {X_val.shape}")

        group_train = train_df.groupby('user_id').size().to_list()
        group_val   = val_df.groupby('user_id').size().to_list()

        assert sum(group_train) == X_train.shape[0], "Mismatch between group sizes and rows in X_train"
        assert sum(group_val)   == X_val.shape[0],   "Mismatch between group sizes and rows in X_val"

        logger.info("Creating DMatrix for XGBoost...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(group_train)
        dval = xgb.DMatrix(X_val, label=val_df['rating'].values)
        dval.set_group(group_val)

        return dtrain, dval, val_df

    except Exception as e:
        logger.error(f"Error creating XGBoost data: {e}")
        raise