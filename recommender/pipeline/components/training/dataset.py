from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
from google.cloud import bigquery
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the BigQuery table and project details
PROJECT_ID = 'oolola'
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
def create_tf_dataset(combined_dataset: tf.data.Dataset):
    try:
        tf.random.set_seed(42)

        # Shuffle the dataset with a large buffer size
        # Ensure the buffer size is large enough to cover randomness but not too large to exhaust memory
        shuffle_buffer_size = 200000  # Smaller buffer size for faster shuffling
        logger.info(f"Shuffling the dataset with buffer size {shuffle_buffer_size}...")
        shuffled = combined_dataset.shuffle(buffer_size=shuffle_buffer_size, seed=42, reshuffle_each_iteration=True)

        # Calculate relative proportions for splitting
        train_ratio = 0.8

        ds_length = int(tf.data.experimental.cardinality(shuffled).numpy())
        logger.info(f"Length of the dataset: {ds_length}")

        # Define the split function for large datasets
        def split_dataset(dataset, train_ratio):
            # Determine split points
            trainds = dataset.take(int(train_ratio * ds_length))
            testds = dataset.skip(int(train_ratio * ds_length))

            return trainds, testds

        # Perform the split
        logger.info("Splitting the dataset into training and testing sets...")
        trainds, testds = split_dataset(shuffled, train_ratio)

        # Optimize performance with prefetching
        train_batch_size = 128
        test_batch_size = 64

        # Optimize datasets with batching, caching, and prefetching
        logger.info(f"Batching the datasets with train batch size {train_batch_size} and test batch size {test_batch_size}...")
        trainds = trainds.batch(train_batch_size).cache().prefetch(tf.data.AUTOTUNE)
        testds = testds.batch(test_batch_size).cache().prefetch(tf.data.AUTOTUNE)
        return trainds, testds
    except Exception as e:
        logger.error(f"Error creating TensorFlow dataset: {e}")
        raise

def create_xgb_data(xgb_df: DataFrame):
    try:
        # Set the seed for reproducibility
        random_seed = 42

        # Shuffle the DataFrame
        logger.info("Shuffling the DataFrame...")
        shuffled_df = xgb_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        train_df, val_df = train_test_split(shuffled_df, test_size=0.2, random_state=random_seed)

        # Identify overlapping rows
        common_rows = train_df[['user_id', 'title']].merge(val_df[['user_id', 'title']], how='inner')

        # Remove overlapping rows from the validation set
        if not common_rows.empty:
            val_df = val_df[~val_df[['user_id', 'title']].apply(tuple, axis=1).isin(common_rows.apply(tuple, axis=1))]
            logger.info(f"{len(common_rows)} overlapping rows removed from the validation set.")

        assert train_df[['user_id', 'title']].merge(val_df[['user_id', 'title']], how='inner').empty, "Data leakage detected!"

        # Create feature and label arrays
        logger.info("Creating feature and label arrays...")
        X_train = np.vstack(train_df['features'].values)
        y_train = train_df['rating'].values
        X_val = np.vstack(val_df['features'].values)
        y_val = val_df['rating'].values

        # Add content-based features to the feature matrix
        logger.info("Adding content-based features to the feature matrix...")
        additional_train_features = train_df.drop(columns=['user_id', 'title', 'rating', 'genres', 'features']).values
        additional_val_features = val_df.drop(columns=['user_id', 'title', 'rating', 'genres', 'features']).values

        # Concatenate the additional features
        X_train = np.hstack([X_train, additional_train_features])
        X_val = np.hstack([X_val, additional_val_features])

        # Verify the shapes of the feature matrices
        logger.info(f"Shape of X_train: {X_train.shape}")
        logger.info(f"Shape of X_val: {X_val.shape}")

        # Group sizes for ranking
        group_train = train_df.groupby('user_id').size().to_list()
        group_val = val_df.groupby('user_id').size().to_list()

        # Verify that the sum of group sizes matches the number of rows
        assert sum(group_train) == X_train.shape[0], "Mismatch between group sizes and number of rows in X_train"
        assert sum(group_val) == X_val.shape[0], "Mismatch between group sizes and number of rows in X_val"

        # Create DMatrix for XGBoost
        logger.info("Creating DMatrix for XGBoost...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(group_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(group_val)

        return dtrain, dval, y_val
    except Exception as e:
        logger.error(f"Error creating XGBoost data: {e}")
        raise