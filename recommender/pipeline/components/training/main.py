import tensorflow as tf
import os, datetime
import tensorflow_recommenders as tfrs
from google.cloud import bigquery
import numpy as np
import pandas as pd
from typing import Dict, Text
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the BigQuery table and project details
PROJECT_ID = 'oolola'
DATASET_ID = 'movie_data'
TABLE_ID   = 'preprocessed_data'
timestamp  = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
output_dir = 'movie-data-1'

# Function to load movies from BigQuery
def load_movies_bq():
    try:
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT movieId
        FROM `{PROJECT_ID}.{DATASET_ID}.movies`
        LIMIT 100000
        """
        query_job = client.query(query)
        return query_job.to_dataframe()
    except Exception as e:
        logger.error(f"Error loading movies from BigQuery: {e}")
        raise

# Function to load ratings from BigQuery
def load_ratings_bq():
    try:
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT userId, movieId
        FROM `{PROJECT_ID}.{DATASET_ID}.ratings`
        LIMIT 100000
        """
        query_job = client.query(query)
        return query_job.to_dataframe()
    except Exception as e:
        logger.error(f"Error loading ratings from BigQuery: {e}")
        raise

# Function to get movie titles based on movie IDs
def get_titles(movie_ids: list):
    try:
        client = bigquery.Client(project=PROJECT_ID)
        # Convert the list of movie IDs to a comma-separated string
        movie_ids_str = ', '.join(map(str, movie_ids))
        query = f"""
        SELECT title
        FROM `{PROJECT_ID}.{DATASET_ID}.movies`
        WHERE movieId IN ({movie_ids_str})
        """
        query_job = client.query(query)
        return query_job.to_dataframe()['title'].tolist()
    except Exception as e:
        logger.error(f"Error getting titles from BigQuery: {e}")
        raise

# Define the embedding dimension
embedding_dimension = 32

# Define the custom recommendation model
class RecommendationModel(tfrs.Model):
    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # Extract user and movie embeddings
        user_embeddings = self.user_model(features["userId"])
        positive_movie_embeddings = self.movie_model(features["movieId"])

        # Compute the loss and metrics
        return self.task(user_embeddings, positive_movie_embeddings)

    def get_config(self):
        # Serialize the model configuration
        base_config = super().get_config().copy()
        config = {
            "user_model": tf.keras.utils.serialize_keras_object(self.user_model),
            "movie_model": tf.keras.utils.serialize_keras_object(self.movie_model),
            "task": tf.keras.utils.serialize_keras_object(self.task)
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        # Deserialize the model configuration
        user_model_config = config.pop("user_model")
        movie_model_config = config.pop("movie_model")
        task_config = config.pop("task")

        user_model = tf.keras.utils.deserialize_keras_object(user_model_config)
        movie_model = tf.keras.utils.deserialize_keras_object(movie_model_config)
        task = tf.keras.utils.deserialize_keras_object(task_config)

        return cls(user_model, movie_model, task, **config)

# Function to create the model
def create_model(user_model, movie_model, task):
    model = RecommendationModel(user_model, movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    return model

# Function to train the model
def train():
    # Load data from BigQuery
    try:
        logger.info("Loading data from BigQuery...")
        movies_bq = load_movies_bq()
        ratings_bq = load_ratings_bq()

        # Convert the DataFrame to a dictionary of lists
        logger.info("Converting DataFrames to dictionaries...")
        ratings_dict = {key: list(value) for key, value in ratings_bq[['movieId', 'userId']].to_dict(orient='list').items()}
        movies_dict = {key: list(value) for key, value in movies_bq[['movieId']].to_dict(orient='list').items()}

        # Create TensorFlow datasets
        logger.info("Creating TensorFlow datasets...")
        ratings = tf.data.Dataset.from_tensor_slices(ratings_dict)
        movies = tf.data.Dataset.from_tensor_slices(movies_dict)

        # Map the datasets to extract relevant features
        logger.info("Mapping datasets to extract features...")
        ratings = ratings.map(lambda x: {
            "movieId": x["movieId"],
            "userId": x["userId"],
        })
        movies = movies.map(lambda x: x["movieId"])

        # Shuffle and split the data into training and testing sets
        logger.info("Shuffling and splitting the data...")
        tf.random.set_seed(42)
        shuffled = ratings.shuffle(10000, seed=42, reshuffle_each_iteration=False)
        train = shuffled.take(8000)
        test = shuffled.skip(8000).take(2000)

        # Batch the data
        logger.info("Batching the data...")
        movieIds = movies.batch(10000)
        userIds = ratings.batch(100000).map(lambda x: x["userId"])

        # Get unique movie and user IDs
        logger.info("Getting unique movie and user IDs...")
        unique_movieIds = np.unique(np.concatenate(list(movieIds)))
        unique_userIds = np.unique(np.concatenate(list(userIds)))
        single_user = unique_userIds[0]

        # Define user and movie models
        logger.info("Defining user and movie models...")
        user_model = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(
                vocabulary=unique_userIds, mask_token=None),
            tf.keras.layers.Embedding(len(unique_userIds) + 1, embedding_dimension)
        ])

        movie_model = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(
                vocabulary=unique_movieIds, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movieIds) + 1, embedding_dimension)
        ])

        # Define the retrieval task and metrics
        logger.info("Defining retrieval task and metrics...")
        metrics = tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(movie_model)
        )
        task = tfrs.tasks.Retrieval(
            metrics=metrics
        )

        # Instantiate and compile the model
        logger.info("Creating and compiling the model...")
        model = create_model(user_model, movie_model, task)

        # Prepare the data for training
        cached_train = train.shuffle(10000).batch(8192).cache()
        cached_test = test.batch(4096).cache()

        # Define callbacks for training
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        savedmodel_dir = os.path.join(output_dir, 'savedmodel')
        model_export_path = os.path.join(savedmodel_dir, timestamp)
        checkpoint_path = os.path.join(output_dir, 'checkpoints')
        tensorboard_path = os.path.join(output_dir, 'tensorboard')

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, timestamp),
            save_best_only=True,
            save_weights_only=True,
            monitor='factorized_top_k/top_1_categorical_accuracy',
            mode='min'
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(tensorboard_path)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='factorized_top_k/top_1_categorical_accuracy',
            patience=10,
            restore_best_weights=True
        )

        # Train the model
        model.fit(
            cached_train,
            epochs=3,
            validation_data=cached_test,
            verbose=1,
            callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback]
        )

        # Create a model that takes in raw query features and recommends movies
        logger.info("Creating a model that takes in raw query features and recommends movies...")
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
            tf.data.Dataset.from_tensor_slices(unique_movieIds).batch(128).map(model.movie_model)
        )

        # Get recommendations for a single user
        _, movieIds = index(tf.constant([single_user]))
        movieIds = movieIds[0, :3].numpy().tolist()
        print(f"Recommendations for user {single_user}: {movieIds}")
        titles = get_titles(movieIds)
        print(f"Recommendations for user {single_user}: {titles}")

        # Save the model
        tf.saved_model.save(index, model_export_path)
        logger.info(f"Model saved to {model_export_path}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    train()