import tensorflow as tf
import os, datetime
import tensorflow_recommenders as tfrs
from google.cloud import bigquery
import numpy as np
import pandas as pd
import pprint
from typing import Dict, Text

# Define the BigQuery table and project details
PROJECT_ID = 'oolola'
DATASET_ID = 'movie_data'
TABLE_ID = 'preprocessed_data'
# Load data from BigQuery
def load_ratings_data_from_bigquery():
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
    SELECT userId, title, movieId, rating
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """
    query_job = client.query(query)
    return query_job.to_dataframe()

def load_movies_data_from_bigquery():
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
    SELECT title, movieId, genres
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """
    query_job = client.query(query)
    return query_job.to_dataframe()

def train():

    # Convert the genres column to a list of tensors
    bq_ratings_data = load_ratings_data_from_bigquery()
    bq_movies_data = load_movies_data_from_bigquery()

    # # Convert the DataFrame to a dictionary of lists
    ratings_dict = {key: list(value) for key, value in bq_ratings_data.to_dict(orient='list').items()}
    # Create the TensorFlow dataset
    ratings = tf.data.Dataset.from_tensor_slices(ratings_dict)

    movies_dict = {key: list(value) for key, value in bq_movies_data.to_dict(orient='list').items()}
    movies = tf.data.Dataset.from_tensor_slices(movies_dict)


    movie_dict = {key: list(value) for key, value in bq_movies_data.to_dict(orient='list').items()}
    movies = tf.data.Dataset.from_tensor_slices(movie_dict)

    ratings = ratings.map(lambda x: {
        "title": x["title"],
        "userId": x["userId"],
    })
    movies = movies.map(lambda x: x["title"])

    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80000)
    test = shuffled.skip(80000).take(20000)

    movie_titles = movies.batch(100000)
    user_ids = ratings.batch(1000000).map(lambda x: x["userId"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    embedding_dimension = 32

    user_model = tf.keras.Sequential([
    tf.keras.layers.IntegerLookup(
        vocabulary=unique_user_ids, mask_token=None),
    # We add an additional embedding to account for unknown tokens.
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    movie_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
    tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    for movie in movies.take(5):
        print(movie, movie.shape)

    metrics = tfrs.metrics.FactorizedTopK(
    candidates=movies.batch(128).map(movie_model)
    )
    metrics._candidates

    task = tfrs.tasks.Retrieval(
    metrics=metrics
    )

    # Define the embedding dimension

    class RecommendationModel(tfrs.Model):

    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task
        
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["userId"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)

    # Instantiate and compile the model
    model = RecommendationModel(user_model, movie_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    # Prepare the data for training
    cached_train = train.shuffle(10000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    # Train the model
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
        callbacks=[tensorboard_callback, early_stopping_callback]
    )

if __name__ == '__main__':
    train()