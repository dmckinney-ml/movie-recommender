import tensorflow as tf
import os, datetime
import tensorflow_recommenders as tfrs
from google.cloud import bigquery
import numpy as np
import pandas as pd
from typing import Dict, Text
import logging
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the BigQuery table and project details
PROJECT_ID = 'oolola'
DATASET_ID = 'movie_data'
GENRES_TABLE_ID = 'genres'
RATINGS_TABLE_ID = 'ratings'
timestamp  = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
output_dir = 'gs://movie-data-1/trained-model'

# Function to load movies from BigQuery
def load_movies_bq():
    try:
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT movie_id, genres
        FROM `{PROJECT_ID}.{DATASET_ID}.genres`
        WHERE movie_id >= 1 AND movie_id <= 10000
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
        SELECT userId, movieId, rating
        FROM `{PROJECT_ID}.{DATASET_ID}.ratings`
        WHERE movieId >= 1 AND movieId <= 10000
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
        FROM `{PROJECT_ID}.{DATASET_ID}.genres`
        WHERE movieId IN ({movie_ids_str})
        """
        query_job = client.query(query)
        return query_job.to_dataframe()['title'].tolist()
    except Exception as e:
        logger.error(f"Error getting titles from BigQuery: {e}")
        raise

# Define the custom recommendation model using the Functional API
class RecommendationModel(tfrs.Model):
    def __init__(self, user_model, movie_model, genre_model, rating_model, rating_task, retrieval_task):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.genre_model = genre_model
        self.rating_model = rating_model
        self.rating_task = rating_task
        self.retrieval_task = retrieval_task

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_id"])
        genre_embeddings = self.genre_model(features["genres"])
        rating_predictions = self.rating_model([features["user_id"], features["movie_id"], features["genres"]])
        return user_embeddings, movie_embeddings, genre_embeddings, rating_predictions

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        ratings = features.pop("rating")
        user_embeddings, movie_embeddings, genre_embeddings, rating_predictions = self(features)
        rating_loss = self.rating_task(labels=ratings, predictions=rating_predictions)
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)
        return rating_loss + retrieval_loss
    
# Function to create the model
def create_model(unique_user_ids, unique_movie_ids, num_genres):
    embedding_dimension = 32
    user_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='user_id')
    movie_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='movie_id')
    genre_input = tf.keras.layers.Input(shape=(num_genres,), dtype=tf.int32, name='genres')

    user_lookup = tf.keras.layers.IntegerLookup(vocabulary=unique_user_ids, mask_token=None)
    movie_lookup = tf.keras.layers.IntegerLookup(vocabulary=unique_movie_ids, mask_token=None)

    user_embedding = tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)(user_lookup(user_input))
    movie_embedding = tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension)(movie_lookup(movie_input))
    genre_embedding = tf.keras.layers.Embedding(num_genres, embedding_dimension)(genre_input)

    genre_embedding_mean = tf.reduce_mean(genre_embedding, axis=1)

    concatenated_embeddings = tf.concat([user_embedding, movie_embedding, genre_embedding_mean], axis=1)

    dense_1 = tf.keras.layers.Dense(256, activation="relu")(concatenated_embeddings)
    dense_2 = tf.keras.layers.Dense(128, activation="relu")(dense_1)
    rating_output = tf.keras.layers.Dense(1)(dense_2)

    user_model = tf.keras.Model(inputs=user_input, outputs=user_embedding)
    movie_model = tf.keras.Model(inputs=movie_input, outputs=movie_embedding)
    genre_model = tf.keras.Model(inputs=genre_input, outputs=genre_embedding_mean)
    rating_model = tf.keras.Model(inputs=[user_input, movie_input, genre_input], outputs=rating_output)

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=tf.data.Dataset.from_tensor_slices(unique_movie_ids).batch(128).map(movie_model)
    )
    rating_task = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    retrieval_task = tfrs.tasks.Retrieval(
        metrics=metrics
    )

    model = RecommendationModel(user_model, movie_model, genre_model, rating_model, rating_task, retrieval_task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    return model

# Create FAISS index
def extract_embeddings(model, unique_user_ids, unique_movie_ids):
    """
    Extract embeddings for all users and movies using the trained model.
    
    Args:
    - model: The trained recommendation model.
    - unique_user_ids: List of unique user IDs.
    - unique_movie_ids: List of unique movie IDs.
    
    Returns:
    - user_embeddings: Embeddings for all users.
    - movie_embeddings: Embeddings for all movies.
    """
    # Extract movie embeddings
    movie_ids = np.array(unique_movie_ids)
    movie_embeddings = model.movie_model(tf.constant(movie_ids, dtype=tf.int32)).numpy()

    # Extract user embeddings
    user_ids = np.array(unique_user_ids)
    user_embeddings = model.user_model(tf.constant(user_ids, dtype=tf.int32)).numpy()

    return user_embeddings, movie_embeddings

def index_movie_embeddings(movie_embeddings):
    """
    Index the movie embeddings using FAISS.
    
    Args:
    - movie_embeddings: Embeddings for all movies.
    
    Returns:
    - index: FAISS index with movie embeddings.
    """
    # Dimension of the embeddings
    embedding_dimension = movie_embeddings.shape[1]

    # Create a FAISS index
    index = faiss.IndexFlatL2(embedding_dimension)

    # Add movie embeddings to the index
    index.add(movie_embeddings)

    return index

def recommend_movies(model, index, unique_movie_ids, user_id, k=10):
    """
    Recommend movies for a given user by querying the FAISS index.
    
    Args:
    - model: The trained recommendation model.
    - index: FAISS index with movie embeddings.
    - unique_movie_ids: List of unique movie IDs.
    - user_id: The user ID for which to make recommendations.
    - k: Number of recommendations to retrieve (default is 10).
    
    Returns:
    - recommended_movie_ids: List of recommended movie IDs.
    """
    # Get the embedding for the given user
    user_embedding = model.user_model(tf.constant([user_id], dtype=tf.int32)).numpy()

    # Query the FAISS index
    distances, indices = index.search(user_embedding, k)

    # Get the recommended movie IDs
    recommended_movie_ids = np.array(unique_movie_ids)[indices[0]]

    return recommended_movie_ids

# Function to train the model
def train():
    # Load data from BigQuery
    try:
        logger.info("Loading data from BigQuery...")
        movies_bq = load_movies_bq()
        ratings_bq = load_ratings_bq()

        # Convert the DataFrame to a dictionary of lists
        logger.info("Converting DataFrames to dictionaries...")
        ratings_dict = {key: list(value) for key, value in ratings_bq[['movieId', 'userId', 'rating']].to_dict(orient='list').items()}
        movies_dict = {key: list(value) for key, value in movies_bq[['movie_id', 'genres']].to_dict(orient='list').items()}

        # Create TensorFlow datasets
        logger.info("Creating TensorFlow datasets...")
        ratings = tf.data.Dataset.from_tensor_slices(ratings_dict)
        movies = tf.data.Dataset.from_tensor_slices(movies_dict)

        # Map the datasets to extract relevant features
        logger.info("Mapping datasets to extract features...")
        ratings = ratings.map(lambda x: {
            "movie_id": x["movieId"],
            "user_id": x["userId"],
            "rating": x["rating"]
        })
        movies = movies.map(lambda x: {
            "movie_id": x["movie_id"],
            "genres": x["genres"]
        })

        movies_dict = {movie["movie_id"]: movie["genres"] for movie in movies.as_numpy_iterator()}

        def combine_datasets(rating):
            movie_id = rating["movie_id"]
            genres = tf.py_function(
                func=lambda x: movies_dict.get(int(x.numpy()), [0] * 19),
                inp=[movie_id],
                Tout=tf.int32
            )
            genres.set_shape([19])
            rating["genres"] = genres
            return rating

        # Step 3: Use the map function to combine
        combined_dataset = ratings.map(combine_datasets)
        combined_dataset = combined_dataset.map(lambda x: {
            "movie_id": x["movie_id"],
            "user_id": x["user_id"],
            "genres": x["genres"],
            "rating": x["rating"]
        })
        # Shuffle and split the data into training and testing sets
        logger.info("Shuffling and splitting the data...")
        # Set the seed for reproducibility
        tf.random.set_seed(42)

        # Shuffle the dataset with a large buffer size
        # Ensure the buffer size is large enough to cover randomness but not too large to exhaust memory
        shuffle_buffer_size = 100000  # Smaller buffer size for faster shuffling
        shuffled = combined_dataset.shuffle(buffer_size=shuffle_buffer_size, seed=42, reshuffle_each_iteration=True)

        # Calculate relative proportions for splitting
        train_ratio = 0.8

        ds_length = int(tf.data.experimental.cardinality(shuffled).numpy())
        print(f"Length of the dataset: {ds_length}")

        # Define the split function for large datasets
        def split_dataset(dataset, train_ratio):
            # Determine split points
            trainds = dataset.take(int(train_ratio * ds_length))
            testds = dataset.skip(int(train_ratio * ds_length))

            return trainds, testds

        # Perform the split
        trainds, testds = split_dataset(shuffled, train_ratio)

        # Optimize performance with prefetching
        train_batch_size = 8192
        eval_test_batch_size = 4096

        # Optimize datasets with batching, caching, and prefetching
        trainds = trainds.batch(train_batch_size).cache().prefetch(tf.data.AUTOTUNE)
        testds = testds.batch(eval_test_batch_size).cache().prefetch(tf.data.AUTOTUNE)

        # Batch the data
        logger.info("Batching the data...")
        movieIds = movies.batch(100000).map(lambda x: x["movie_id"])
        userIds = ratings.batch(1000000).map(lambda x: x["user_id"])
        genres = movies.batch(100000).map(lambda x: {
                "movie_id": x["movie_id"],
                "genres": x["genres"]
            })

        # Get unique movie and user IDs
        logger.info("Getting unique movie and user IDs...")
        unique_movieIds = np.unique(np.concatenate(list(movieIds)))
        unique_userIds = np.unique(np.concatenate(list(userIds)))
        unique_genres = [
                    'Action',
                    'Adventure',
                    'Animation',
                    'Children',
                    'Comedy',
                    'Crime',
                    'Drama',
                    'Documentary',
                    'Fantasy',
                    'Film-Noir',
                    'Horror',
                    'IMAX',
                    'Musical',
                    'Mystery',
                    'Romance',
                    'Sci-Fi',
                    'Thriller',
                    'War',
                    'Western'
                ]

        # Define callbacks for training
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        model_export_path = os.path.join(output_dir, 'saved-model', timestamp)
        checkpoint_path = os.path.join(output_dir, 'checkpoints', f"{timestamp}_cp.ckpt")
        tensorboard_path = os.path.join(output_dir, 'tensorboard', timestamp)
        faiss_path = os.path.join('trained_model', 'faiss', f"{timestamp}_faiss.index")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_path),
            save_best_only=True,
            save_weights_only=True,
            monitor='factorized_top_k/top_10_categorical_accuracy',
            mode='max'
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(tensorboard_path, histogram_freq=1)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='factorized_top_k/top_10_categorical_accuracy',
            patience=2,
            restore_best_weights=True
        )
         # Instantiate and compile the model
        logger.info("Creating and compiling the model...")
        model = create_model(unique_userIds, unique_movieIds, len(unique_genres))
        
        # Train the model
        model.fit(
            trainds,
            epochs=3,
            callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback]
        )

        model.save_weights(f"{output_dir}/weights/{timestamp}_weights.h5")
        # Extract embeddings
        user_embeddings, movie_embeddings = extract_embeddings(model, unique_userIds, unique_movieIds)

        # Index movie embeddings
        index = index_movie_embeddings(movie_embeddings)
        faiss.write_index(index, faiss_path)


    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == '__main__':
    train()