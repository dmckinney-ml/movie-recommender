import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
import logging
from models import (create_hypermodel_tuner, tune_hypermodel, train_hypermodel,
                    tune_xgb_model, train_xgb_model)
from dataset import create_xgb_data, load_movies_bq, load_ratings_bq, create_tf_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Function to train the model
def train():
    try:
        # Load data from BigQuery
        logger.info("Loading data from BigQuery...")
        movies_bq = load_movies_bq()
        ratings_bq = load_ratings_bq()

        # Convert the DataFrame to a dictionary of lists
        logger.info("Converting DataFrames to dictionaries...")
        ratings_dict = {key: list(value) for key, value in ratings_bq[['title', 'user_id', 'rating']].to_dict(orient='list').items()}
        movies_dict = {key: list(value) for key, value in movies_bq[['title', 'genres']].to_dict(orient='list').items()}

        # Create TensorFlow datasets
        logger.info("Creating TensorFlow datasets...")
        ratings = tf.data.Dataset.from_tensor_slices(ratings_dict)
        movies = tf.data.Dataset.from_tensor_slices(movies_dict)

        # Map the datasets to extract relevant features
        logger.info("Mapping datasets to extract features...")
        ratings = ratings.map(lambda x: {
            "title": x["title"],
            "user_id": x["user_id"],
            "rating": x["rating"]
        })
        movies = movies.map(lambda x: {
            "title": x["title"],
            "genres": x["genres"]
        })

        # Create a dictionary to map movie titles to genres
        movies_dict = {movie["title"].numpy().decode('utf-8'): movie["genres"].numpy() for movie in movies}

        # Function to combine datasets by adding genres to ratings
        def combine_datasets(rating):
            def lookup_genres(title):
                title_str = title.numpy().decode('utf-8')  # Convert to numpy and decode the title from bytes to string
                return movies_dict.get(title_str, [0] * 19)

            genres = tf.py_function(
                func=lookup_genres,
                inp=[rating["title"]],
                Tout=tf.int32
            )
            genres.set_shape([19])
            rating["genres"] = genres
            return rating

        # Combine datasets
        logger.info("Combining datasets...")
        combined_dataset = ratings.map(combine_datasets)
        combined_dataset = combined_dataset.map(lambda x: {
            "title": x["title"],
            "user_id": x["user_id"],
            "genres": x["genres"],
            "rating": x["rating"]
        }, num_parallel_calls=tf.data.AUTOTUNE)

        # User-stratified train/test split — pass ratings_bq so the split is by
        # user_id rather than by row, preventing embedding-level data leakage.
        # train_combined is also returned for XGBoost feature extraction.
        logger.info("Shuffling and splitting the data...")
        trainds, testds, train_combined, train_ratings_bq = create_tf_dataset(ratings_bq, combine_datasets)

        # Batch the data
        logger.info("Batching the data...")
        titles = movies.batch(100000).map(lambda x: x["title"])
        user_ids = ratings.batch(1000000).map(lambda x: x["user_id"])

        # Get unique movie and user IDs
        logger.info("Getting unique movie and user IDs...")
        unique_titles = np.unique(np.concatenate(list(titles)))
        unique_user_ids = np.unique(np.concatenate(list(user_ids)))
        unique_genres = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Drama', 'Documentary', 'Fantasy',
            'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        # Create and tune the hypermodel
        logger.info("Creating and tuning the hypermodel...")
        tuner = create_hypermodel_tuner(unique_user_ids, unique_titles, unique_genres, timestamp)
        tuned_model = tune_hypermodel(tuner, trainds, testds, 12)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"trained_model/tpe/checkpoints/{timestamp}_cp.ckpt",
            save_best_only=True,
            save_weights_only=True,
            monitor='val_factorized_top_k/top_5_categorical_accuracy',
            mode='max'
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(f"trained_model/tpe/tensorboard/{timestamp}_cp.ckpt", histogram_freq=1)
        train_hypermodel(tuned_model, trainds, testds, 12, callbacks=[checkpoint_callback, tensorboard_callback])

        # Extract user and movie embeddings — batched to avoid OOM on large catalogues
        logger.info("Extracting user and movie embeddings...")
        batch_size = 512

        titles_arr = np.array(unique_titles)
        movie_embeddings = np.vstack([
            tuned_model.movie_model(tf.constant(titles_arr[i:i+batch_size], dtype=tf.string)).numpy()
            for i in range(0, len(titles_arr), batch_size)
        ]).astype(np.float32)

        user_ids_arr = np.array(unique_user_ids)
        user_embeddings = np.vstack([
            tuned_model.user_model(tf.constant(user_ids_arr[i:i+batch_size], dtype=tf.int32)).numpy()
            for i in range(0, len(user_ids_arr), batch_size)
        ]).astype(np.float32)

        # Create a dictionary to map user IDs and movie IDs to their embeddings
        user_embedding_dict = {user_id: embedding for user_id, embedding in zip(unique_user_ids, user_embeddings)}
        movie_embedding_dict = {movie_id: embedding for movie_id, embedding in zip(unique_titles, movie_embeddings)}
        movie_embedding_dict = {k.decode('utf-8'): v for k, v in movie_embedding_dict.items()}

        # Create feature vectors by combining user, movie, and genres embeddings
        def create_feature_vector(row):
            user_id = row['user_id']
            title = row['title'].decode('utf-8')
            genres = row['genres']

            if user_id not in user_embedding_dict:
                raise KeyError(f"User ID {user_id} not found in user_embedding_dict")
            if title not in movie_embedding_dict:
                raise KeyError(f"Title '{title}' not found in movie_embedding_dict")

            user_embedding = user_embedding_dict[user_id]
            movie_embedding = movie_embedding_dict[title]

            return np.concatenate([user_embedding, movie_embedding, genres])

        # Convert combined dataset to DataFrame
        # Use the TRAINING partition only — building XGBoost features from the
        # full dataset would leak held-out interactions into XGBoost training.
        logger.info("Converting training dataset to DataFrame for XGBoost...")
        xgb_df = pd.DataFrame(train_combined.as_numpy_iterator())

        # Apply the function to create feature vectors
        try:
            logger.info("Creating feature vectors...")
            xgb_df['features'] = xgb_df.apply(create_feature_vector, axis=1)
        except KeyError as e:
            logger.error(f"Error creating feature vectors: {e}")
            raise

        # Create the DMatrix for XGBoost
        logger.info("Creating DMatrix for XGBoost...")
        dtrain, dval, val_df = create_xgb_data(xgb_df)

        # Tune and train the XGBoost model
        logger.info("Tuning and training the XGBoost model...")
        best_xgb_params = tune_xgb_model(dtrain, dval, val_df)
        bst = train_xgb_model(dtrain, dval, best_xgb_params, timestamp)

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise