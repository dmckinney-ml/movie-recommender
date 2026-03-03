import platform

import tensorflow_recommenders as tfrs
from typing import Dict, Text
import tensorflow as tf
from keras_tuner import HyperModel, Hyperband, Objective
from keras_tuner.engine.hyperparameters import HyperParameters
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import ndcg_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
OUTPUT_DIR = 'gs://movie-data-1/multitask-xgb-trained-model'

class RecommendationModel(tfrs.Model):
    def __init__(self, user_model, movie_model, genre_model, rating_model, rating_task, retrieval_task, rating_weight=1.0, retrieval_weight=1.0):
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
        rating_predictions = self.rating_model([features["user_id"], features["title"], features["genres"]])
        return user_embeddings, movie_embeddings, genre_embeddings, rating_predictions

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        ratings = features.pop("rating")
        user_embeddings, movie_embeddings, genre_embeddings, rating_predictions = self(features)
        rating_loss = self.rating_task(labels=ratings, predictions=rating_predictions)
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)

class RecommendationHyperModel(HyperModel):
    def __init__(self, unique_user_ids, unique_titles, num_genres, rating_weight=2.0, retrieval_weight=0.5):
        self.unique_user_ids = unique_user_ids
        self.unique_titles = unique_titles
        self.num_genres = num_genres
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def build(self, hp):
        embedding_dimension = hp.Int('embedding_dimension', min_value=32, max_value=256, step=32)

        # L2 regularisation applied to all embedding and dense layers
        l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log')

        user_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='user_id')
        movie_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='title')
        genre_input = tf.keras.layers.Input(shape=(self.num_genres,), dtype=tf.float32, name='genres')

        user_lookup = tf.keras.layers.IntegerLookup(vocabulary=self.unique_user_ids, mask_token=None)
        movie_lookup = tf.keras.layers.StringLookup(vocabulary=self.unique_titles, mask_token=None)

        user_embedding = tf.keras.layers.Embedding(
            len(self.unique_user_ids) + 1,
            embedding_dimension,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(user_lookup(user_input))
        movie_embedding = tf.keras.layers.Embedding(
            len(self.unique_titles) + 1,
            embedding_dimension,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(movie_lookup(movie_input))
        genre_embedding = tf.keras.layers.Dense(
            embedding_dimension,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(genre_input)

        concatenated_embeddings = tf.concat([user_embedding, movie_embedding, genre_embedding], axis=1)

        dense_1 = tf.keras.layers.Dense(
            hp.Int('units_1', min_value=128, max_value=512, step=64),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(concatenated_embeddings)
        dense_1 = tf.keras.layers.BatchNormalization()(dense_1)
        dropout_1 = tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.3, max_value=0.6, step=0.1))(dense_1)

        dense_2 = tf.keras.layers.Dense(
            hp.Int('units_2', min_value=64, max_value=256, step=32),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(dropout_1)
        dense_2 = tf.keras.layers.BatchNormalization()(dense_2)
        dropout_2 = tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.3, max_value=0.6, step=0.1))(dense_2)
        rating_output = tf.keras.layers.Dense(1)(dropout_2)

        user_model = tf.keras.Model(inputs=user_input, outputs=user_embedding)
        movie_model = tf.keras.Model(inputs=movie_input, outputs=movie_embedding)
        genre_model = tf.keras.Model(inputs=genre_input, outputs=genre_embedding)
        rating_model = tf.keras.Model(inputs=[user_input, movie_input, genre_input], outputs=rating_output)

        metrics = tfrs.metrics.FactorizedTopK(
            candidates=tf.data.Dataset.from_tensor_slices(self.unique_titles).batch(128).map(movie_model)
        )
        rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        retrieval_task = tfrs.tasks.Retrieval(
            metrics=metrics,
            temperature=0.1
        )
        model = RecommendationModel(user_model, movie_model, genre_model, rating_model, rating_task, retrieval_task, self.rating_weight, self.retrieval_weight)

        # Define the learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'),
            decay_steps=hp.Int('decay_steps', min_value=500, max_value=1000, step=100),
            decay_rate=hp.Float('decay_rate', min_value=0.8, max_value=0.9, step=0.05),
            staircase=True
        )

        # Use legacy Adam on Apple Silicon (Metal backend), standard Adam on Linux/GCP.
        # tf.keras.optimizers.legacy is not available on Linux.
        if platform.system() == 'Darwin':
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer)
        return model

def create_hypermodel_tuner(unique_user_ids, unique_titles, unique_genres, timestamp):
    try:
        logger.info("Creating Hyperband tuner...")
        tuner = Hyperband(
            RecommendationHyperModel(unique_user_ids, unique_titles, len(unique_genres)),
            objective=Objective("val_root_mean_squared_error", direction="min"),
            max_epochs=12,
            factor=3,
            directory='gs://movie-data-1/tuning',
            project_name=f'{timestamp}/movie_recommendation',
        )
        return tuner
    except Exception as e:
        logger.error(f"Error creating Hyperband tuner: {e}")
        raise

def tune_hypermodel(tuner: Hyperband, train_ds, val_ds, epochs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=3, mode='min')]):
    try:
        logger.info("Tuning hypermodel...")
        tuner.search(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        np.save(f'{OUTPUT_DIR}/params/best_hps.npy', best_hps.values)
        tuned_model = tuner.hypermodel.build(best_hps)
        return tuned_model
    except Exception as e:
        logger.error(f"Error tuning hypermodel: {e}")
        raise

def train_hypermodel(tuned_model, train_ds, val_ds, epochs, callbacks=[]):
    try:
        logger.info("Training hypermodel...")
        history = tuned_model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        return history
    except Exception as e:
        logger.error(f"Error training hypermodel: {e}")
        raise

def reload_hypermodel(tuner: Hyperband):
    try:
        logger.info("Reloading hypermodel...")
        best_hps_values = np.load("best_hps.npy", allow_pickle=True).item()
        best_hps = HyperParameters()
        for key, value in best_hps_values.items():
            best_hps.Fixed(key, value)
        tuned_model = tuner.hypermodel.build(best_hps)
        tuned_model.load_weights(f'{OUTPUT_DIR}/tpe/weights/20250122063906_weights')
        return tuned_model
    except Exception as e:
        logger.error(f"Error reloading hypermodel: {e}")
        raise

def tune_xgb_model(dtrain, dval, val_df):
    """Tune XGBoost hyperparameters with Optuna.

    Args:
        dtrain: XGBoost DMatrix for training data.
        dval:   XGBoost DMatrix for validation data.
        val_df: Validation DataFrame with columns ['user_id', 'rating'] aligned
                to the rows of dval — used for per-user NDCG computation.
    """
    try:
        logger.info("Tuning XGBoost model with Optuna...")

        def objective(trial):
            param = {
                'objective': 'rank:pairwise',
                'eval_metric': 'ndcg',
                'eta': trial.suggest_float('eta', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'lambda': trial.suggest_float('lambda', 0.0, 1.0),
            }

            bst = xgb.train(param, dtrain, num_boost_round=300, evals=[(dval, 'eval')],
                            early_stopping_rounds=20, verbose_eval=False)
            y_pred = bst.predict(dval)

            min_pred, max_pred = np.min(y_pred), np.max(y_pred)
            y_pred_scaled = 0.5 + (y_pred - min_pred) * 4.5 / (max_pred - min_pred)

            # Per-user NDCG — computing globally over the full val set trivially
            # inflates the score to ~0.99. Average over individual users instead.
            user_ndcg_scores = []
            val_df_reset = val_df.reset_index(drop=True)
            for uid in val_df_reset['user_id'].unique():
                mask = (val_df_reset['user_id'] == uid).values
                true_r = val_df_reset.loc[mask, 'rating'].values
                pred_r = y_pred_scaled[mask]
                if len(true_r) > 1:
                    user_ndcg_scores.append(ndcg_score([true_r], [pred_r]))

            return np.mean(user_ndcg_scores) if user_ndcg_scores else 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        np.save(f'{OUTPUT_DIR}/params/best_params.npy', best_params)
        return best_params
    except Exception as e:
        logger.error(f"Error tuning XGBoost model: {e}")
        raise

def train_xgb_model(dtrain, dval, best_params, timestamp):
    try:
        logger.info("Training XGBoost model...")
        bst = xgb.train(
            best_params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=10,
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=20,
                    metric_name='rmse',
                    data_name='eval',
                    min_delta=1e-4
                )
            ]
        )
        logger.info(f"Best iteration: {bst.best_iteration}, Best score: {bst.best_score:.5f}")
        bst.save_model(f'{OUTPUT_DIR}/xgb/models/{timestamp}_model.json')
        return bst
    except Exception as e:
        logger.error(f"Error training XGBoost model: {e}")
        raise

def reload_xgb_model(timestamp):
    try:
        logger.info("Reloading XGBoost model...")
        bst = xgb.Booster()
        bst.load_model(f'{OUTPUT_DIR}/xgb/models/{timestamp}_model.json')
        return bst
    except Exception as e:
        logger.error(f"Error reloading XGBoost model: {e}")
        raise