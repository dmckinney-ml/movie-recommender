import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions

# Define pipeline options
options = PipelineOptions(save_main_session=True)
google_cloud_options = options.view_as(GoogleCloudOptions)
options.view_as(StandardOptions).runner = 'DataflowRunner'
google_cloud_options.project = 'oolola'  # Replace with your project ID
google_cloud_options.region = 'us-central1'  # Set your region
google_cloud_options.job_name = 'dataflow-movie-preprocessing'
google_cloud_options.staging_location = 'gs://movie-data-1/staging'  # Your GCS staging bucket
google_cloud_options.temp_location = 'gs://movie-data-1/temp'  # Your GCS temp bucket

# Define the pipeline
def run():
    with beam.Pipeline(options=options) as p:
        # Read from BigQuery and extract unique genres
        movies = (
            p
            | "Read Movies Table" >> beam.io.ReadFromBigQuery(
                query="""
                SELECT movieId, title, genres
                FROM `oolola.movie_data.movies`
                WHERE movieId IN (1, 2, 3, 4, 5)
                """,
                use_standard_sql=True
            )
            | "Map Movies to Key-Value Pairs" >> beam.Map(lambda row: (row['movieId'], row))
        )

        ratings = (
            p
            | "Read Ratings Table" >> beam.io.ReadFromBigQuery(
                query="""
                SELECT userId, movieId, rating, timestamp
                FROM `oolola.movie_data.ratings`
                WHERE movieId IN (1, 2, 3, 4, 5)
                """,
                use_standard_sql=True
            )
            | "Map Ratings to Key-Value Pairs" >> beam.Map(lambda row: (row['movieId'], row))
        )

        # Extract unique genres from the movies data
        unique_genres = [
            'Action',
            'Adventure',
            'Animation',
            'Children',
            'Comedy',
            'Crime',
            'Drama',
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

        # Perform data transformations
        preprocessed_data = (
            {'movies': movies, 'ratings': ratings}
            | "Join Data" >> beam.CoGroupByKey()
            | "Transform Data" >> beam.FlatMap(preprocess_data, unique_genres=unique_genres)  # Pass unique genres for encoding
        )

        # Write the preprocessed data to BigQuery
        preprocessed_data | "Write to BigQuery" >> beam.io.WriteToBigQuery(
            table='oolola:movie_data.preprocessed_data',
            schema=get_bq_schema(),  # Dynamically generate schema
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        )

# Update the preprocess_data function to dynamically accept genres
def preprocess_data(element, unique_genres):
    movie_id, grouped_data = element
    movies = {movie['movieId']: movie for movie in grouped_data['movies']}
    ratings = grouped_data['ratings']

    for rating in ratings:
        # Get the genres for the current movie
        movie_genres = movies.get(rating['movieId'], {}).get('genres', '').split('|')

        # Generate multi-hot encoding vector for the movie
        genre_vector = [1 if genre in movie_genres else 0 for genre in unique_genres]

        yield {
            'userId': rating['userId'],
            'movieId': rating['movieId'],
            'title': movies.get(rating['movieId'], {}).get('title', None),
            'genres': genre_vector,  # Use the multi-hot encoding for genres
            'rating': rating['rating'],
            'timestamp': rating['timestamp'],
        }

# Dynamically generate the BigQuery schema
def get_bq_schema():
    return {
        'fields': [
            {'name': 'userId', 'type': 'INTEGER'},
            {'name': 'movieId', 'type': 'INTEGER'},
            {'name': 'title', 'type': 'STRING'},
            {'name': 'genres', 'type': 'INTEGER', 'mode': 'REPEATED'},
            {'name': 'rating', 'type': 'FLOAT'},
            {'name': 'timestamp', 'type': 'TIMESTAMP'},
        ]
    }

if __name__ == "__main__":
    run()
