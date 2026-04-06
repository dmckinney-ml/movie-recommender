import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions, WorkerOptions
import re
import datetime

# Define pipeline options
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
options = PipelineOptions(save_main_session=True)
google_cloud_options = options.view_as(GoogleCloudOptions)
options.view_as(StandardOptions).runner = 'DataflowRunner'
worker_options = options.view_as(WorkerOptions)
worker_options.sdk_container_image = 'us-central1-docker.pkg.dev/YOUR-PROJECT-ID/YOUR-PROJECT-ID-repo/moviedata-dataflow_preprocessing:latest'

google_cloud_options.project = 'YOUR-PROJECT-ID'  # Replace with your project ID
google_cloud_options.region = 'us-central1'  # Set your region
google_cloud_options.job_name = 'dataflow-movie-preprocessing-{}'.format(timestamp)
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
                FROM `YOUR-PROJECT-ID.movie_data.movies`
                """,
                use_standard_sql=True
            )
            | "Map Movies to Key-Value Pairs" >> beam.Map(lambda row: (row['movieId'], row))
        )

        ratings = (
            p
            | "Read Ratings Table" >> beam.io.ReadFromBigQuery(
                query="""
                SELECT movieId, rating, userId, timestamp
                FROM `YOUR-PROJECT-ID.movie_data.ratings`
                """,
                use_standard_sql=True
            )
            | "Map Ratings to Key-Value Pairs" >> beam.Map(lambda row: (row['movieId'], row))
        )

        # Perform a join operation to replace movieId with title
        joined = (
            {'movies': movies, 'ratings': ratings}
            | "Join Movies and Ratings" >> beam.CoGroupByKey()
            | "Replace movieId with title" >> beam.FlatMap(replace_movie_id_with_title)
        )

        # Filter out ratings without a parseable year — no year-range restriction
        # so the full 32M dataset (including pre-2021 classics) enters training.
        filtered_movies = (
            joined
            | "Filter Ratings With Valid Year" >> beam.Filter(lambda row: row['year'] is not None)
        )

        # Write the filtered data to BigQuery
        filtered_movies | "Write Ratings with Titles to BigQuery" >> beam.io.WriteToBigQuery(
            table='YOUR-PROJECT-ID:movie_data.ratings_with_titles',
            schema=get_ratings_schema(),  # Dynamically generate schema
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
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

        # Perform data transformations
        preprocessed_movies = (
            movies
            | "Transform Data" >> beam.FlatMap(preprocess_movies, unique_genres=unique_genres)  # Pass unique genres for encoding
            | "Filter Movies With Valid Year" >> beam.Filter(lambda row: row['year'] is not None)
        )

        # Write the preprocessed data to BigQuery
        preprocessed_movies | "Write preprocessed movie table to BigQuery" >> beam.io.WriteToBigQuery(
            table='YOUR-PROJECT-ID:movie_data.preprocessed_movies',
            schema=get_movies_schema(),  # Dynamically generate schema
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        )

def preprocess_movies(element, unique_genres):
    movie = element[1]  # Extract the movie dictionary
    movie_id = movie['movieId']
    title = movie['title']
    genres = movie['genres'].split('|')

    # Extract year from title
    match = re.search(r'\((\d{4})\)', title)
    if match:
        year = int(match.group(1))
        title = title[:match.start()].strip()
    else:
        year = None

    genre_vector = [1 if genre in genres else 0 for genre in unique_genres]
    yield {
        'movie_id': movie_id,
        'title': title,
        'year': year,
        'genres': genre_vector,
    }

def get_movies_schema():
    return {
        'fields': [
            {'name': 'movie_id', 'type': 'INTEGER'},
            {'name': 'title', 'type': 'STRING'},
            {'name': 'year', 'type': 'INTEGER'},
            {'name': 'genres', 'type': 'INTEGER', 'mode': 'REPEATED'},
        ]
    }

def replace_movie_id_with_title(element):
    movie_id, grouped = element
    movies = grouped['movies']
    ratings = grouped['ratings']

    if not movies:
        return

    # Exclude movies with fewer than 5 ratings to filter noise while keeping
    # popular classics that were previously excluded by the year > 2020 filter.
    if len(ratings) < 5:
        return

    title = movies[0]['title']
    year = None
    match = re.search(r'\((\d{4})\)', title)
    if match:
        year = int(match.group(1))
        title = title[:match.start()].strip()

    for rating in ratings:
        yield {
            'title': title,
            'year': year,
            'rating': rating['rating'],
            'user_id': rating['userId'],
            'timestamp': rating['timestamp']
        }

def get_ratings_schema():
    return {
        'fields': [
            {'name': 'title', 'type': 'STRING'},
            {'name': 'year', 'type': 'INTEGER'},
            {'name': 'rating', 'type': 'FLOAT'},
            {'name': 'user_id', 'type': 'INTEGER'},
            {'name': 'timestamp', 'type': 'TIMESTAMP'},
        ]
    }

if __name__ == "__main__":
    run()
