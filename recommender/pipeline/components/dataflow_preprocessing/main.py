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
                """,
                use_standard_sql=True
            )
            | "Map Movies to Key-Value Pairs" >> beam.Map(lambda row: (row['movieId'], row))
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
        preprocessed_genres = (
            movies
            | "Transform Data" >> beam.FlatMap(preprocess_for_genres, unique_genres=unique_genres)  # Pass unique genres for encoding
        )

        # Write the preprocessed data to BigQuery
        preprocessed_genres | "Write Genres table to BigQuery" >> beam.io.WriteToBigQuery(
            table='oolola:movie_data.genres',
            schema=get_genres_schema(),  # Dynamically generate schema
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        )

def preprocess_for_genres(element, unique_genres):
    movie = element[1]  # Extract the movie dictionary
    movie_id = movie['movieId']
    title = movie['title']
    genres = movie['genres'].split('|')
    genre_vector = [1 if genre in genres else 0 for genre in unique_genres]
    yield {
        'movie_id': movie_id,
        'title': title,
        'genres': genre_vector,
    }

def get_genres_schema():
    return {
        'fields': [
            {'name': 'movie_id', 'type': 'INTEGER'},
            {'name': 'title', 'type': 'STRING'},
            {'name': 'genres', 'type': 'INTEGER', 'mode': 'REPEATED'},
        ]
    }

if __name__ == "__main__":
    run()
