from os import path
import kfp.compiler as compiler
import kfp.components as comp
import kfp.dsl as dsl
import google.cloud.aiplatform as aip

# Set the path to the components and the compiled pipeline tar file
HERE = path.abspath(path.dirname(__file__))
COMPONENT_DIR = path.join(HERE, "components")
PREPROCESSING_YAML = path.join(COMPONENT_DIR, 'dataflow_preprocessing/preprocessing.yaml')
PIPELINE_YAML = path.join(HERE, 'movie-pipeline.yaml') 

# Define your pipeline
@dsl.pipeline(
    name='Movie Dataflow Pipeline',
    description='Preprocess data using Google Cloud Dataflow with a custom container.'
)
def pipeline(
    gcs_bucket_name: str = 'movie-data-1',
):
    # Load the preprocessing component from the YAML file
    preprocessing_op = comp.load_component_from_file(PREPROCESSING_YAML)
    
    # Invoke the operation and pass necessary parameters
    preprocessing_task = preprocessing_op(
        bucket=gcs_bucket_name
    )

# Compile the pipeline into a .tar.gz file
if __name__ == '__main__':
    # Compile the pipeline
    compiler.Compiler().compile(pipeline, PIPELINE_YAML, type_check=False)
    aip.init(
        project='oolola',
        location='us-central1',
    )

    # Prepare the pipeline job
    job = aip.PipelineJob(
        display_name="movie-recommendation-pipeline",
        template_path=PIPELINE_YAML,
        pipeline_root='gs://pipeline-root-1',
        parameter_values={
            'gcs_bucket_name': 'movie-data-1'
        }
    )

    job.submit(service_account='pipeline-svc-acct@oolola.iam.gserviceaccount.com')