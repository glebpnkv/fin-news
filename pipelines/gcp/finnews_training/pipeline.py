import logging
import os

import google.auth
from kfp import dsl
from kfp.dsl import Output, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)

# This is a placeholder. You can replace this string before compilation
# or use a build script to inject the correct URI.
# Example: os.environ.get("PIPELINE_BASE_IMAGE", "your-default-image")
BASE_IMAGE = os.environ.get("PIPELINE_BASE_IMAGE", "YOUR_ARTIFACT_REGISTRY_IMAGE_URI_HERE")
DEFAULT_GCP_REGION = "us-central1"
_, project_id = google.auth.default()


@dsl.component(base_image=BASE_IMAGE)
def write_train_data_op(
    project_id: str,
    gcp_region: str,
    embeddings_query_template: str,
    returns_query_template: str,
    start_time: str = "2021-01-01",
    end_time: str = "2026-01-01",
    output_dir: Output[Dataset] = Output[Dataset],
):
    import logging
    import os
    import shutil

    from google.cloud import bigquery
    from google.cloud import bigquery_storage_v1

    from fin_news_models.finnews.data import write_embeddings_ts_dataset, write_returns_ts_dataset

    # Configure Cloud Logging
    logger = logging.getLogger(__name__)
    logger.info("Starting 'write_train_data_op' step")

    # Set up a temporary directory
    temp_data_dir = "/tmp/data"
    os.makedirs(temp_data_dir, exist_ok=True)

    # Define BigQuery client
    client = bigquery.Client(project=project_id, location=gcp_region)
    bqstorage = bigquery_storage_v1.BigQueryReadClient()

    # Bigquery job configs with start_time and end_time parameters
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", start_time),
            bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", end_time),
        ]
    )

    # Set up counts queries
    embeddings_count_query_template = f"""
    WITH t as ({embeddings_query_template}) SELECT COUNT(*) as count from t
    """
    returns_count_query_template = f"""
    WITH t as ({returns_query_template}) SELECT COUNT(*) as count from t
    """
    # Set up data queries

    # Get sizes of embeddings and returns tables
    job_embeddings_counts = client.query(embeddings_count_query_template, job_config=job_config)
    job_returns_counts = client.query(returns_count_query_template, job_config=job_config)
    n_embeddings = job_embeddings_counts.to_dataframe()["count"][0]
    n_returns = job_returns_counts.to_dataframe()["count"][0]

    # Writing tensorstore datasets
    output_dir_embeddings = os.path.join(temp_data_dir, "embeddings")
    output_dir_returns = os.path.join(temp_data_dir, "returns")

    job_embeddings = client.query(embeddings_query_template, job_config=job_config)
    df_embeddings_iter = (
        job_embeddings
        .result()
        .to_dataframe_iterable(
            bqstorage_client=bqstorage,
            max_queue_size=2,
            max_stream_count=1,
        )
    )
    write_embeddings_ts_dataset(df_embeddings_iter, n_embeddings, output_dir_embeddings)

    job_returns = client.query(returns_query_template, job_config=job_config)
    df_returns_iter = (
        job_returns
        .result()
        .to_dataframe_iterable(
            bqstorage_client=bqstorage,
            max_queue_size=2,
            max_stream_count=1,
        )
    )
    write_returns_ts_dataset(df_returns_iter, n_returns, output_dir_returns)

    # Move all outputs
    shutil.move(temp_data_dir, output_dir.path)
    logger.info(f"Successfully wrote {n_embeddings} embeddings and {n_returns} returns as tensorstore datasets")
    logger.info(f"'write_train_data_op' step completed successfully.")


@dsl.pipeline(
    name="finnews-model-training-pipeline",
    description="FinNews model training pipeline",
)
def finnews_model_training_pipeline(
    embeddings_query_template: str,
    returns_query_template: str,
    gcp_region: str = DEFAULT_GCP_REGION,
    start_time: str = "2021-01-01",
    end_time: str = "2026-01-01"
):
    write_train_data_task = write_train_data_op(
        project_id=project_id,
        gcp_region=gcp_region,
        embeddings_query_template=embeddings_query_template,
        returns_query_template=returns_query_template,
        start_time=start_time,
        end_time=end_time
    )
    write_train_data_task.set_display_name("Write train data to GCS")
    write_train_data_task.set_caching_options(True)
