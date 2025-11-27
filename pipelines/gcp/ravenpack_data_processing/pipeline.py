import logging
import os

import google.auth
from kfp import dsl
from kfp.dsl import Input, Output, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# This is a placeholder. You can replace this string before compilation
# or use a build script to inject the correct URI.
# Example: os.environ.get("PIPELINE_BASE_IMAGE", "your-default-image")
BASE_IMAGE = os.environ.get("PIPELINE_BASE_IMAGE", "YOUR_ARTIFACT_REGISTRY_IMAGE_URI_HERE")
DEFAULT_GCP_REGION = "us-central1"
_, project_id = google.auth.default()


@dsl.component(base_image=BASE_IMAGE)
def process_ravenpack_data_op(
    gcp_bucket: str,
    gcp_prefix: str,
    output_dir: Output[Dataset],
):
    import logging

    import polars as pl

    # Configure Cloud Logging
    logger = logging.getLogger(__name__)
    logger.info("Starting 'process_ravenpack_data' step")

    from fin_news_ravenpack_data.process import process_ravenpack_data

    logger.info("Downloading input data from GCS")
    logger.info(f"source: {f"gs://{gcp_bucket}/{gcp_prefix}/data/*.jsonl"}")
    df = pl.read_ndjson(
        source=f"gs://{gcp_bucket}/{gcp_prefix}/data/*.jsonl",
        storage_options={"google_bucket": gcp_bucket},
        include_file_paths="file_path"
    )

    logger.info("Processing data")
    df = process_ravenpack_data(df)

    # Upload processed Ravenpack data to GCS
    logger.info(
        f"Uploading processed Ravenpack data to {output_dir.path}"
    )
    df.write_parquet(output_dir.path)

    logger.info("'process_ravenpack_data' step completed successfully.")


@dsl.component(base_image=BASE_IMAGE)
def write_news_to_bigquery_op(
    processed_parquet: Input[Dataset],
    project_id: str,
    gcp_region: str,
    output_dataset_id: str = "fin_news_datasets",
    output_table_name: str = "ravenpack_news",
    time_partition_field: str = "timestamp",
    stable_key_field: str = "id",
):
    import logging
    from datetime import datetime, timedelta

    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound

    # Configure Cloud Logging
    logger = logging.getLogger(__name__)
    logger.info("Starting 'process_ravenpack_data' step")

    # Defining BigQuery client
    client = bigquery.Client(project=project_id, location=gcp_region)

    try:
        # Create the dataset for the job's output if it does not exist
        dataset_ref = bigquery.Dataset(f"{project_id}.{output_dataset_id}")
        dataset_ref.location = gcp_region
        _ = client.create_dataset(dataset_ref, exists_ok=True)

        output_table_id = f"{project_id}.{output_dataset_id}.{output_table_name}"
        staging_table_name = f"{output_table_name}_staging"
        staging_table_id = f"{project_id}.{output_dataset_id}.{staging_table_name}"

        # Load Parquet into staging table (truncate each run)
        staging_job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        load_job = client.load_table_from_uri(
            processed_parquet.uri,
            staging_table_id,
            job_config=staging_job_config,
        )
        load_job.result()
        staging_table = client.get_table(staging_table_id)
        staging_table.expires = datetime.now() + timedelta(days=7)
        client.update_table(staging_table, ["expires"])


        # Ensure main table exists; if not, create from staging schema
        try:
            target_table = client.get_table(output_table_id)
        except NotFound:
            target_table = bigquery.Table(
                output_table_id,
                schema=staging_table.schema,
            )
            target_table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=time_partition_field,
            )
            target_table.clustering_fields = [stable_key_field]
            target_table = client.create_table(target_table)
            logger.info("Created target table %s", output_table_id)

        # 5) MERGE from staging into main table to avoid duplicates on stable_key_field
        merge_sql = f"""
        MERGE `{output_table_id}` T
        USING `{staging_table_id}` S
        ON T.{stable_key_field} = S.{stable_key_field}
        WHEN NOT MATCHED THEN
          INSERT ROW
        """

        merge_job = client.query(merge_sql, location=gcp_region)
        merge_job.result()

        # Number of rows actually written into the main table (after dedup)
        num_rows_inserted = merge_job.num_dml_affected_rows
        logger.info(
            "write_news_to_bigquery_op: inserted %s new rows into %s after deduplication",
            num_rows_inserted,
            output_table_id,
        )

    except Exception as e:
        logger.error(f"Error in write_news_to_bigquery_op: {str(e)}")
        raise


@dsl.pipeline(
    name="ravenpack-data-process-pipeline",
    description="Processes extracted Ravenpack data and saves to BigQuery"
)
def ravenpack_data_process_pipeline(
    gcp_bucket: str,
    gcp_prefix: str,
    gcp_region: str = DEFAULT_GCP_REGION,
    output_dataset_id: str = "fin_news_datasets",
    output_table_name: str = "ravenpack_news",
    time_partition_field: str = "timestamp",
    stable_key_field: str = "id",
):
    # 1. Download Ravenpack Data
    download_task = process_ravenpack_data_op(
        gcp_bucket=gcp_bucket,
        gcp_prefix=gcp_prefix,
    )
    download_task.set_display_name("Process Ravenpack Data")

    # 2. Write to BigQuery
    write_task = write_news_to_bigquery_op(
        processed_parquet=download_task.outputs["output_dir"],
        project_id=project_id,
        gcp_region=gcp_region,
        output_dataset_id=output_dataset_id,
        output_table_name=output_table_name,
        time_partition_field=time_partition_field,
        stable_key_field=stable_key_field
    )
    write_task.set_display_name("Write Ravenpack Data to BigQuery")
