import logging
import os
from typing import Optional

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
DEFAULT_EMBEDDING_MODEL = "text-embedding-005"
VERTEX_COMPONENT_PACKAGES = [
    "google-cloud-aiplatform",
    "google-cloud-bigquery",
    "google-genai",
    "kfp",
]
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
    logger.info(f"source: 'gs://{gcp_bucket}/{gcp_prefix}/data/*.jsonl'")
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


@dsl.component(
    base_image="python:3.12",
    packages_to_install=VERTEX_COMPONENT_PACKAGES,
)
def write_news_to_bigquery_op(
    processed_parquet: Input[Dataset],
    project_id: str,
    gcp_region: str,
    output_dataset_id: str = "fin_news_datasets",
    output_table_name: str = "ravenpack_news",
    time_partition_field: str = "timestamp",
    stable_key_field: str = "id",
    new_rows_table_name: Optional[str] = None,
    table_expiry_days: int = 7,
) -> str:
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
        new_rows_table_name = new_rows_table_name or f"{output_table_name}_new_batch"
        new_rows_table_id = f"{project_id}.{output_dataset_id}.{new_rows_table_name}"

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


        # Ensure the main table exists; if not, create from staging schema
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
            _ = client.create_table(target_table)
            logger.info("Created target table %s", output_table_id)

        # Build a table with only the new rows from this load
        new_rows_sql = f"""
        CREATE OR REPLACE TABLE `{new_rows_table_id}` AS
        WITH deduped_staging AS (
            SELECT * EXCEPT(row_num)
            FROM (
                SELECT
                    S.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY {stable_key_field}
                        ORDER BY {time_partition_field} DESC
                    ) AS row_num
                FROM `{staging_table_id}` S
            )
            WHERE row_num = 1
        )
        SELECT
            S.*
        FROM deduped_staging S
        LEFT JOIN `{output_table_id}` T
            ON S.{stable_key_field} = T.{stable_key_field}
        WHERE T.{stable_key_field} IS NULL
        """
        new_rows_job = client.query(new_rows_sql, location=gcp_region)
        new_rows_job.result()

        # Ensure the table with requests expires automatically
        new_rows_table = client.get_table(new_rows_table_id)
        new_rows_table.expires = datetime.now() + timedelta(days=table_expiry_days)
        client.update_table(new_rows_table, ["expires"])

        insert_sql = f"""
        INSERT INTO `{output_table_id}`
        SELECT * FROM `{new_rows_table_id}`
        """
        insert_job = client.query(insert_sql, location=gcp_region)
        insert_job.result()

        # Number of rows actually written into the main table (after dedup)
        num_rows_inserted = insert_job.num_dml_affected_rows
        logger.info(
            "write_news_to_bigquery_op: inserted %s new rows into %s after deduplication",
            num_rows_inserted,
            output_table_id,
        )

    except Exception as e:
        logger.error(f"Error in write_news_to_bigquery_op: {str(e)}")
        raise

    return new_rows_table_id


@dsl.component(
    base_image="python:3.12",
    packages_to_install=VERTEX_COMPONENT_PACKAGES,
)
def prepare_text_for_vertex_ai_batch_embedding_op(
    project_id: str,
    region: str,
    input_table_id: str,
    text_field: str,
    output_dataset_id: str = "vertexai_batch_jobs",
    output_table_name_prefix: str = "embedding_pipeline",
    table_expiry_days: int = 7,
) -> str:
    import datetime
    import logging

    from google.cloud import bigquery

    logger = logging.getLogger(__name__)
    logger.info("Starting data prep step for Vertex AI batch embeddings")

    client = bigquery.Client(project=project_id, location=region)

    dataset_ref = bigquery.Dataset(f"{project_id}.{output_dataset_id}")
    dataset_ref.location = region
    _ = client.create_dataset(dataset_ref, exists_ok=True)

    output_table_name = f"{output_table_name_prefix}_input"
    output_table_id = f"{project_id}.{output_dataset_id}.{output_table_name}"

    formatted_query = f"""
    SELECT
        * EXCEPT({text_field}),
        TO_JSON_STRING(
            STRUCT(
              STRUCT(
                ARRAY[
                  STRUCT({text_field} AS text)
                ] AS parts
              ) AS content
            )
          ) AS content
    FROM
        `{input_table_id}`
    """

    create_table_query = f"""
    CREATE OR REPLACE TABLE `{output_table_id}` AS (
        {formatted_query}
    )
    """
    logger.info("Creating Vertex AI request table: %s", output_table_id)
    logger.info("Running Query: %s", formatted_query)
    query_job = client.query(create_table_query, location=region)
    query_job.result()

    table_ref = dataset_ref.table(output_table_name)
    table = client.get_table(table_ref)
    schema_names = [field.name for field in table.schema]
    if "content" not in schema_names:
        raise ValueError("Input table must contain a 'content' column for Vertex AI Batch Prediction")

    table.expires = datetime.datetime.now() + datetime.timedelta(days=table_expiry_days)
    client.update_table(table, ["expires"])
    logger.info("Request table %s expires on %s", output_table_id, table.expires)

    return output_table_id


@dsl.component(
    base_image="python:3.12",
    packages_to_install=VERTEX_COMPONENT_PACKAGES,
)
def run_vertex_ai_batch_embedding_op(
    project_id: str,
    region: str,
    input_table_id: str,
    model: str,
    output_dataset_id: str = "vertexai_batch_jobs",
    output_table_name_prefix: str = "embedding_pipeline",
    table_expiry_days: int = 7,
) -> str:
    import datetime
    import logging
    import time

    from google import genai
    from google.cloud import bigquery
    from google.genai.types import CreateBatchJobConfig, HttpOptions

    logger = logging.getLogger(__name__)
    logger.info("Running Vertex AI batch embedding job")

    bigquery_client = bigquery.Client(project=project_id, location=region)

    output_table_name = f"{output_table_name_prefix}_processed"
    output_table_id = f"{project_id}.{output_dataset_id}.{output_table_name}"

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=region,
        http_options=HttpOptions(api_version="v1"),
    )

    job = client.batches.create(
        model=model,
        src=f"bq://{input_table_id}",
        config=CreateBatchJobConfig(
            display_name=f"Batch Embedding Job - {input_table_id}",
            dest=f"bq://{output_table_id}"
        )
    )
    job_name = job.name
    logger.info(f"Created batch job: {job_name}")

    # Waiting for the job to finish
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'}

    logger.info(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name)  # Initial get
    while batch_job.state.name not in completed_states:
        logger.info(f"Current state: {batch_job.state.name}")
        time.sleep(30)  # Wait for 60 seconds before polling again
        batch_job = client.batches.get(name=job_name)

    logger.info(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == 'JOB_STATE_FAILED':
        logger.error(f"Error: {batch_job.error}")
        raise RuntimeError(batch_job.error)

    # Adding expiry to the table
    table = bigquery_client.get_table(output_table_id)
    table.expires = datetime.datetime.now() + datetime.timedelta(days=table_expiry_days)
    bigquery_client.update_table(table, ["expires"])
    logger.info(f"Table {output_table_id} will expire on {table.expires}")

    return output_table_id


@dsl.component(
    base_image="python:3.12",
    packages_to_install=VERTEX_COMPONENT_PACKAGES,
)
def update_embeddings_table_op(
    project_id: str,
    region: str,
    input_table_id: str,
    output_dataset_id: str = "fin_news_datasets",
    output_table_name: str = "ravenpack_news_embeddings",
    time_partition_field: str = "timestamp",
    stable_key_field: str = "id",
):
    import logging

    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound

    logger = logging.getLogger(__name__)
    logger.info("Updating embeddings destination table from batch job output")

    bigquery_client = bigquery.Client(project=project_id, location=region)
    output_table_id = f"{project_id}.{output_dataset_id}.{output_table_name}"

    # Ensure destination dataset exists (mirrors write_news_to_bigquery_op behaviour)
    dataset_ref = bigquery.Dataset(f"{project_id}.{output_dataset_id}")
    dataset_ref.location = region
    _ = bigquery_client.create_dataset(dataset_ref, exists_ok=True)

    # Get schema of the input (Vertex batch output) table
    input_table = bigquery_client.get_table(input_table_id)
    input_field_names = [field.name for field in input_table.schema]

    # Validate that the time_partition_field exists in the input table schema
    if time_partition_field not in input_field_names:
        raise ValueError(
            f"time_partition_field '{time_partition_field}' not found in input table "
            f"{input_table_id} schema. Available fields: {input_field_names}"
        )

    # Check if the destination embeddings table already exists
    try:
        _ = bigquery_client.get_table(output_table_id)
        logger.info("Destination embeddings table %s already exists", output_table_id)
    except NotFound:
        logger.info(
            "Destination embeddings table %s does not exist yet; creating a partitioned table",
            output_table_id,
        )

        # Build output schema:
        #  - Start from input table schema
        #  - Drop the 'predictions' field
        #  - Add token_count (INT64) and embedding_vector (ARRAY<FLOAT64>)
        output_schema = [
            field for field in input_table.schema if field.name != "predictions"
        ]

        # Append token_count and embedding_vector fields
        output_schema.append(
            bigquery.SchemaField("token_count", "INT64", mode="NULLABLE")
        )
        output_schema.append(
            bigquery.SchemaField("embedding_vector", "FLOAT64", mode="REPEATED")
        )

        output_table = bigquery.Table(output_table_id, schema=output_schema)
        output_table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=time_partition_field,
        )
        output_table.clustering_fields = [stable_key_field]
        _ = bigquery_client.create_table(output_table)
        logger.info("Created embeddings table %s", output_table_id)

    # Insert data into the (possibly newly created) embeddings table
    insert_query = f"""
    INSERT INTO `{output_table_id}`
    SELECT
        * EXCEPT(predictions),
        CAST(
            JSON_VALUE(predictions, '$[0].embeddings.statistics.token_count')
            AS INT64
        ) AS token_count,
        ARRAY(
            SELECT FLOAT64(value)
            FROM UNNEST(JSON_QUERY_ARRAY(predictions, '$[0].embeddings.values')) AS value
        ) AS embedding_vector
    FROM 
        `{input_table_id}`
    """
    logger.info("Inserting embeddings into %s", output_table_id)
    try:
        insert_job = bigquery_client.query(insert_query, location=region)
        insert_job.result()
        logger.info("Embeddings successfully written to %s", output_table_id)
    except Exception as exc:
        logger.error("Error while inserting into embeddings table: %s", exc)
        raise

    # Log the resulting row count
    table_ref = dataset_ref.table(output_table_name)
    table = bigquery_client.get_table(table_ref)
    logger.info("Embeddings table %s now has %s rows", output_table_id, table.num_rows)


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
    text_field: str = "text",
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedding_requests_dataset_id: str = "fin_news_vertexai_batch_jobs",
    embedding_requests_table_prefix: str = "embedding_pipeline",
    embedding_results_table_name: str = "embedding_output",
    embedding_requests_table_expiry_days: int = 7,
):
    # 1. Download RavenPack Data
    download_task = process_ravenpack_data_op(
        gcp_bucket=gcp_bucket,
        gcp_prefix=gcp_prefix,
    )
    download_task.set_display_name("Process Ravenpack Data")
    download_task.set_caching_options(True)

    # 2. Write to BigQuery
    write_task = write_news_to_bigquery_op(
        processed_parquet=download_task.outputs["output_dir"],
        project_id=project_id,
        gcp_region=gcp_region,
        output_dataset_id=output_dataset_id,
        output_table_name=output_table_name,
        time_partition_field=time_partition_field,
        stable_key_field=stable_key_field,
    )
    write_task.set_display_name("Write Ravenpack Data to BigQuery")
    write_task.set_caching_options(True)

    # 3. Prepare RavenPack Data for Vertex AI Batch Embedding job
    prepare_task = prepare_text_for_vertex_ai_batch_embedding_op(
        project_id=project_id,
        region=gcp_region,
        input_table_id=write_task.output,
        text_field=text_field,
        output_dataset_id=embedding_requests_dataset_id,
        output_table_name_prefix=embedding_requests_table_prefix,
        table_expiry_days=embedding_requests_table_expiry_days,
    )
    prepare_task.set_display_name("Prepare Vertex AI Batch Requests")
    prepare_task.set_caching_options(True)

    # 4. Run Vertex AI Batch Embedding job
    vertex_batch_task = run_vertex_ai_batch_embedding_op(
        project_id=project_id,
        region=gcp_region,
        input_table_id=prepare_task.output,
        model=embedding_model_name,
        output_dataset_id=embedding_requests_dataset_id,
        output_table_name_prefix=embedding_requests_table_prefix,
        table_expiry_days=embedding_requests_table_expiry_days,
    )
    vertex_batch_task.set_display_name("Run Vertex AI Batch Embeddings")
    vertex_batch_task.set_caching_options(True)

    # 5. Update embeddings table with results from Vertex AI Batch Embedding job
    update_embeddings_task = update_embeddings_table_op(
        project_id=project_id,
        region=gcp_region,
        input_table_id=vertex_batch_task.output,
        output_dataset_id=output_dataset_id,
        output_table_name=embedding_results_table_name,
        time_partition_field=time_partition_field,
        stable_key_field=stable_key_field,
    )
    update_embeddings_task.set_display_name("Update Embeddings Table")
    update_embeddings_task.set_caching_options(True)
