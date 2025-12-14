import logging
import os
from typing import List, Optional
from kfp import dsl
from kfp.dsl import Output, Dataset

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


@dsl.component(base_image=BASE_IMAGE)
def download_ravenpack_data(
    gcp_bucket: str,
    gcp_prefix: str,
    gcp_region: str,
    bigdata_secret_name: str,
    ravenpack_company_ids: List[str],
    start_time: str,
    end_time: str,
    timestep_delta: str,
    ravenpack_source_ids: Optional[List[str]] = None,
    num_documents: int = 10,
    max_concurrency: int = 5,
    max_consecutive_failures: int = 5,
    output_dir: Output[Dataset] = Output[Dataset],
):
    import os
    import logging
    from bigdata_client import Bigdata
    from google.cloud import secretmanager

    from finnews.utils.gcp import upload_dir_to_gcp
    from fin_news_ravenpack_data.extract import run_extraction_job

    # Configure Cloud Logging
    logger = logging.getLogger(__name__)
    logger.info("Starting 'download_ravenpack_data' step")

    # Prepare output directory
    os.makedirs(output_dir.path, exist_ok=True)

    client = secretmanager.SecretManagerServiceClient()
    # Access secret value
    response = client.access_secret_version(request={"name": bigdata_secret_name})
    BIGDATA_API_KEY = response.payload.data.decode("utf-8")

    bigdata = Bigdata(api_key=BIGDATA_API_KEY)

    logging.info("Downloading RavenPack data")
    run_extraction_job(
        bigdata=bigdata,
        ravenpack_company_ids=ravenpack_company_ids,
        start_time=start_time,
        end_time=end_time,
        timestep_delta=timestep_delta,
        output_dir=output_dir.path,
        ravenpack_source_ids=ravenpack_source_ids,
        num_documents=num_documents,
        max_concurrency=max_concurrency,
        max_consecutive_failures=max_consecutive_failures
    )

    # 5. Upload directory with extracted Ravenpack data to GCS
    # Layout: gs://<bucket>/<prefix>/...
    logger.info(
        f"Uploading directory with extracted Ravenpack data to "
        f"gs://{gcp_bucket}/{gcp_prefix}"
    )
    upload_dir_to_gcp(
        bucket_name=gcp_bucket,
        local_dir=output_dir.path + "/",
        dst_prefix=f"{gcp_prefix}",
        region=gcp_region,
    )

    logger.info("'download_ravenpack_data' step completed successfully.")


@dsl.pipeline(
    name="ravenpack-data-extract-pipeline",
    description="Extracts Ravenpack BigQuery data and saves to GCS"
)
def ravenpack_data_extract_pipeline(
    gcp_bucket: str,
    gcp_prefix: str,
    bigdata_secret_name: str,
    ravenpack_company_ids: List[str],
    start_time: str,
    end_time: str,
    timestep_delta: str,
    ravenpack_source_ids: Optional[List[str]] = None,
    num_documents: int = 10,
    max_concurrency: int = 5,
    max_consecutive_failures: int = 5,
    gcp_region: str = "us-central1",
):
    # 1. Download Ravenpack Data
    download_task = download_ravenpack_data(
        gcp_bucket=gcp_bucket,
        gcp_prefix=gcp_prefix,
        gcp_region=gcp_region,
        bigdata_secret_name=bigdata_secret_name,
        ravenpack_company_ids=ravenpack_company_ids,
        start_time=start_time,
        end_time=end_time,
        timestep_delta=timestep_delta,
        ravenpack_source_ids=ravenpack_source_ids,
        num_documents=num_documents,
        max_concurrency=max_concurrency,
        max_consecutive_failures=max_consecutive_failures,
    )
    download_task.set_display_name("Download Ravenpack Data")
