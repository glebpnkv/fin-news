import argparse
import asyncio
import json
import logging
import os
from datetime import date, datetime
from enum import Enum
from functools import partial

import pandas as pd
from bigdata_client import Bigdata
from dotenv import load_dotenv

from fin_news_ravenpack_data.utils import (
    run_atomic_news_request,
    validate_ravenpack_ids,
)

load_dotenv()

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_SOURCES = [
    "5A5702", # Benzinga
    "9D69F1"  # MT Newswires
]


def json_serializer(obj):
    """
    Serializes an object into a JSON-compatible format.

    This function converts various object types into formats suitable for
    JSON serialization. Specifically, it supports `datetime.date` and
    `datetime.datetime` instances by converting them to ISO 8601 strings,
    `Enum` instances by using the value attribute, and Pydantic models (v1
    and v2) by calling appropriate methods for dictionary representation.

    Parameters
    ----------
    obj : Any
        The object to be serialized. Accepted types include instances of
        `datetime.date`, `datetime.datetime`, `Enum`, or Pydantic models
        (v1 or v2).

    Returns
    -------
    str or dict
        A JSON-compatible representation of the object. For dates and
        times, this is the ISO 8601 string. For enumerations, the value of
        the enumeration. For Pydantic objects, a Python dictionary of the
        model fields is returned.

    Raises
    ------
    TypeError
        If the provided object type is not supported or is not
        serializable.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump()
    raise TypeError(f"Type {type(obj)} not serializable")


async def run_extraction_for_one_company(
    bigdata: Bigdata,
    entity_id: str,
    time_range,
    output_dir_data: str,
    output_dir_stats: str,
    sources: list[str],
    num_documents: int,
    semaphore: asyncio.Semaphore,
    max_consecutive_failures: int = 5
):
    """
    Executes an asynchronous data extraction task for a specific entity (company)
    within the provided time range and writes the results to designated output directories.

    This function manages individual entity data extraction by iterating through time periods,
    handling both successful and failed attempts. It processes data in chunks, allows for
    controlled concurrency using semaphores, and saves results and execution statistics.

    Parameters
    ----------
    bigdata : Bigdata
        Instance used to perform atomic data extraction requests.
    entity_id : str
        Identifier for the company to extract data for.
    time_range : list
        A list of timestamps defining the time intervals to extract data.
    output_dir_data : str
        The directory path to save extracted data for the entity.
    output_dir_stats : str
        The directory path to save statistics related to the data extraction process.
    sources : list of str
        List of data sources to include in the extraction.
    num_documents : int
        Maximum number of documents to retrieve per request.
    semaphore : asyncio.Semaphore
        Semaphore object to limit the global concurrency of extraction tasks.
    max_consecutive_failures : int, optional
        Maximum allowable consecutive failures before aborting extraction for the entity.
        Default is 5.

    Returns
    -------
    None
        This function does not return a value but writes data to specified output files.

    Raises
    ------
    Exception
        If a failure occurs during data extraction or file operations, it will be logged
        and tracked in the statistics file.

    Notes
    -----
    - The function opens the entity-specific data file at the start and maintains it open
      for the duration of processing to improve performance.
    - Successful and failed extraction attempts, along with metadata, are logged into a
      statistics CSV file specific to each entity for auditing and debugging purposes.
    - Concurrency is managed via an asyncio Semaphore object to maintain control over global
      resource usage.
    - The JSON serialization of documents uses a custom serializer to handle nested Enums.
    """
    entity_filename = f"{entity_id}.jsonl"
    entity_filepath = os.path.join(output_dir_data, entity_filename)
    stats_filepath = os.path.join(output_dir_stats, f"{entity_id}.csv")

    logger.info(f"Processing entity: {entity_id}")

    stats = []
    consecutive_failures = 0
    loop = asyncio.get_running_loop()

    # Open file in append mode
    # We keep the file open for the duration of this entity's processing
    with open(entity_filepath, 'a', encoding='utf-8') as f:
        for i in range(len(time_range) - 1):
            if consecutive_failures >= max_consecutive_failures:
                logger.warning(f"Aborting {entity_id} after {consecutive_failures} consecutive failures.")
                break

            period_start = time_range[i]
            period_end = time_range[i + 1]

            record = {
                "entity_id": entity_id,
                "start_time": period_start,
                "end_time": period_end,
                "status": "pending",
                "error": None,
                "doc_count": 0
            }

            try:
                # Use semaphore to limit global concurrency
                async with semaphore:
                    # Run the blocking synchronous API call in a thread pool
                    # This releases the event loop to process other entities
                    func = partial(
                        run_atomic_news_request,
                        bigdata=bigdata,
                        entity_id=entity_id,
                        start_time=period_start.isoformat(),
                        end_time=period_end.isoformat(),
                        sources=sources,
                        num_documents=num_documents
                    )
                    documents = await loop.run_in_executor(None, func)

                # Write documents to file immediately
                for doc in documents:
                    # dump doc.dict() but use default=json_serializer to handle nested Enums
                    f.write(json.dumps(doc.dict(), default=json_serializer) + '\n')

                record["status"] = "success"
                record["doc_count"] = len(documents)
                consecutive_failures = 0  # Reset counter on success
                logger.info(f"{entity_id} [{period_start} - {period_end}]: wrote {record["doc_count"]} documents")

            except Exception as e:
                record["status"] = "failure"
                record["error"] = str(e)
                consecutive_failures += 1
                logger.error(f"{entity_id} [{period_start} - {period_end}]: Failed request: {e}")

            stats.append(record)

    # Save execution statistics
    if stats:
        pd.DataFrame(stats).to_csv(stats_filepath, index=False)


def run_extraction_job(
    bigdata: Bigdata,
    ravenpack_company_ids: list[str],
    start_time: str,
    end_time: str,
    timestep_delta: str,
    output_dir: str,
    ravenpack_source_ids: list[str] | None = None,
    num_documents: int = 10,
    max_concurrency: int = 5,
    max_consecutive_failures: int = 5
):
    """
    Execute an extraction job for multiple companies over a specified time range and timestep intervals.

    The function coordinates the validation of company and source IDs, preparation of output directories,
    time range setup, and orchestration of extraction tasks. It ensures that the extraction is carried out
    with specified concurrency and stability limits.

    Parameters
    ----------
    bigdata : Bigdata
        A reference to the Bigdata object responsible for handling data queries and storage.
    ravenpack_company_ids : list of str
        A list of RavenPack company IDs to extract information for.
    start_time : str
        The start time for the extraction period in ISO 8601 format.
    end_time : str
        The end time for the extraction period in ISO 8601 format.
    timestep_delta : str
        The frequency of extraction time intervals (e.g., '1H', '1D') in Pandas-style string format.
    output_dir : str
        The base directory where the extracted data, metadata, and execution stats will be saved.
    ravenpack_source_ids : list of str or None, optional
        A list of RavenPack source IDs to filter the extraction results, by default None, where defaults
        sources will be used.
    num_documents : int, optional
        The maximum number of documents to fetch per query, by default 10.
    max_concurrency : int, optional
        The maximum number of concurrent jobs that can run, by default 5.
    max_consecutive_failures : int, optional
        The maximum number of consecutive failures allowed for a single company before skipping it,
        by default 5.

    Returns
    -------
    None
        The function does not return any value. The extracted data and execution metadata are saved in
        the specified output directory.

    Notes
    -----
    This is an orchestrating function that runs asynchronous tasks to extract data efficiently.
    Before starting the data extraction, it validates the provided RavenPack IDs for companies and
    sources, ignoring invalid ones.
    """
    # Validating ravenpack_company_ids
    valid_companies_dict, invalid_companies = validate_ravenpack_ids(
        bigdata=bigdata,
        ravenpack_ids=ravenpack_company_ids,
        id_type="company",
        max_concurrency=max_concurrency
    )
    valid_companies = list(valid_companies_dict.keys())
    if len(invalid_companies) > 0:
        logger.info(f"Invalid companies: {invalid_companies}")
    if len(valid_companies) == 0:
        logger.info("All companies were invalid - exiting.")
        return None

    # Validating sources
    if ravenpack_source_ids is None:
        ravenpack_source_ids = DEFAULT_SOURCES
    valid_sources_dict, invalid_sources = validate_ravenpack_ids(
        bigdata=bigdata,
        ravenpack_ids=ravenpack_source_ids,
        id_type="source",
        max_concurrency=max_concurrency
    )
    valid_sources = list(valid_sources_dict.keys())
    if len(invalid_sources) > 0:
        logger.info(f"Invalid sources: {invalid_sources}")
    if len(valid_sources) == 0:
        logger.info("All sources were invalid - exiting.")
        return None

    # Setting up output directories
    output_dir_metadata = os.path.join(output_dir, "metadata")
    output_dir_data = os.path.join(output_dir, "data")
    output_dir_stats = os.path.join(output_dir, "execution_stats")
    os.makedirs(output_dir_metadata, exist_ok=True)
    os.makedirs(output_dir_data, exist_ok=True)
    os.makedirs(output_dir_stats, exist_ok=True)

    # Saving the metadata of valid IDs
    _ = (
        pd.DataFrame(valid_companies_dict.values())
        .to_parquet(os.path.join(output_dir_metadata, "df_valid_companies.parquet"))
    )
    _ = (
        pd.DataFrame(valid_sources_dict.values())
        .to_parquet(os.path.join(output_dir_metadata, "df_valid_sources.parquet"))
    )
    logger.info(f"Saved metadata of valid IDs to {output_dir_metadata}.")

    # Casting start_time and end_time to datetime objects
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    # Creating a time range
    time_range = pd.date_range(start=start_time, end=end_time, freq=timestep_delta)

    # Preparing batches of jobs
    logger.info(f"Starting extraction for {len(valid_companies)} companies over {len(time_range) - 1} time intervals.")

    async def _orchestrate_extraction():
        # Semaphore to control how many entities are queried in parallel
        semaphore = asyncio.Semaphore(max_concurrency)

        tasks = []
        for entity_id in valid_companies:
            tasks.append(
                run_extraction_for_one_company(
                    bigdata=bigdata,
                    entity_id=entity_id,
                    time_range=time_range,
                    output_dir_data=output_dir_data,
                    output_dir_stats=output_dir_stats,
                    sources=valid_sources,
                    num_documents=num_documents,
                    semaphore=semaphore,
                    max_consecutive_failures=max_consecutive_failures
                )
            )

        await asyncio.gather(*tasks)

    # Run the async orchestration
    asyncio.run(_orchestrate_extraction())

    logger.info("Extraction job completed.")
    return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument("--ravenpack-company-ids", nargs="+", type=str, required=True,
                           help="List of RavenPack company IDs")
    argparser.add_argument("--start-time", type=str, required=True,
                           help="Start time for data extraction")
    argparser.add_argument("--end-time", type=str, required=True,
                           help="End time for data extraction")
    argparser.add_argument("--timestep-delta", type=str, required=True,
                           help="Time step for data extraction")
    argparser.add_argument("--output-dir", type=str, required=True,
                           help="Output directory for extracted data")
    argparser.add_argument("--ravenpack-source-ids", nargs="+", type=str, default=None,
                           help="List of RavenPack source IDs")
    argparser.add_argument("--num-documents", type=int, default=10,
                           help="Number of documents to fetch per request")
    argparser.add_argument("--concurrency", type=int, default=10,
                           help="Number of concurrent requests")
    argparser.add_argument("--max-consecutive-failures", type=int, default=5,
                           help="Maximum number of consecutive failures before aborting")
    
    args = argparser.parse_args()
    
    # Load environment variables
    _ = load_dotenv()
    BIGDATA_API_KEY = os.getenv("BIGDATA_API_KEY")
    max_concurrency = int(os.getenv("BIGDATA_MAX_PARALLEL_REQUESTS", args.concurrency))
    bigdata = Bigdata(api_key=BIGDATA_API_KEY)

    run_extraction_job(
        bigdata=bigdata,
        ravenpack_company_ids=args.ravenpack_company_ids,
        start_time=args.start_time,
        end_time=args.end_time,
        timestep_delta=args.timestep_delta,
        output_dir=args.output_dir,
        ravenpack_source_ids=args.ravenpack_source_ids,
        num_documents=args.num_documents,
        max_concurrency=max_concurrency,
        max_consecutive_failures=args.max_consecutive_failures
    )
