import argparse
import logging
import os

from bigdata_client import Bigdata
from dotenv import load_dotenv

from fin_news_ravenpack_data.extract import run_extraction_job

load_dotenv()

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
