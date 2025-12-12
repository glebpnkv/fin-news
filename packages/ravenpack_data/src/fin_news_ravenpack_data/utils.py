import concurrent.futures
import logging
import operator
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import date, datetime
from enum import Enum
from functools import reduce
from typing import Literal

from bigdata_client import Bigdata
from bigdata_client.daterange import AbsoluteDateRange
from bigdata_client.document import Document
from bigdata_client.models.search import DocumentType
from bigdata_client.query import Entity, Source

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def validate_ravenpack_ids(
    bigdata: Bigdata,
    ravenpack_ids: list[str],
    id_type: Literal["source", "company"],
    max_concurrency: int = 10
) -> tuple[dict, list[str]]:
    """
    Validates a list of RavenPack IDs by using the provided Bigdata client to query the
    knowledge graph for each ID, determining validity and obtaining additional data.

    IDs that cannot be found or cause errors during querying are recorded separately as
    invalid IDs.

    :param bigdata: A Bigdata client instance used to interact with the knowledge graph.
    :type bigdata: Bigdata
    :param ravenpack_ids: A list of RavenPack ID identifiers to be validated.
    :param id_type: The type of ID to validate ("source" or "company").
    :param max_concurrency: Maximum number of concurrent requests to make during validation.
    :return: A tuple containing two elements:
        1. A dictionary of valid IDs where the keys are the entity identifiers and
           the values are the details of the first matched entity in the knowledge graph.
        2. A list of invalid IDs that could not be validated or caused querying errors.
    :rtype: tuple[dict, list[str]]
    """
    valid_ids = {}
    invalid_ids = []

    method_map = {
        "source": "find_sources",
        "company": "find_companies"
    }

    # Map for logging labels
    label_map = {
        "source": "Source",
        "company": "Entity"
    }

    if id_type not in method_map:
        raise ValueError(f"Invalid id_type '{id_type}'. Expected one of {list(method_map.keys())}")

    method_name = method_map[id_type]
    label = label_map[id_type]

    # Dynamically retrieve the search method
    try:
        search_method = getattr(bigdata.knowledge_graph, method_name)
    except AttributeError as e:
        # We should not proceed if the method is not found
        raise AttributeError(f"Method '{method_name}' not found: {e}")

    def _query_method(id_input):
        out = []
        try:
            out = search_method(id_input)
        except Exception as e:
            logger.error(f"Error fetching {label.lower()} '{id_input}': {e}")
        return out


    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures_to_values = {
            # executor.submit(_query_method, bigdata, cur_id): cur_id
            executor.submit(_query_method, cur_id): cur_id
            for cur_id in ravenpack_ids
        }

        # Unpacking results
        for future in concurrent.futures.as_completed(futures_to_values):
            cur_id = futures_to_values[future]
            cur_result = future.result()
            if len(cur_result) > 0:
                valid_ids[cur_id] = cur_result[0].dict()
            else:
                invalid_ids.append(cur_id)

    return valid_ids, invalid_ids


def run_atomic_news_request(
    bigdata: Bigdata,
    entity_id: str,
    start_time: str,
    end_time: str,
    sources: list[str],
    num_documents: int = 10
) -> list[Document]:
    # Creating a date range
    date_range = AbsoluteDateRange(start_time, end_time)

    entity = Entity(entity_id)
    query = entity

    # Adding sources
    query_sources = reduce(operator.or_, [Source(x) for x in sources])
    query &= query_sources

    search = bigdata.search.new(
        query=query,
        date_range=date_range,
        scope=DocumentType.NEWS
    )

    documents = search.run(limit=num_documents)

    return documents