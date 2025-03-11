import datetime

import pytz


def get_query_time() -> str:
    """
    Get the current time in a human-readable format.
    """
    query_time = datetime.datetime.now(pytz.timezone("Europe/Zurich"))
    return query_time.strftime("%Y-%m-%d %H:%M:%S")
