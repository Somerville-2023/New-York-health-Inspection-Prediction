import os
import pandas as pd
from env import key_token


def acquire_ny():
    """This function will acquire the ny data either from a local copy of a .csv file or via api call if file does
    not exist."""
    filename = 'ny.csv'  # File name
    if os.path.isfile(filename):  # Checks for local file
        return pd.read_csv(filename)  # Returns local file if there is one
    else:
        from sodapy import Socrata
        # Create client
        # NOTE: YOU NEED A KEY TOKEN
        client = Socrata("data.cityofnewyork.us", key_token)
        # Make request
        results = client.get("43nn-pn8j", limit=500_000)
        # Convert to pandas DataFrame
        results_df = pd.DataFrame.from_records(results)
        results_df.to_csv(filename, index=False)  # Cache file
    return results_df  # Return file
