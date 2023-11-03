import pandas as pd
import os
import env

# ---------------------------------------------------------------------------------------------------------------------
# NY csv file acquisition function


def acquire_ny():
    """This function acquires the ny health inspection data. It will look for a local copy and read it but if it
    cannot find a local copy then it will perform and api request to acquire it."""
    filename = 'ny.csv'  # File name
    if os.path.isfile(filename):  # Checks for local file
        return pd.read_csv(filename)  # Returns local file if there is one
    else:
        from sodapy import Socrata
        # Create client
        # NOTE: YOU NEED A KEY TOKEN
        client = Socrata("data.cityofnewyork.us", env.key_token)
        # Make request
        results = client.get("43nn-pn8j", limit=500_000)
        # Convert to pandas DataFrame
        results_df = pd.DataFrame.from_records(results)
        results_df.to_csv(filename, index=False)  # Cache file
    return results_df  # Return file
