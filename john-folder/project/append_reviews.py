import pandas as pd
from pytz import timezone

def review_dt_fix(dt_str):
    # Define the local timezone (e.g., Eastern Time)
    local_tz = timezone('America/New_York')
    
    # Parse the string into a datetime object
    dt = pd.to_datetime(dt_str)

    # Check if datetime is timezone-aware (has valid timezone info)
    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
        # Convert to local timezone
        dt = dt.astimezone(local_tz)

    # Return only the date part
    return dt.date()

def reviews_pipeline(df):
        # fill any null with blank text
        df['review_text'] = df['review_text'].fillna('')
        # Apply the conversion and formatting to the 'publish_time' column.
        df['publish_time'] = df['publish_time'].apply(review_dt_fix) # The output is still a string
        # Convert the 'publish_time' column to datetime, handling ISO8601 format
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        
        return df

def ny_concat_reviews(inspection_row, reviews_df):
    '''
    We use camis to match reviews then we use last_inspection_date and inspection_date to filter reviews that fall within the date ranges. Finally we concatenate the review_text and return a string.  
    '''
    # Get the camis
    camis = inspection_row['camis']
    
    # If last_inspection_date is not null, we add one day to it to get the start date
    start_date = inspection_row['last_inspection_date'] + pd.Timedelta(days=1) if pd.notnull(inspection_row['last_inspection_date']) else inspection_row['inspection_date']
    end_date = inspection_row['inspection_date']
    
    # Filter reviews that match the camis and fall within the date range
    matching_reviews = reviews_df[
        (reviews_df['camis'] == camis) &
        (reviews_df['publish_time'] >= start_date) &
        (reviews_df['publish_time'] <= end_date)
    ]
    
    # Concatenate the review texts
    concatenated_reviews = ' '.join(matching_reviews['review_text'].dropna())
    
    return concatenated_reviews

def ny_last_inspection(df):
    # Make inspection_date into a datetime
    df['inspection_date'] = pd.to_datetime(df['inspection_date'])
    
    # Sort the dummy dataframe by 'camis' and 'inspection_date'
    df_sorted = df.sort_values(['camis', 'inspection_date'])

    # Create the 'last_inspection_date' by shiting the inspection date 1
    df_sorted['last_inspection_date'] = df_sorted.groupby('camis')['inspection_date'].shift(1)

    # Fill NaT in 'last_inspection_date' with 'inspection_date' - 1 year
    df_sorted['last_inspection_date'] = df_sorted['last_inspection_date'].fillna(df_sorted['inspection_date'] - pd.DateOffset(years=1))

    return df_sorted

def ny_pipeline(inspections_df, reviews_df):
    inspections_df = ny_last_inspection(inspections_df)
    # Apply the function to each row in the inspections dataframe
    inspections_df['concatenated_reviews'] = inspections_df.apply(lambda row: ny_concat_reviews(row, reviews_df), axis=1)
    return inspections_df