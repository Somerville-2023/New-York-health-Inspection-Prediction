import pandas as pd

def load_csv(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath, index_col=False)

def drop_cols(df, columns):
    """Drop specified columns from the DataFrame."""
    df.drop(columns, axis=1, inplace=True)
    return df

def drop_nulls(df, columns):
    """Drop rows with nulls in specified columns."""
    df.dropna(subset=columns, inplace=True)
    return df

def drop_upcoming(df):
    # Filter out rows with 'inspection_date' as '1900-01-01'
    df = df[df['inspection_date'] != '1900-01-01']

    # Reset the index to ensure continuous index values
    df.reset_index(drop=True, inplace=True)
    
    return df

def filter_types(df, types):
    """Filter out specific inspection types."""
    df = df[~df['inspection_type'].str.startswith(tuple(types))]
    return df

def infer_violations(df):
    """Handle specific conditions for violation fields."""
    conditions = [
        (df['violation_code'].isna() & df['action'].str.startswith("No violations were recorded at the time of this inspection.")),
        (df['violation_code'].isna() & df['action'].str.startswith("Establishment re-opened") & (df['critical_flag'] == 'Not Applicable'))
    ]
    for condition in conditions:
        df.loc[condition, ['violation_code', 'violation_description']] = ['none', 'No violations were recorded']
    return df

def remove_violations(df):
    """Remove rows with certain violation conditions."""
    condition = (df['violation_code'].isna()) & (df['action'].str.startswith("Violations were cited in the following area(s)"))
    df.drop(df[condition].index, inplace=True)
    return df

def format_phone(df):
    """Format the 'phone' column."""
    df['phone'].fillna('1000000000', inplace=True)
    df['phone'] = df['phone'].str.replace(r'\D', '', regex=True)
    df['phone'] = df['phone'].str.strip().replace(['', '0000000000'], '1000000000')
    return df

def convert_cols(df, columns):
    """Convert specified columns to int then string."""
    for column in columns:
        df[column] = df[column].astype(int).astype(str)
    return df

def format_date(df):
    """Format 'inspection_date'."""
    df['inspection_date'] = pd.to_datetime(df['inspection_date']).dt.strftime('%Y-%m-%d')
    return df

def set_score(df):
    # Find the maximum score per inspection
    max_scores = df.groupby(['camis', 'inspection_date'])['score'].transform('max')

    # Update the 'score' column with the maximum score for each inspection
    df['score'] = max_scores
    
    return df



def prepare_data(filepath):
    """Prepare data using the pipeline."""
    df = load_csv(filepath)
    df = drop_cols(df, ['grade', 'grade_date'])
    df = drop_nulls(df, ['bin', 'council_district'])
    df = drop_upcoming(df)
    df = filter_types(df, ["Calorie Posting", "Pre-permit", "Smoke-Free Air Act", "Trans Fat", "Administrative"])
    df = infer_violations(df)
    df = remove_violations(df)
    df['score'] = df['score'].astype(int)  # Convert 'score' to int
    df = format_phone(df)
    df = convert_cols(df, ['zipcode', 'community_board', 'council_district', 'census_tract', 'bin', 'bbl'])
    df = format_date(df)
    df = set_score(df)
    df.drop_duplicates(inplace=True)
    return df