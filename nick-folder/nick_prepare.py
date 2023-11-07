import pandas as pd
import numpy as np
import os

import re
import nick_acquire as a

# NLP imports
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import unicodedata

from datetime import timedelta
from pytz import timezone

# nltk.download('wordnet')
# nltk.download('omw-1.4')

# ---------------------------------------------------------------------------------------------------------------------
# NY data prep functions


def remove_columns(ny, trash_columns=['bin', 'bbl', 'nta', 'census_tract', 'council_district', 'community_board',
                                      'grade_date', 'critical_flag', 'inspection_type', 'record_date']):
    """This function just removes the columns passed into from the dataframe"""
    ny = ny.drop(columns=trash_columns)
    return ny


def clean_inspection_dates(data):
    ny = data.copy()
    ny = ny[ny.inspection_date != '1900-01-01T00:00:00.000']  # Remove all values with no inspections done
    return ny


def clean_phones(data):
    """This function cleans up all the phone numbers from the dataframe."""
    ny = data.copy()  # Create copy of dataframe

    ny = ny[ny.phone.notna()]  # Drop all nulls

    new_phone = []  # Create empty list to append new, clean phone numbers into.

    for phone in ny.phone:  # Iterate through all phone numbers
        new_phone.append(re.sub(r'\D', '', phone))  # Remove all non-digit characters
    ny.phone = new_phone  # Replace series with new values

    # Create new list with 0s for missing phones
    newer_phones = [phone if len(phone) > 1 else '0' for phone in ny.phone]

    ny.phone = newer_phones  # Replace series with new values

    ny['phone'] = pd.to_numeric(ny['phone'], errors='coerce')  # Convert to numeric

    ny['phone'] = ny['phone'].astype(int)  # Convert it to an integer

    return ny  # Return df


def clean_zipcodes(data):
    """This function cleans up all zip codes. It fills in the nulls with 0s and then converts everything to int."""
    ny = data.copy()
    # Clean zipcodes by filling nulls with 0 and then converting to integers
    ny = ny[ny.zipcode.notna()]
    ny.zipcode = ny.zipcode.astype(str).apply(lambda x: re.findall(r'\d+', x)[0])
    ny.zipcode = ny.zipcode.fillna(0)
    ny.zipcode = ny.zipcode.astype(int)
    return ny  # Return df


def clean_streets(ny):
    """Remove nulls from street"""
    ny = ny[ny.street.notna()]
    return ny  # Return df


def clean_scores(data):
    """This function cleans up the scores from the dataframe."""
    ny = data.copy()
    # THE SECTION OF CODE BELOW IS COMMENTED OUT BECAUSE IT PROVED TO NOT BE USEFUL AT THE MOMENT
    # Create a new list of scores that replaces null scores for no violation for 0s
    # new_scores = []  # Empty list
    # for score, rep in zip(ny.score, ny.action.str.contains('No violation')):  # Loop through 2 iterable values
    #     if rep:  # If no violation, append score 0
    #         new_scores.append(0)
    #     else:  # Else keep score the same
    #         new_scores.append(score)
    # ny.score = new_scores

    ny = ny[ny.score.notna()]  # Drop all nulls

    ny.score = ny.score.astype(int)  # Convert data to integer

    return ny  # Return df


def clean_actions(ny):
    """This function cleans up the action column. It relabels the values with a more concise one."""
    # Remove nulls from action
    ny = ny[ny.action.notna()]
    # Rename actions to something more concise
    ny.action = np.where(ny.action == 'Violations were cited in the following area(s).', 'Violations cited', ny.action)
    ny.action = np.where(ny.action == 'Establishment Closed by DOHMH. Violations were cited in the following area(s) '
                                      'and those requiring immediate action were addressed.', 'Closed', ny.action)
    ny.action = np.where(ny.action == 'Establishment re-opened by DOHMH.', 'Re-opened', ny.action)
    ny.action = np.where(ny.action == 'No violations were recorded at the time of this inspection.', 'No violations',
                         ny.action)
    return ny  # Return df


def clean_grades(data):
    """This function cleans up the grades by redetermining them based off the score."""
    ny = data.copy()  # Create copy of df
    # Create empty list to hold new values for restaurant
    new_grades = []
    # Use scores to determine grades
    for grade, score in zip(ny.grade, ny.score):
        if score <= 13:
            new_grades.append('A')
        elif score <= 27:
            new_grades.append('B')
        elif score > 27:
            new_grades.append('C')
    ny.grade = new_grades  # Update grade column
    return ny  # Return df


def clean_violations(data):
    """This function cleans up the violation codes. If there is no violations under actions, the code and description
        will be filled with 'No violation' instead of being null."""
    ny = data.copy()
    # Create empty lists
    new_codes = []
    new_description = []
    # Loop through actions and violations
    for action, code, description in zip(ny.action, ny.violation_code, ny.violation_description):
        if action == 'No violations':  # If there is no violations, append no violations to code and description
            new_codes.append('No violation')
            new_description.append('No violation')
        else:
            new_codes.append(code)
            new_description.append(description)

    # Replace df values with new ones
    ny.violation_code = new_codes
    ny.violation_description = new_description

    return ny  # Return data


def combine_address(ny):
    """This function combines the addresses of the restaurants into one single feature."""
    full_addy = ny.building + ' ' + ny.street + ' ' + ny.zipcode.astype(str)  # Concat the address together
    ny['full_address'] = full_addy  # Create new feature
    ny = ny.drop(columns=['building', 'street', 'zipcode'])  # Drop old features
    return ny  # Return df


def clean_ny(ny):
    """This function just takes in all other cleaning functions for ny data and cleans each element of it"""

    ny = remove_columns(ny)  # Removes useless columns from ny health inspection data

    ny = clean_phones(ny)  # Clean phone numbers

    ny = clean_zipcodes(ny)  # Cleans zip codes

    ny = clean_streets(ny)  # Cleans streets

    ny = clean_inspection_dates(ny)

    ny = clean_scores(ny)  # Cleans scores

    ny = clean_actions(ny)  # Cleans actions

    ny = clean_grades(ny)  # Cleans grades

    ny = clean_violations(ny)  # Cleans violation codes and descriptions

    ny = ny.dropna()  # Drops all remaining null values

    ny = combine_address(ny)  # Combine the address related columns into one

    ny = ny.reset_index(drop=True)  # Reset the index of dataframe after dropping all the other values

    return ny  # Return clean dataframe


def aggregate_violations(ny):
    """This function will aggregate all rows for each inspection for each restaurant into on row by combining the
       violations."""
    # Create aggregated df indexed by camis and inspection_date
    agg_violations = ny.groupby(['camis', 'inspection_date']).agg({'violation_code': lambda x: x.tolist(),
                                                                   'violation_description': lambda x: x.tolist()})
    # Create separate df without code & description
    ny2 = ny.drop(columns=['violation_code', 'violation_description']).copy()
    ny2 = ny2.drop_duplicates()  # Drop duplicates

    # Create empty lists
    agg_data_code = []
    agg_data_description = []

    # Loop through df without duplicates and create lists of aggregated violations
    for cam, date in zip(ny2.camis, ny2.inspection_date):
        agg_data_code.append(agg_violations.loc[(cam, date)][0])
        agg_data_description.append(agg_violations.loc[(cam, date)][1])

    # Insert new, aggregated violations into df
    ny2['violation_code'] = agg_data_code
    ny2['violation_description'] = agg_data_description

    return ny2  # Return df


def clean_code(ny):
    """This function removes 'No violation' from the rows that shouldn't have it. Some rows contained both violation
    codes and 'No violation'."""
    # Create empty lists
    clean_codes = []
    clean_description = []

    # Loop through lists and remove 'No violation' if there are more than one element in each list
    for row1, row2 in zip(ny.violation_code, ny.violation_description):

        code_list1 = row1
        code_list2 = row2

        if len(code_list1) > 1 and 'No violation' in code_list1:
            code_list1.remove('No violation')
            clean_codes.append(code_list1)
        else:
            clean_codes.append(code_list1)

        if len(code_list2) > 1 and 'No violation' in code_list2:
            code_list2.remove('No violation')
            clean_description.append(code_list2)
        else:
            clean_description.append(code_list2)

    # Reassign new data to dataframe
    ny.violation_code = clean_codes
    ny.violation_description = clean_description

    return ny  # Return df


def join_lists(ny):
    """This function joins all the contents of the lists in code, and description into one string."""

    # Create empty lists
    joined_code = []
    joined_description = []

    # Join violation codes with a ' ' between elements
    for row in ny.violation_code:
        joined_code.append(' '.join(row))

    # Join violation description with a ' ' between elements
    for row in ny.violation_description:
        joined_description.append(' '.join(row))

    ny.violation_code = joined_code
    ny.violation_description = joined_description

    return ny  # Return df


def final_ny():
    """This function just combines all the previous functions into one. It will acquire and process the data."""
    filename = 'clean_ny.csv'  # File name
    if os.path.isfile(filename):  # Checks for local file
        return pd.read_csv(filename)  # Returns local file if there is one
    else:
        ny = a.acquire_ny()  # Acquire data, from local .csv file or api request if no .csv file is present

        ny = clean_ny(ny)  # Cleans the data

        # Aggregates the data into one row per inspection and compiles the violation data into a list per row
        ny = aggregate_violations(ny)

        ny = clean_code(ny)  # Removes 'No violation' from lists it shouldn't be in

        ny = join_lists(ny)  # Unpacks (combines) lists into one string

        ny.to_csv(filename, index=False)  # Cache file

    return ny  # Return df

# --------------------------------------------------------------------------------------------------------

# NLP Functions


def basic_clean(filthy_data):
    """This function takes in a string and makes everything lowercase, unicodes the data, and removes all
    non-alphanumeric charachters and non-space characters from the string and returns it."""
    filthy_data = filthy_data.lower()  # Convert to lowercase
    # Remove non ASCII characters
    filthy_data = unicodedata.normalize('NFKD', filthy_data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    clean_data = re.sub(r"[^a-z0-9'\s]", "", filthy_data)  # Remove non-alphanumeric and non-space characters
    return clean_data  # Return data


def tokenize(data):
    """This function simply tokenizes the data."""
    tokenizer = ToktokTokenizer()
    data = tokenizer.tokenize(data, return_str=True)
    return data


def remove_stopwords(string, extra_words=[], exclude_words=[]):
    """ This function takes in a string, optional extra_words and exclued_words parameters with default empty lists
     and returns a string """
    stopword_list = stopwords.words('english')
    # use set casting to remove any excluded stopwords
    stopword_set = set(stopword_list) - set(exclude_words)
    # add in extra words to stopwords set using a union
    stopword_set = stopword_set.union(set(extra_words))
    # split the document by spaces
    words = string.split()
    # every word in our document that is not a stopword
    filtered_words = [word for word in words if word not in stopword_set]
    # join it back together with spaces
    string_without_stopwords = ' '.join(filtered_words)
    return string_without_stopwords


def lemmatize(data):
    """This function lemmatizes a body of text and returns it."""
    wnl = nltk.stem.WordNetLemmatizer()  # Create lemmatizer object
    lemmas = [wnl.lemmatize(word) for word in data.split()]  # Use list comprehension to create list of lemmatized words
    lemmatized_data = ' '.join(lemmas)  # Rejoin words
    return lemmatized_data  # Return data


def stem(data):
    """This function stems a body of text and returns it."""
    ps = nltk.porter.PorterStemmer()  # Create stem object
    stems = [ps.stem(word) for word in data.split()]  # Use list comprehension to create list of stemmed words
    stemmed_data = ' '.join(stems)  # Rejoin words
    return stemmed_data  # Return data


def cleanse(dataframe, col='', stemm=True, lem=True, extra_words=[], exclude_words=[]):
    """This function takes in a dataframe and a column name and will create 1-3 extra columns with clean, stemmed,
    and lemmatized data. It will also remove all stopwords."""
    df = dataframe.copy()  # Create copy of df
    df['clean'] = df[col].apply(basic_clean)  # Create column containing clean text
    # Remove stopwords.
    df['clean'] = df['clean'].apply(remove_stopwords, extra_words=extra_words, exclude_words=exclude_words)
    if stemm:
        df['stemmed'] = df.clean.apply(stem)  # Creates stemmed data column
    if lem:
        df['lemmatized'] = df.clean.apply(lemmatize)  # Creates lemmatized data column
    return df  # Return df


# --------------------------------------------------------------------------------------------------------

# Google review data functions


def clean_api_reviews(api_data):
    """This function cleans the api review data"""
    df = api_data.copy()  # Create copy of data
    cols = ['camis', 'publish_time', 'review_text', 'review_rating']  # Select columns to keep, and order them
    df = df[cols]  # Filter columns
    df.publish_time = pd.to_datetime(df.publish_time)  # Convert publish_time to pd.datetime object
    return df  # Return df


def clean_dates(data):
    """This function takes in the scraped reviews and cleans up the relative date columns, so it is ready to be used"""
    scrape_reviews = data.copy()  # Create copy of df
    scrape_reviews.relative_date = scrape_reviews.relative_date.apply(lambda x: x[:-4])  # Removes ' ago' from dates
    # Changes dates with 'a xxxxx' to '1 xxxxx'
    scrape_reviews.relative_date = [re.sub(r'^a', '1', date) if date[0] == 'a' else date
                                    for date in scrape_reviews.relative_date]
    return scrape_reviews  # Return df


def adjust_dates(scrape_reviews):
    """This function adjusts the dates. It will estimate the actual publish date of a review based off how many
    reviews the restaurant has for a certain year. Example: If a restaurant has 365 reviews in a year it will assume
    an average of 1 review a day and adjsut all dates that say 'x years' accordingly"""
    dataframes = []  # Create empty list to store dataframes

    # Isolate each restaurant by id
    for restaurant_id in scrape_reviews.id.unique():
        # Create dataframe of ONE restaurant
        restaurant = scrape_reviews[scrape_reviews.id == restaurant_id].copy()

        # Create df of review counts per relative_date and calculate average distribution of reviews
        place = scrape_reviews[scrape_reviews.id == restaurant_id]
        review_counts = pd.DataFrame(place.relative_date.value_counts())
        review_counts = review_counts.rename(columns={review_counts.columns[0]: 'relative_date'})
        review_counts['increment'] = 365 / review_counts.relative_date

        # Create empty list for new dates, i variable to count increments, and previous_year to track year
        new_dates = []
        i = 0
        previous_year = '1 years'

        for date in restaurant.relative_date:
            if 'year' in date:  # If date is in years, function will adjust it to estimated date
                if date != previous_year:  # When date changes from 'x years' to 'x + 1 years' counters are reset
                    i = 0
                    previous_year = date
                # Calculate adjusted date
                adjusted_date = (365 * (int(re.findall(r'\d+', date)[0]))) + (review_counts.loc[date].increment * i)
                i += 1
                new_dates.append(str(round(adjusted_date)))  # Append adjsuted date
            else:
                new_dates.append(date)  # Append normal date if date < 1 year
        restaurant['new_date'] = new_dates  # Replace dates with new_dates
        dataframes.append(restaurant)  # Append dataframe to list of dataframes
    reviews = pd.concat(dataframes)  # Join all dataframes
    return reviews  # Return joined data


def calculate_days(data):
    """This function will convert all times into day equivalent."""
    reviews = data.copy()  # Create copy of df
    new_date = []  # Create empty list
    for date in reviews.new_date:
        unit = re.sub(r'[^a-z]', '', date)  # Indentifies unit of time (hours, days, weeks, etc...)
        if 'second' in unit:
            new_date.append('1')  # Any amount of hours will be converted to one day ago
        elif 'minute' in unit:
            new_date.append('1')  # Any amount of hours will be converted to one day ago
        elif 'hour' in unit:
            new_date.append('1')  # Any amount of hours will be converted to one day ago
        elif 'day' in unit:
            new_date.append(re.sub(r'[^0-9]', '', date))  # Appends number of days to list
        elif 'week' in unit:
            new_date.append(int(re.sub(r'[^0-9]', '', date))*7)  # Appends number of weeks * 7 to list
        elif 'month' in unit:
            new_date.append(int(re.sub(r'[^0-9]', '', date))*30)  # Appends number of months * 30 to list
        else:
            new_date.append(date)  # If date unit is not found, append date

    reviews['newer_dates'] = new_date  # Creates new date column
    # Calculates estimated review date based off retrieval time and adjusted relative time
    reviews['final_date'] = [pd.to_datetime(retrieval_date) - timedelta(days=n) for retrieval_date, n
                             in zip(reviews.retrieval_date, reviews.newer_dates.astype(int))]
    return reviews  # Return df


def clean_additional(data):
    df = data.copy()
    df['additional'] = df.r_additional.apply(lambda x: re.findall(r'\'([^\']+)?\'', x))
    important_additionals = \
        ['Service: 5', 'Service: 4', 'Service: 3', 'Service: 2', 'Service: 1',
         'Atmosphere: 5', 'Atmosphere: 4', 'Atmosphere: 3', 'Atmosphere: 2', 'Atmosphere: 1',
         'Food: 5',  'Food: 4',  'Food: 3',  'Food: 2',  'Food: 1', 'Price per person $10–20',
         'Price per person $20–30', 'Price per person $30–50', 'Price per person $50–100', 'Price per person $100+']
    df['additional'] = df.additional.apply(lambda x: [ele for ele in x if ele in important_additionals])
    df['additional'] = df.additional.apply(lambda x: ' '.join(x))
    return df


def create_additional_columns(df):
    clean = df.copy()
    clean['service'] = \
        [int(re.findall(r'Service: (\d)', x)[0]) if re.search(r'Service: (\d)', x) else None for x in clean.additional]

    clean['atmosphere'] = \
        [int(re.findall(r'Atmosphere: (\d)', x)[0]) if re.search(r'Atmosphere: (\d)', x) else None for x in
         clean.additional]

    clean['food'] = \
        [int(re.findall(r'Food: (\d)', x)[0]) if re.search(r'Food: (\d)', x) else None for x in clean.additional]

    clean['price_per_person'] = \
        [re.findall(r'\$(\S+)', x)[0] if re.search(r'\$(\S+)', x) else None for x in clean.additional]

    return clean


def adjust_prices(data):
    df = data.copy()
    new_prices = []

    for price in df.price_per_person:
        if price == '10–20':
            new_prices.append(15)
        elif price == '20–30':
            new_prices.append(25)
        elif price == '30–50':
            new_prices.append(40)
        elif price == '50–100':
            new_prices.append(75)
        elif price == '100+':
            new_prices.append(100)
        else:
            new_prices.append(None)
    df.price_per_person = new_prices
    return df


def clean_reviews(data):
    """This function will polish off the scraped reviews dataframe."""
    final_df = data.copy()  # Create copy of df
    # Columns to keep
    cols = ['id', 'final_date', 'caption', 'rating', 'service', 'atmosphere', 'food', 'price_per_person']
    final_df = final_df[cols]  # Reassign columns
    final_df.rating = final_df.rating.astype(int)  # Change ratings to integers
    final_df.columns = ['camis', 'publish_time', 'review_text', 'review_rating', 'service', 'atmosphere', 'food',
                        'price_per_person']  # Rename columns
    return final_df  # Return df


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
    df['publish_time'] = df['publish_time'].apply(review_dt_fix)  # The output is still a string
    # Convert the 'publish_time' column to datetime, handling ISO8601 format
    df['publish_time'] = pd.to_datetime(df['publish_time'])

    return df


def cleanse_reviews(scraped_data, api_data=None):
    """This function combines each step into one function to yield the final product in one function call."""
    df = scraped_data.copy()  # Creates copy of scraped df
    df = clean_dates(df)  # Clean dates
    df = adjust_dates(df)  # Adjust dates
    df = calculate_days(df)  # Calculate estimated publish time
    df = clean_additional(df)
    df = create_additional_columns(df)
    df = adjust_prices(df)
    df = clean_reviews(df)  # Finish cleaning df
    if api_data is not None:
        df2 = api_data.copy()  # Creates copy of api df
        df2 = clean_api_reviews(df2)  # Clean the api df
        df = pd.concat([df, df2])  # Return the two joined df
    df = reviews_pipeline(df)  # Adjust dates
    return df  # Return the two joined df


def ny_concat_reviews(inspection_row, reviews_df):
    """We use camis to match reviews then we use last_inspection_date and inspection_date to filter reviews that fall
    within the date ranges. Finally, we concatenate the review_text and return a string."""
    # Get the camis
    camis = inspection_row['camis']

    # If last_inspection_date is not null, we add one day to it to get the start date
    start_date = inspection_row['last_inspection_date'] + pd.Timedelta(days=1) if pd.notnull(
        inspection_row['last_inspection_date']) else inspection_row['inspection_date']
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
    df_sorted['last_inspection_date'] = df_sorted['last_inspection_date'].fillna(
        df_sorted['inspection_date'] - pd.DateOffset(years=1))

    return df_sorted


def ny_pipeline(inspections_df, reviews_df):
    inspections_df = ny_last_inspection(inspections_df)
    # Apply the function to each row in the inspections dataframe
    inspections_df['concatenated_reviews'] = inspections_df.apply(lambda row: ny_concat_reviews(row, reviews_df),
                                                                  axis=1)
    return inspections_df


def acquire_ny_reviews(include_api=False):
    """This function combines the entire pipeline into one function. It will look for ny_reviews, which is the df
    of the combined clean ny inspection data and review data, if that isn't found it will then begin creating that df"""
    if os.path.isfile('ny_reviews.csv'):  # Checks for local file
        print('ny_reviews.csv found!')
        return pd.read_csv('ny_reviews.csv')  # Returns local file if there is one

    if os.path.isfile('clean_ny.csv'):  # Checks for local file
        ny = final_ny()  # If file is found, it is read and assigned to ny variable
        print('clean_ny.csv found!')
    else:
        print('clean_ny.csv not found! Requesting data...')  # File is requested via api if not found
        ny = final_ny()
        print('Data acquired, and cached.')  # File is acquired and processed and cached

    if os.path.isfile('scraped_reviews.csv'):  # Checks for local file
        scraped_reviews = pd.read_csv('scraped_reviews.csv')  # Reads local file is there is one
        print('scraped_reviews.csv found!')  # Tells user file is found
    else:
        print('scraped_reviews.csv NOT found!')  # Function will end if it cannot find the scraped reviews data
        raise Exception('Function needs either scraped_reviews.csv or ny_reviews.csv saved locally to continue.')

    if include_api:  # Function will read api data if include api is true
        if os.path.isfile('api_reviews.csv'):  # Checks for local file
            api_reviews = pd.read_csv('api_reviews.csv')  # Assigns local file if there is one
            print('api_reviews.csv found!')
        else:
            print('api_reviews.csv NOT found!')
            raise Exception('Function could not find api_reviews.csv saved locally.')
    else:
        api_reviews = None  # If include_api is false api reviews is assigned no value

    reviews = cleanse_reviews(scraped_reviews, api_reviews)  # Cleans and concatenates reviews together into one df

    # Appends review data to ny inspection data based off review and inspection date
    ny_reviews = ny_pipeline(ny, reviews)

    ny_reviews.to_csv('ny_reviews.csv', index=False)  # Cache file
    print('reviews.csv cached')  # Tells user reviews csv was cached

    return ny_reviews  # Return df


# ---------------------------------------------------------------------------------------------------------------------
# Census data prep functions
