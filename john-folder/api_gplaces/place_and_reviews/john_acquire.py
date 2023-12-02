import pandas as pd
import numpy as np
import os
import requests
import env

def check_and_load_csv(start_year, end_year):
    filename = f'nyc_health_inspections_{start_year}_to_{end_year}.csv' if start_year != end_year else f'nyc_health_inspections_{start_year}.csv'
    if os.path.isfile(filename):
        print(f"CSV file for {start_year} to {end_year} already exists. Loading data from the CSV.")
        return pd.read_csv(filename)
    return None

def make_api_request(start_year, end_year, app_token, offset, page_size):
    base_url = 'https://data.cityofnewyork.us/resource/43nn-pn8j.json'
    url = f'{base_url}?$where=inspection_date between "{start_year}-01-01T00:00:00.000" and "{end_year}-12-31T23:59:59.999"&$$app_token={app_token}&$offset={offset}&$limit={page_size}'
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
    return response.json()

def process_data(start_year, end_year, app_token, max_observations=None):
    offset = 0
    page_size = 1000
    all_data = []
    
    while max_observations is None or len(all_data) < max_observations:
        remaining_observations = max_observations - len(all_data) if max_observations is not None else page_size
        actual_page_size = min(page_size, remaining_observations)
        data = make_api_request(start_year, end_year, app_token, offset, actual_page_size)
        if not data:
            break
        all_data.extend(data)
        offset += actual_page_size
        if max_observations is not None and len(all_data) >= max_observations:
            break

    return pd.DataFrame(all_data)

def get_health_inspection_data(start_year, end_year, app_token, max_observations=None):
    if end_year < start_year:
        raise ValueError("End year must be greater than or equal to start year")

    df = check_and_load_csv(start_year, end_year)
    if df is not None:
        return df

    df = process_data(start_year, end_year, app_token, max_observations)
    csv_filename = f'nyc_health_inspections_{start_year}_to_{end_year}.csv' if start_year != end_year else f'nyc_health_inspections_{start_year}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Health inspection data from {start_year} to {end_year} retrieved and saved to {csv_filename}.")
    
    return df


def construct_text_query(row):
    text_query = f"{row['dba']} {row['full_address']} {row['boro']}"
    # text_query = f"{row['dba']} {row['building']} {row['street']} {row['boro']}"
    # if pd.notnull(row['zipcode']):
    #     text_query += f" {int(row['zipcode'])}"
    if pd.notnull(row['phone']):
        text_query += f" {row['phone']}"
    # if pd.notnull(row['cuisine_description']):
    #     text_query += f" {row['cuisine_description']}"
    return text_query

def process_place(place, row, places_results, reviews_results):
    display_name = place.get('displayName', {}).get('text', '').strip()
    formatted_address = place.get('formattedAddress', '').strip()
    address_first_part = formatted_address.split(',')[0].strip()

    note = ''
    if display_name.lower() == address_first_part.lower():
        note = 'Possible address only, no specific business match'

    places_results.append({
        'camis': row['camis'],
        'place_id': place.get('id', ''),
        'display_name': display_name,
        'formatted_address': formatted_address,
        'note': note
    })

    if note == '':
        for review in place.get('reviews', []):
            review_id = review.get('name', '').split('/')[-1]
            try:
                contributor_id = review.get('authorAttribution', {}).get('uri', '').split('/')[-2]
            except:
                contributor_id = "unknown"
            reviews_results.append({
                'camis': row['camis'],
                'place_id': place.get('id', ''),
                'review_id': review_id,
                'review_relative_time': review.get('relativePublishTimeDescription', ''),
                'review_rating': review.get('rating', ''),
                'review_text': review.get('text', {}).get('text', ''),
                'review_language': review.get('text', {}).get('languageCode', ''),
                'author_display_name': review.get('authorAttribution', {}).get('displayName', ''),
                'contributor_id': contributor_id,
                'author_photo_uri': review.get('authorAttribution', {}).get('photoUri', ''),
                'publish_time': review.get('publishTime', '')
            })

def log_api_call(current_row, text_query, response, places_data, api_logs):
    for place in places_data:
        display_name = place.get('displayName', {}).get('text', '').strip()
        formatted_address = place.get('formattedAddress', '').strip()
        address_first_part = formatted_address.split(',')[0].strip()

        note = ''
        if display_name.lower() == address_first_part.lower():
            note = 'Possible address only, no specific business match'

        api_logs.append({
            'row_number': current_row,
            'query': text_query,
            'status_code': response.status_code,
            'display_name': display_name,
            'formatted_address': formatted_address,
            'note': note
        })

# def save_progress(places_results, reviews_results, api_logs):
#     pd.DataFrame(places_results).to_csv('places_progress.csv', index=False)
#     pd.DataFrame(reviews_results).to_csv('reviews_progress.csv', index=False)
#     pd.DataFrame(api_logs).to_csv('api_log_progress.csv', index=False)
def save_progress(places_results, reviews_results, api_logs):
    pd.DataFrame(places_results).to_csv('places_progress.csv', mode='a', header=False, index=False)
    pd.DataFrame(reviews_results).to_csv('reviews_progress.csv', mode='a', header=False, index=False)
    pd.DataFrame(api_logs).to_csv('api_log_progress.csv', mode='a', header=False, index=False)



def main(inspections_df, g_places_api_key, save_interval=10):
    # Initialize empty lists to store the results for places, reviews, and API logs
    places_results = []
    reviews_results = []
    api_logs = []

    # Define the URL and headers
    url = 'https://places.googleapis.com/v1/places:searchText'
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': g_places_api_key,
        'X-Goog-FieldMask': 'places.id,places.displayName,places.formattedAddress,places.reviews'
    }
    # Check if there is saved progress
    try:
        csv_row = pd.read_csv('api_log_progress.csv')
        last_row_processed = csv_row.tail(1).row_number.iloc[0]
        print(f"Resuming from row: {last_row_processed + 1}")
    except FileNotFoundError:
        last_row_processed = -1

    # Initialize empty lists for the first run
    if not os.path.isfile('places_progress.csv'):
        pd.DataFrame(columns=['camis', 'place_id', 'display_name', 'formatted_address', 'note']).to_csv('places_progress.csv', index=False)
    if not os.path.isfile('reviews_progress.csv'):
        pd.DataFrame(columns=['camis', 'place_id', 'review_id', 'review_relative_time', 'review_rating', 'review_text', 'review_language', 'author_display_name', 'contributor_id', 'author_photo_uri', 'publish_time']).to_csv('reviews_progress.csv', index=False)
    if not os.path.isfile('api_log_progress.csv'):
        pd.DataFrame(columns=['row_number', 'query', 'status_code', 'display_name', 'formatted_address', 'note']).to_csv('api_log_progress.csv', index=False)

    
    current_row = 0

    for index, row in inspections_df.iterrows():
        current_row += 1
        
        # Skip already processed rows
        if current_row <= last_row_processed:
            continue
        
        text_query = construct_text_query(row)

        # Define the query
        data = {'textQuery': text_query}

        # Make the POST request
        response = requests.post(url, headers=headers, json=data)

        # Process the response
        if response.status_code == 200:
            places_data = response.json().get('places', [])
            
            for place in places_data:
                process_place(place, row, places_results, reviews_results)

            log_api_call(current_row, text_query, response, places_data, api_logs)
        else:
            print(f"Failed to retrieve data for {text_query}: {response.status_code}")
            log_api_call(current_row, text_query, response, [], api_logs)

        # Save progress at regular intervals
        if current_row % save_interval == 0:
            save_progress(places_results, reviews_results, api_logs)
            # Clear the lists after saving
            places_results.clear()
            reviews_results.clear()
            api_logs.clear()
            print(f'saved progress, row #{current_row}')
            
    # Save the final results
    pd.DataFrame(places_results).to_csv('places_final.csv', index=False)
    pd.DataFrame(reviews_results).to_csv('reviews_final.csv', index=False)
    pd.DataFrame(api_logs).to_csv('api_log_final.csv', index=False)
