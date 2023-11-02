import os
import pandas as pd
import json
import requests

def construct_text_query(row):
    text_query = f"{row['dba']} {row['full_address']} {row['boro']}"
    if pd.notnull(row['phone']):
        text_query += f" {row['phone']}"
    return text_query

def process_place(place, row, places_results):
    # Extract place ID from the 'name' field
    place_id = place.get('name', '').split('/')[-1]

    places_results.append({
        'camis': row['camis'],
        'place_id': place_id
    })

def log_api_call(current_row, text_query, response, places_data, api_logs):
    for place in places_data:
        # Extract place ID from the 'name' field
        place_id = place.get('name', '').split('/')[-1]

        api_logs.append({
            'row_number': current_row,
            'query': text_query,
            'status_code': response.status_code,
            'place_id': place_id
        })

def save_progress(places_results, api_logs):
    pd.DataFrame(places_results).to_csv('places_progress.csv', mode='a', header=False, index=False)
    pd.DataFrame(api_logs).to_csv('api_log_progress.csv', mode='a', header=False, index=False)

def main(inspections_df, g_places_api_key, save_interval=10):
    places_results = []
    api_logs = []

    url = 'https://places.googleapis.com/v1/places:searchText'
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': g_places_api_key,
        'X-Goog-FieldMask': 'places.name'
    }

    try:
        csv_row = pd.read_csv('api_log_progress.csv')
        last_row_processed = csv_row.tail(1).row_number.iloc[0]
        print(f"Resuming from row: {last_row_processed + 1}")
    except FileNotFoundError:
        last_row_processed = -1

    if not os.path.isfile('places_progress.csv'):
        pd.DataFrame(columns=['camis', 'place_id']).to_csv('places_progress.csv', index=False)
    if not os.path.isfile('api_log_progress.csv'):
        pd.DataFrame(columns=['row_number', 'query', 'status_code', 'place_id']).to_csv('api_log_progress.csv', index=False)

    current_row = 0

    for index, row in inspections_df.iterrows():
        current_row += 1
        
        if current_row <= last_row_processed:
            continue

        text_query = construct_text_query(row)
        data = {'textQuery': text_query}
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            places_data = response.json().get('places', [])
            for place in places_data:
                process_place(place, row, places_results)

            log_api_call(current_row, text_query, response, places_data, api_logs)
        else:
            print(f"Failed to retrieve data for {text_query}: {response.status_code}")
            log_api_call(current_row, text_query, response, [], api_logs)

        if current_row % save_interval == 0:
            save_progress(places_results, api_logs)
            places_results.clear()
            api_logs.clear()
            print(f'saved progress, row #{current_row}')
            
    pd.DataFrame(places_results).to_csv('places_final.csv', index=False)
    pd.DataFrame(api_logs).to_csv('api_log_final.csv', index=False)

