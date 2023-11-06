# -*- coding: utf-8 -*-
from googlemaps import GoogleMapsScraper
from datetime import datetime, timedelta
import argparse
import csv
from termcolor import colored
import time
import os
import requests
from stem import Signal
from stem.control import Controller



ind = {'most_relevant' : 0 , 'newest' : 1, 'highest_rating' : 2, 'lowest_rating' : 3 }
HEADER = ['id', 'id_review', 'caption', 'relative_date', 'retrieval_date', 'rating', 'username', 'n_review_user', 'url_user', 'r_additional'] #'n_photo_user'
HEADER_W_SOURCE = ['id_review', 'caption', 'relative_date','retrieval_date', 'rating', 'username', 'n_review_user', 'url_user', 'r_additional', 'url_source'] #'n_photo_user'

# Function to write to CSV
def csv_writer(source_field, ind_sort_by, path='data/'):
    outfile = ind_sort_by + '_gm_reviews.csv'
    mode = 'a' if os.path.exists(path + outfile) else 'w'
    targetfile = open(path + outfile, mode=mode, encoding='utf-8', newline='\n')
    writer = csv.writer(targetfile, quoting=csv.QUOTE_MINIMAL)

    # Write header only if in write mode
    if mode == 'w':
        header = HEADER_W_SOURCE if source_field else HEADER
        writer.writerow(header)

    return writer, targetfile

# Function to get the last ID from the CSV
def get_last_id_from_csv(input_file):
    try:
        with open(input_file, "r", encoding='utf-8') as f1:
            lines = f1.readlines()
            if len(lines) > 1:  # More than just the header
                last_line = lines[-1]
                last_id = last_line.split(',')[0]  # Assuming the ID is in the first column
                return last_id
            else:
                return None  # Only header exists, no data
    except FileNotFoundError:
        # If the file doesn't exist, return None
        return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Google Maps reviews scraper.')
    parser.add_argument('--N', type=int, default=100, help='Number of reviews to scrape')
    parser.add_argument('--i', type=str, default='urls.txt', help='target URLs file')
    parser.add_argument('--sort_by', type=str, default='newest', help='most_relevant, newest, highest_rating or lowest_rating')
    parser.add_argument('--place', dest='place', action='store_true', help='Scrape place metadata')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Run scraper using browser graphical interface')
    parser.add_argument('--source', dest='source', action='store_true', help='Add source url to CSV file (for multiple urls in a single file)')
    parser.set_defaults(place=False, debug=False, source=False)

    args = parser.parse_args()

    # Initialize CSV writer
    writer, targetfile = csv_writer(args.source, args.sort_by)
    input_file = 'data/' + args.sort_by + '_gm_reviews.csv'
    last_id = get_last_id_from_csv(input_file)


with GoogleMapsScraper(debug=args.debug) as scraper:
    print("Scraper initialized successfully.")
    
    count = 0
    renew_interval = 10  # Renew IP every 10 places
    
    # Test Tor connection
    if not scraper.test_tor_connection():
        print("Exiting: Tor connection failed.")
        exit(1)
        

    with open(args.i, 'r') as urls_file:
        reader = csv.DictReader(urls_file)
        start_processing = last_id is None  # Start processing immediately if last_id is None
        
        for row in reader:
            url_id = row['id']
            
            if not start_processing and url_id == last_id:
                start_processing = True
                continue  # Skip the URL that matches the last_id as it's already processed
            
            if not start_processing:
                continue  # Skip until the last processed ID is found

            url = row['url']
            # for url in urls_file:
            print(f"Processing URL: {url.strip()}")
            if args.place:
                account_info = scraper.get_account(url)
                print(f"Account Info: {account_info}")
            else:
                print("Attempting to sort reviews...")
                error = scraper.sort_by(url, ind[args.sort_by])
                print(f"Errors while readying to scrape: {error}")

                if error == 0:
                    n = 0
                    print(f"Starting to scrape reviews for {url.strip()}...")

                    while n < args.N:
                        print(colored(f'[Scraping from review {n}]', 'cyan'))
                        reviews = scraper.get_reviews(n)
                        print(f"Number of reviews fetched: {len(reviews)}")

                        if len(reviews) == 0:
                            print("No more reviews to fetch. Breaking out of the loop.")
                            break

                        for r in reviews:
                            ##This section starts the inclusion of ID
                            # row_data = list(r.values())
                            row_data = [url_id] + list(r.values())
                            # Ends inclusion of ID
                            if args.source:
                                row_data.append(url[:-1])
                            writer.writerow(row_data)

                        n += len(reviews)
                else:
                    print(f"Error encountered when sorting reviews for URL: {url.strip()}")

            # Increment count and check if it's time to renew IP
            count += 1
            print(f'The current page count is {count}')
            # if count % renew_interval == 0:
            #     print("Renewing Tor IP...")
            #     scraper.renew_tor_ip()
            if count % renew_interval == 0:
                print("Renewing Tor IP...")
                try:
                    scraper.renew_tor_ip()
                    print("Tor IP renewal attempted.")
                except Exception as e:
                    print(f"Failed to renew Tor IP: {e}")
                    # Decide what to do next, possibly break out of the loop or handle the exception in some way
                    # For example, you might want to stop the entire scraping process if you can't renew the IP
                    break
                
    # Close the file when done with writing
    targetfile.close()