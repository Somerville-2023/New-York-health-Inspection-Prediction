import logging
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Function to create and return the driver
def get_driver(debug=False):
    # Logging for debugging
    print("Starting get_driver")
    print("Initializing ChromeDriver...")

    # Specify the path to Chrome Beta and ChromeDriver
    chrome_binary_path = "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"
    driver_path = "/Users/jongarcia/codeup-data-science/googlemaps-scraper/chromedriver-mac-x64 BETA/chromedriver"

    # Set up Chrome options
    options = Options()
    options.binary_location = chrome_binary_path

    # Headless option based on debug flag
    if not debug:
        options.add_argument("--headless")
    else:
        options.add_argument("--window-size=1920,1080")

    options.add_argument("--disable-notifications")
    options.add_argument("--accept-lang=en-GB")
    options.add_argument("--window-size=1920,1080")
    
    # Configure the WebDriver to use the Tor SOCKS proxy
    options.add_argument("--proxy-server=socks5://127.0.0.1:9050")

    # Create the WebDriver
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    print("ChromeDriver successfully initialized.")
    return driver

# Function to get IP address without using Tor
def get_ip_without_tor():
    response = requests.get('http://httpbin.org/ip')
    return response.json()['origin']

# Function to get IP address using Tor
def get_ip_with_tor(driver):
    driver.get('http://httpbin.org/ip')
    response_text = driver.find_element(By.TAG_NAME, 'pre').text
    return eval(response_text)['origin']

# Function to test Tor connection
def test_tor_connection(driver):
    print("Testing Tor connection...")

    ip_without_tor = get_ip_without_tor()
    ip_with_tor = get_ip_with_tor(driver)

    if ip_without_tor != ip_with_tor:
        print("IP masking successful: Your IP address is being masked by Tor.")
        print(f"Real IP: {ip_without_tor}")
        print(f"Tor IP: {ip_with_tor}")
        return True
    else:
        print("IP masking unsuccessful: Your IP address is not being masked by Tor.")
        print(f"Real IP: {ip_without_tor}")
        print(f"Tor IP: {ip_with_tor}")
        return False

# Main execution
if __name__ == "__main__":
    driver = get_driver(debug=True)
    tor_connection_test_result = test_tor_connection(driver)
    driver.quit()
