# How to Set Up for Web Scraping using Tor

### 1. Copy the Project Folder

Copy the "scraper_gmaps" folder located under the "john-folder" to your own working directory. This folder contains the initial files needed for the scraping project.

If you would like to use terminal try:

```sh
cp -r /path/to/john-folder/scraper_gmaps /path/to/your/folder
```

**Note:** Replace `/path/to/john-folder/scraper_gmaps` and `/path/to/your/folder` with the actual paths on your system.

### 2. Create a New Conda Environment

Create a new Conda environment named `j_scraper` with the latest version of Python.

```sh
conda create --name j_scraper python=3.12
```

### 3. Activate the Conda Environment

Activate the newly created Conda environment using one of the commands below:

```sh
source activate j_scraper
```
**OR**
```sh
conda activate j_scraper
```

If the activation was successful you should the new environment in parenthesis:
```sh
(j_scraper) name-Macbook pat%
```

### 4. Install Required Python Packages

Install the necessary Python packages for your web scraping project using `pip`.

```sh
pip install pandas selenium requests stem termcolor bs4
```

### 5. Install Tor Using Homebrew

Install Tor on your system using Homebrew.

```sh
brew install tor
```

Run Tor to ensure its installed properly

```sh
tor
```

### 6. Install Chrome Beta and download ChromeDriver Beta

Download and install [Chrome Beta](https://www.google.com/chrome/beta/) 👈 using the default settings.

**Also** download ChromeDriver `Beta` from:

For  [mac-x64](https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/120.0.6099.5/mac-x64/chrome-mac-x64.zip). 

For  [mac-arm64](https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/120.0.6099.5/mac-arm64/chromedriver-mac-arm64.zip). 

if the links above expire, go directly to the google chrome driver site 👉  [Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/#beta). 

After downloading ChromeDriver Beta, you will have to unzip it. You can choose to keep it in the download folder, it won't affect anything. However you `must` make sure to make ChromeDriver executable:

```sh
chmod +x /path/to/chromedriver-mac-x64 BETA/chromedriver
```
**Note:** Replace `/path/to/chromedriver` with the actual path to the ChromeDriver on your system. Ensure ChromeDriver is placed in the root of your scraping project folder and made executable if you are on a Unix-like system.


You can also give it permission `manually` by going to  downloads/chromedriver-mac-x64 BETA/ and running the chromedriver. You will be asked to give it permission. 

`hint` click on the **?** in the pop up window, then read and follow the blue link near the top that will take you directly to the settings page. 



### 7. Modify googlemaps.py function

The python module will need to know the location of both the installed Chrome Beta app **and** the chromedriver

open googlemaps.py and cmd + f and search for "__get_driver"
function. It will be the second or third result. 

Update the file paths. chrome_binary_path ***may** be the same as it uses the default install path for chrome browser beta, but driver_path for chromedriver will have to be updated. Should be something similar to /path/to/chromedriver-mac-x64 BETA/chromedriver

Code snippet of the __get_driver function:

```sh
    def __get_driver(self, debug=False):
        logging.debug(f"Starting __get_driver")
        logging.debug("Initializing ChromeDriver...")
        # Specify the path to the Chrome Beta binary
        chrome_binary_path = "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"

        # Specify the path to the ChromeDriver executable for Chrome Beta
        driver_path = "/Users/jongarcia/codeup-data-science/googlemaps-scraper/chromedriver-mac-x64 BETA/chromedriver"
```

### 8 Create files manually.
You will manually need to create 2 files, I haven't had time to add a programmatic way to create the files in the script sorry. 

Create a file named "url_list.csv" and place it in the scraper_gmaps folder. add these 2 rows which will be used for testing purposes when we run the script. 

```sh
id,url
41073979,https://www.google.com/maps/place/?q=place_id:ChIJ79lG6ylawokRHzGFI0I0V8k
50055023,https://www.google.com/maps/place/?q=place_id:ChIJw0ul69tcwokRUXZtYK2viEw
41046488,https://www.google.com/maps/place/?q=place_id:ChIJN8ko6jxawokRh4mOVeF82dg

```

Under the scraper_gmaps folder you will see the data folder. Create an empty file named "newest_gm_reviews.csv".

### 9 Run the script.
Open the file named "scraper_notebook.ipynb" and run the first cell. select the kernel j_scraper when asked.


OPTIONAL:

### 10A. Create a password


First think of a password. Using terminal, create a hashed password:
  ```sh
  tor --hash-password chosen_password
  ```

your raw `chosen_password` will turned into a 16 digit hash for Tor's control port authentication:. You need to keep both the raw password and the hashed password. 


The raw pass will be saved to your env file as torp. 

The hash password will be saved to the torrc file which stores the tor configuration which we will create in the next step. 

### 10B Create the torrc config file

We need to create a file named torrc which will hold the tor configuration in path /usr/local/etc/tor/

You have 2 options:

  - 1) find the folder manually, create the torrc file using a text editor and add these lines:
      ```sh
      ControlPort 9051
      SOCKSPort 9050
      HashedControlPassword YOUR-HASHED-PASS
      ```

 - 3) create a new torrc file using terminal.
    ```sh
    sudo sh -c 'echo -e "ControlPort 9051\nSOCKSPort 9050\nHashedControlPassword YOUR_PASS" > /usr/local/etc/tor/torrc'
```

Open the `torrc` file with Visual Studio Code for editing.

```sh
code /usr/local/etc/tor/torrc
```

Using terminal, create a hashed password for Tor's control port authentication:
  ```sh
  tor --hash-password your_chosen_password
  ```
  Copy the resulting hashed password. Now open the `torrc` file with Visual Studio Code for editing.

```sh
code /usr/local/etc/tor/torrc
```
- Paste your password
  ```sh
  HashedControlPassword 16:YOUR_HASHED_PASSWORD_HERE
  ```
  **Note:** Replace `your_chosen_password` with the password you wish to use for Tor's `HashedControlPassword` and `16:YOUR_HASHED_PASSWORD_HERE` with the actual hashed password you generated. Save your changes in the `torrc` file.
### 6. Restart Tor

start the Tor service to apply the configuration changes:
```sh
brew services restart tor
```

**Note:** The command to restart Tor may vary depending on your system;some may require `sudo`.