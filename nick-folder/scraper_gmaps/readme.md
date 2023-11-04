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
**Note:** Replace `/path/to/chromedriver` with the actual path to the ChromeDriver on your system.

You can also give it permission `manually` by going to  downloads/chromedriver-mac-x64 BETA/ and running the chromedriver. You will be asked to give it permission. 

`hint` click on the **?** in the pop up window, then read and follow the blue link near the top that will take you directly to the settings page. 


### 7. Modify googlemaps.py function

The python module will need to know the location of both the installed Chrome Beta app **and** the chromedriver

open googlemaps.py and cmd + f and search for "__get_driver"
function. It will be the second or third result. 

Update the file paths. 

Chrome_binary_path **may** not need to be updated as it uses the default install path for chrome browser beta.

Driver_path for chromedriver will have to be updated. It should be something similar to /path/to/chromedriver-mac-x64 BETA/chromedriver

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

### 8 Create needed files manually.
You will need to manually create 2 files, I didn't have time to add a programmatic way to create the files in the script. 

Create a file named "url_list.csv" and place it in the scraper_gmaps folder. add these rows which will be used for testing purposes only. 

```sh
id,url
41073979,https://www.google.com/maps/place/?q=place_id:ChIJ79lG6ylawokRHzGFI0I0V8k
50055023,https://www.google.com/maps/place/?q=place_id:ChIJw0ul69tcwokRUXZtYK2viEw
41046488,https://www.google.com/maps/place/?q=place_id:ChIJN8ko6jxawokRh4mOVeF82dg

```

Under the scraper_gmaps folder you will see the data folder. Create an empty file named "newest_gm_reviews.csv".

### 9 Run the script.
Open the file named "scraper_notebook.ipynb" and run the first cell. select the kernel j_scraper when asked.

___
## OPTIONAL
___


### 10A. Create a password

The following steps are cumbersome, and Im not 100% sure they're required but it probably makes Tor more secure. I just don't understand why yet. 

overall we are going to create a torrc file which is tor's config file. in there we will place 3 lines, one of which is a hashed password. When we run tor using the py module, it uses this config file. However nick ran it without this file and it worked fine. 


First think of a password. Using terminal, create a hashed password:
  ```sh
  tor --hash-password chosen_password
  ```

your raw `chosen_password` will turned into a 16 digit hash for Tor's control port authentication:. You need to keep both the raw password and the hashed password handy for the next steps.  

The hash password will be saved to the torrc config file which we will create in the next step. 

### 10B Create the torrc config file

We need to create the torrc file in path /usr/local/etc/tor/. You have 2 options to create it, whichever one you choose, paste your hashed password in place of "YOUR-HASHED-PASS":

  - option 1: find the folder manually at path /usr/local/etc/tor/. Create the torrc file using a text editor and add these lines:
      ```sh
      ControlPort 9051
      SOCKSPort 9050
      HashedControlPassword YOUR-HASHED-PASS
      ```

 - option 2: create a new torrc file using terminal. This option will require you to enter your Mac password as it is a protected folder location and you need admin rights to create a file here. 
    ```sh
      sudo sh -c 'echo -e "ControlPort 9051\nSOCKSPort 9050\nHashedControlPassword YOUR_PASS" > /usr/local/etc/tor/torrc'
    ```
    After creating the file, open the `torrc` file with VS Code for editing.

    ```sh
    code /usr/local/etc/tor/torrc
    ```

### 10C. Restart Tor

Restart the Tor service to apply the configuration changes:
```sh
brew services restart tor
```

**Note:** The command to restart Tor may vary depending on your system;some may require `sudo`.