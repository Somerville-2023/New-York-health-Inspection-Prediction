# How to Set Up for Web Scraping using Tor

### 1. Copy the Project Folder

Copy the "scraper_gmaps" folder located under the "john-folder" to your own working directory. This folder contains the initial files needed for the scraping project.

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

Activate the newly created Conda environment.

For Unix-like operating systems (Linux, macOS):
```sh
source activate j_scraper
```

Or:
```sh
conda activate j_scraper
```

### 4. Install Required Python Packages

Install the necessary Python packages for your web scraping project using `pip`.

```sh
pip install pandas selenium requests stem termcolor
```

### 5. Install Tor Using Homebrew

Install Tor on your system using Homebrew.

```sh
brew install tor
```

### 6. Edit the Tor Configuration File (`torrc`)

Copy the sample Tor configuration file to create a `torrc` file.

```sh
sudo cp /usr/local/etc/tor/torrc.sample /usr/local/etc/tor/torrc

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

### 7. Restart Tor

Restart the Tor service to apply the configuration changes:
```sh
brew services restart tor
```

**Note:** The command to restart Tor may vary depending on your system;some may require `sudo`.

### 8. Install Chrome Beta and ChromeDriver Beta

Download and install Chrome Beta **and** download ChromeDriver Beta for your operating system from the following resource:

[Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/)

After downloading ChromeDriver Beta, place it in the root of your scraping project folder. For Unix-like operating systems (macOS, Linux), make sure to make ChromeDriver executable:

```sh
chmod +x /path/to/chromedriver
```

**Note:** Replace `/path/to/chromedriver` with the actual path to the ChromeDriver on your system. Ensure ChromeDriver is placed in the root of your scraping project folder and made executable if you are on a Unix-like system.

### 9. Modify googlemaps.py function

Near the bottom of the module you will find the "__get_driver" function.
Update the file paths. chrome_binary_path ***may** be the same as it uses the default install path for chrome beta, but driver_path will have to be updated. 

```sh
    def __get_driver(self, debug=False):
        logging.debug(f"Starting __get_driver")
        logging.debug("Initializing ChromeDriver...")
        # Specify the path to the Chrome Beta binary
        chrome_binary_path = "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"

        # Specify the path to the ChromeDriver executable for Chrome Beta
        driver_path = "/Users/jongarcia/codeup-data-science/googlemaps-scraper/chromedriver-mac-x64 BETA/chromedriver"
```


