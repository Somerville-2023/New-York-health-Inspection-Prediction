import pandas as pd
import unicodedata
import re
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
nltk.download('wordnet')

from markdown import markdown

def basic_clean(data):
    # Replace tabs and line breaks with a space
    data = re.sub(r'[\t\n\r]+', ' ', data)
    
    # Make text lowercase
    data = data.lower()
    
    # Define a regex pattern to match all characters except alphanumeric characters and emojis
    emoji_pattern = re.compile(
        "[^\w\s"  # This matches any character that is not alphanumeric (\w) or space (\s)
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]", flags=re.UNICODE)

    # Remove special characters, preserve emojis and alphanumeric
    data = emoji_pattern.sub(r'', data)
    
    # Replace multiple spaces with a single space
    data = re.sub(r' +', ' ', data)
    
    return data.strip()


def tokenize(data):
    """
    Tokenize the input text data using a tokenizer object and apply additional text processing
    while preserving emojis.
    """
    # Initialize a tokenizer object
    tokenizer = ToktokTokenizer()

    # Tokenize the input data using the tokenizer object
    data = tokenizer.tokenize(data, return_str=True)

    # Define a pattern that includes lowercase letters, numbers, whitespaces, and emojis
    allowed_pattern = re.compile(
        "[^a-z0-9\s"  # lowercase letters, numbers, and whitespaces
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]", flags=re.UNICODE)

    # Remove characters that are not in the allowed pattern
    data = allowed_pattern.sub("", data)

    # Remove single-digit numbers surrounded by spaces (accounting for emojis)
    data = re.sub(r"(?<=\s)\d(?=\s)", "", data)
    
    # Return the processed data
    return data


def stem(data):
    """
    Apply stemming to the input text data using the Porter Stemmer algorithm.

    Args:
        data (str): The input text data to be stemmed.

    Returns:
        str: The stemmed text data after applying stemming to each word.
    """
    # Create an instance of the PorterStemmer class from the nltk library
    ps = nltk.porter.PorterStemmer()
    
    # Split the input data into a list of words
    words = data.split()
    
    # Apply stemming to each word in the input data
    stems = [ps.stem(word) for word in words]

    # Join the stemmed words into a single string with spaces in between
    stemmed_data = ' '.join(stems)

    # Return the stemmed data
    return stemmed_data

def lemmatize(data):
    """
    Apply lemmatization to the input text data using WordNet Lemmatizer.

    Args:
        data (str): The input text data to be lemmatized.

    Returns:
        str: The lemmatized text data after applying lemmatization to each word.
    """
    # Create an instance of WordNetLemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Split the input data into a list of words
    words = data.split()
    
    # Lemmatize each word in the input data
    lemmas = [wnl.lemmatize(word) for word in words]

    # Join the lemmatized words into a single string
    lemmatized_data = ' '.join(lemmas)

    # Return the lemmatized data
    return lemmatized_data


def remove_stopwords(data, extra_words=[], exclude_words=[]):
    """
    Remove stopwords from the input text data while allowing for additional and exclusionary stopwords.

    Args:
        data (str): The input text data from which stopwords will be removed.
        extra_words (list): Additional stopwords to be considered.
        exclude_words (list): Words to be excluded from the list of stopwords.

    Returns:
        str: The text data with stopwords removed based on the provided lists.
    """
    # Create a list of stopwords in English
    stopwords_list = stopwords.words('english')

    # Extend the stopwords_list with the elements from the extra_words list
    stopwords_list.extend(extra_words)

    # Iterate over each word in the exclude_words list
    for word in exclude_words:
        # Check if the word exists in the stopwords_list
        if word in stopwords_list:
            # Remove the word from the stopwords_list
            stopwords_list.remove(word)

    # Split the input data into individual words and filter out stopwords
    words = [word for word in data.split() if word not in stopwords_list]
    
    # Join the filtered words back into a string
    data = ' '.join(words)
    
    # Return the processed data
    return data



def preprocess_text_column(df, extra_words=[], exclude_words=[], method='stem'):
    """
    Preprocess the 'text' column of a DataFrame by applying text cleaning and processing steps.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        extra_words (list): Additional words to include in stopwords removal.
        exclude_words (list): Words to exclude from stopwords removal.
        method (str): Text processing method ('stem' or 'lemmatize').

    Returns:
        None: The function modifies the DataFrame 'df' in place.
    """
    # Apply basic cleaning and tokenization to 'text_contents' column
    df['reviews'] = df['concatenated_reviews'].apply(basic_clean).apply(tokenize)
    
    # Drop the 'text_contents' column
    df.drop(columns='concatenated_reviews', axis=1, inplace=True)
    
    # # Apply stopwords removal and text processing based on the selected method
    # df['text'] = df['text'].apply(lambda x: remove_stopwords(x, extra_words, exclude_words))
    
    if method == 'stem':
        # Apply stemming to the 'text' column
        df['reviews'] = df['reviews'].apply(stem)
    elif method == 'lemmatize':
        # Apply lemmatization to the 'text' column
        df['reviews'] = df['reviews'].apply(lemmatize)
    
    # Apply stopwords removal and text processing based on the selected method
    df['reviews'] = df['reviews'].apply(lambda x: remove_stopwords(x, extra_words, exclude_words))
    
    return df