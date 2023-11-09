import pandas as pd
import nltk
from scipy.stats import chi2_contingency
# import visual as v

def counts_and_ratios(df, column):
    """
    Takes in a dataframe and a string of a single column
    Returns a dataframe with absolute value counts and percentage value counts
    """
    labels = pd.concat([df[column].value_counts(),
                    df[column].value_counts(normalize=True)], axis=1)
    labels.columns = ['n', 'percent']
    labels
    return labels


def join_text(df):
    """
    Join text data from a DataFrame based on grade labels and combine all text data.

    Args:
        df (pd.DataFrame): The DataFrame containing text data and language labels.

    Returns:
        tuple: A tuple containing the following joined text data:
            - a_reviews (str): Concatenated text from the DataFrame where the label is 'a'
            - b_reviews (str): Concatenated text from the DataFrame where the label is 'b'.
            - c_reviews (str): Concatenated text from the DataFrame where the label is 'c'.
            - all_reviews (str): Concatenated text from the entire DataFrame.
    """
    # Join all the text from the DataFrame where the label is 'C++'
    a_reviews = ' '.join(df[df.grade == 'A'].reviews)

    # Join all the text from the DataFrame where the label is 'Python'
    b_reviews = ' '.join(df[df.grade == 'B'].reviews)

    # Join all the text from the DataFrame where the label is 'Other'
    c_reviews = ' '.join(df[df.grade == 'C'].reviews)

    # Join all the text from the entire DataFrame
    all_reviews = ' '.join(df.reviews)
    
    return a_reviews, b_reviews, c_reviews, all_reviews

def list_words(df, text_column):
    """
    Create lists of words from the specified text column of a DataFrame based on grade labels and for all data.

    Args:
        df (pd.DataFrame): The DataFrame containing text data and grade labels.
        text_column (str): The name of the text column to process. Defaults to 'reviews'.

    Returns:
        tuple: A tuple containing the following lists of words:
            - a_words (pd.Series): Words from the specified text column for 'A' graded data.
            - b_words (pd.Series): Words from the specified text column for 'B' graded data.
            - c_words (pd.Series): Words from the specified text column for 'C' graded data.
            - all_words (pd.Series): Words from the specified text column for all data.
    """
    a_words = df[df['grade'] == 'A'][text_column].str.split(expand=True).stack()
    b_words = df[df['grade'] == 'B'][text_column].str.split(expand=True).stack()
    c_words = df[df['grade'] == 'C'][text_column].str.split(expand=True).stack()
    all_words = df[text_column].str.split(expand=True).stack()
    
    return a_words, b_words, c_words, all_words


def word_freq(df, text_column):
    """
    Calculate word frequencies for different grade labels and for all data in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'reviews' text data and grade labels.

    Returns:
        tuple: A tuple containing the following word frequency Series:
            - a_freq (pd.Series): Word frequencies for 'A' graded data.
            - b_freq (pd.Series): Word frequencies for 'B' graded data.
            - c_freq (pd.Series): Word frequencies for 'C' graded data.
            - all_freq (pd.Series): Word frequencies for all data.
    """
    # Create lists of words based on grade labels and for all data
    a_words, b_words, c_words, all_words = list_words(df, text_column)

    # Calculate word frequencies and sort in descending order
    a_freq = pd.Series(a_words).value_counts().sort_values(ascending=False).astype(int)
    b_freq = pd.Series(b_words).value_counts().sort_values(ascending=False).astype(int)
    c_freq = pd.Series(c_words).value_counts().sort_values(ascending=False).astype(int)
    all_freq = pd.Series(all_words).value_counts().sort_values(ascending=False).astype(int)
    
    return a_freq, b_freq, c_freq, all_freq


def word_counts(df, reset_index=True, text_column='reviews'):
    """
    Process and sort word frequency DataFrames.

    Args:
        df (pd.DataFrame): DataFrame containing word frequency DataFrames.
        reset_index (bool, optional): Whether to reset the index and start it at 1. Default is True.

    Returns:
        pd.DataFrame: Sorted and processed word counts DataFrame with "word" column as the first column,
                      and index reset based on the reset_index parameter.
    """
    a_freq, b_freq, c_freq, all_freq = word_freq(df, text_column)
    
    # Concatenate the DataFrames and set column names
    word_counts = (pd.concat([all_freq, a_freq, b_freq, c_freq], axis=1, sort=True)
                    .set_axis(['All', 'A', 'B', 'C'], axis=1)
                    .fillna(0)
                    .applymap(int))

    # Sort by the 'All' column in descending order
    word_counts = word_counts.sort_values(by='All', ascending=False)

    if reset_index:
        # Move the index into a column named "word"
        word_counts.reset_index(inplace=True)
        word_counts.rename(columns={'index': 'word'}, inplace=True)

        # Reset the index and start it at 1
        word_counts.index += 1
        
    return word_counts
