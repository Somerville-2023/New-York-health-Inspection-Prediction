import matplotlib.pyplot as plt

def plot_top_words(word_counts, column='All', top_n=20, figsize=(12, 10), title=None, include_all=True):
    """
    Plots a horizontal bar chart of the top N words in a DataFrame with specified colors for each column.

    Parameters:
    word_counts (DataFrame): A DataFrame containing word counts for different grades.
    column (str): The name of the column to sort by and to place first in the stacked bars. Defaults to 'All'.
    top_n (int): The number of top records to plot. Defaults to 20.
    figsize (tuple): Width, height in inches. Defaults to (12, 10).
    title (str): The title of the plot. If None, a default title will be generated. Defaults to None.
    include_all (bool): Whether to include the 'All' column in the plot. Defaults to True.
    """
    # Define colors for each column
    colors = {'A': 'green', 'B': 'orange', 'C': 'red', 'All': 'grey'}

    # Select the top N rows by specified column
    top_words = word_counts.nlargest(top_n, column)

    # Sort the DataFrame by specified column in ascending order for the plot
    top_words = top_words.sort_values(by=column, ascending=True)

    # Create the columns list, placing the selected column first
    grade_columns = ['A', 'B', 'C']
    if include_all:
        grade_columns.insert(0, 'All')  # Include 'All' column if required
    columns = [column] + [col for col in grade_columns if col != column]

    # Create a list of colors for the plot based on the sorted columns
    plot_colors = [colors[col] for col in columns]

    # Create a horizontal bar chart with the columns in the new order and apply the colors
    ax = top_words[columns].plot(kind='barh', stacked=True, figsize=figsize, width=0.7, color=plot_colors)

    # Set plot title with larger font size
    if title is None:  # Check if a custom title is provided
        title = f'Top {top_n} most common words sorted by "{column}"'
    plt.title(title, fontsize=16)

    # Customize the plot aesthetics
    plt.xlabel('Frequency')  # Add x-axis label
    plt.ylabel('Words')  # Add y-axis label
    plt.xticks([])  # Remove x-ticks
    plt.yticks(fontsize=14)  # Increase font size of y labels
    plt.gca().tick_params(left=False)  # Remove y ticks

    # Add total count at the end of each bar
    for i, value in enumerate(top_words[column]):
        ax.text(value + 3,  # Slightly offset from the end of the bar
                i,  # Y-coordinate (aligned with each bar)
                str(value),  # The total count for the word
                va='center',  # Vertically align in the center of the bar
                fontsize=10)  # Consistent font size

    # Show the plot
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_top_words_freq(word_counts, column, top_n=20, figsize=(12, 10), title=None, include_all=True, custom_colors=None):
    """
    Plots a horizontal bar chart of the top N words in a DataFrame with specified colors for each column.

    Parameters:
    word_counts (DataFrame): A DataFrame containing word counts.
    column (str): The name of the column to sort by and to place first in the stacked bars.
    top_n (int): The number of top records to plot. Defaults to 20.
    figsize (tuple): Width, height in inches. Defaults to (12, 10).
    title (str): The title of the plot. If None, a default title will be generated. Defaults to None.
    include_all (bool): Whether to include the 'All' column in the plot. Defaults to True.
    custom_colors (dict): A dictionary containing color codes to use for each column.
    """
    # Ensure custom_colors is not None and it contains the column key
    if custom_colors is None or column not in custom_colors:
        raise ValueError("custom_colors must be a dictionary containing the column parameter as a key.")

    # If include_all is False and 'All' is in the colors, remove it
    if not include_all and 'All' in custom_colors:
        del custom_colors['All']

    # Select the top N rows by specified column
    top_words = word_counts.nlargest(top_n, column)

    # Sort the DataFrame by specified column in ascending order for the plot
    top_words = top_words.sort_values(by=column, ascending=True)

    # Create the columns list, placing the selected column first
    columns_to_plot = [column] + [col for col in custom_colors.keys() if col != column]

    # Create a list of colors for the plot based on the columns to plot
    plot_colors = [custom_colors[col] for col in columns_to_plot]

    # Create a horizontal bar chart with the columns in the new order and apply the colors
    ax = top_words[columns_to_plot].plot(kind='barh', stacked=True, figsize=figsize, width=0.7, color=plot_colors)

    # Set plot title with larger font size
    if title is None:  # Check if a custom title is provided
        title = f'Top {top_n} most common words sorted by "{column}"'
    plt.title(title, fontsize=16)

    # Customize the plot aesthetics
    plt.xlabel('Frequency')  # Add x-axis label
    plt.ylabel('Words')  # Add y-axis label
    plt.xticks([])  # Remove x-ticks
    plt.yticks(fontsize=14)  # Increase font size of y labels
    plt.gca().tick_params(left=False)  # Remove y ticks

    # Add total count at the end of each bar
    for i, value in enumerate(top_words[column]):
        ax.text(value + 3,  # Slightly offset from the end of the bar
                i,  # Y-coordinate (aligned with each bar)
                str(value),  # The total count for the word
                va='center',  # Vertically align in the center of the bar
                fontsize=10)  # Consistent font size

    # Show the plot
    plt.tight_layout()
    plt.show()
