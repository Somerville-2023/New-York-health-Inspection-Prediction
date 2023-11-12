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

def plot_top_words_freq(word_counts, column, top_n=20, figsize=(12, 10), title=None, include_all=True, custom_colors=None, reference_line_percent=None):
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
    
    # # Create the figure with a transparent background
    # fig = plt.figure(figsize=figsize)
    # fig.patch.set_alpha(0.0)

    # # Create the axes for the plot
    # ax = fig.add_subplot(111)
    # ax.patch.set_alpha(0.0)  # Make the axes background transparent

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
    plt.xlabel('')  # Add x-axis label
    plt.ylabel('')  # Add y-axis label
    plt.xticks([])  # Remove x-ticks
    plt.yticks(fontsize=14)  # Increase font size of y labels
    plt.gca().tick_params(left=False)  # Remove y ticks

    # Add percentages inside each section of the stacked bar
    for i in range(len(top_words)):
        for j, col in enumerate(columns_to_plot):
            # Calculate the percentage of the value relative to the total
            value = top_words.iloc[i][col]
            total = top_words.iloc[i][columns_to_plot].sum()
            percentage = f'{(value / total * 100):.0f}%' if total > 0 else ''
            
            # Calculate the position for the percentage text
            if j == 0:
                x_position = value -2.5 # / 2
            else:
                x_position = top_words.iloc[i][columns_to_plot[:j]].sum() + value -5 #/ 2
            
            # Only add text if there's enough space (i.e., percentage is not too small)
            if value / total > 0.1:
                ax.text(x_position, i, percentage, ha='center', va='center', fontsize=12, color='white', weight='bold')

    # Check if the reference line percentage is provided and draw the line
    if reference_line_percent is not None:
        # Draw a horizontal dotted line across the plot
        plt.axvline(x=reference_line_percent, color='black', linestyle='--', linewidth=2)

        # Optionally, add text to indicate the percentage
        plt.text(reference_line_percent +0.5, plt.gca().get_ylim()[1]+.3, f'Pass Frequency ({reference_line_percent}%)',
                 va='center', ha='center', color='black', fontsize=12)

    # Remove the spines (box around the plot)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove the legend
    ax.legend().remove()

    # # Add total count at the end of each bar
    # for i, value in enumerate(top_words[column]):
    #     ax.text(value - 4,  # Slightly offset from the end of the bar
    #             i,  # Y-coordinate (aligned with each bar)
    #             str(f'{int(round(value))}%'),  # The total count for the word
    #             va='center',  # Vertically align in the center of the bar
    #             fontsize=10,
    #             color='white',
    #             weight='bold')  # Consistent font size

    plt.savefig('filename.png', transparent=True)

    # Show the plot
    plt.tight_layout()
    plt.show()



def plot_top_words_freq_small(word_counts, column, top_n=20, figsize=(12, 10), title=None, include_all=True, custom_colors=None, reference_line_percent=None):
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
    top_words = top_words.sort_values(by=column, ascending=False)

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
    plt.xlabel('')  # Add x-axis label
    plt.ylabel('')  # Add y-axis label
    plt.xticks([])  # Remove x-ticks
    plt.yticks(fontsize=14)  # Increase font size of y labels
    plt.gca().tick_params(left=False)  # Remove y ticks

    # Add percentages inside each section of the stacked bar
    for i in range(len(top_words)):
        for j, col in enumerate(columns_to_plot):
            # Calculate the percentage of the value relative to the total
            value = top_words.iloc[i][col]
            total = top_words.iloc[i][columns_to_plot].sum()
            percentage = f'{(value / total * 100):.0f}%' if total > 0 else ''
            
            # Calculate the position for the percentage text
            if j == 0:
                x_position = value -3 #/ 2
            else:
                x_position = top_words.iloc[i][columns_to_plot[:j]].sum() + value -3 #/ 2
            
            # Only add text if there's enough space (i.e., percentage is not too small)
            if value / total > 0.001:
                ax.text(x_position, i, percentage, ha='center', va='center', fontsize=12, color='white', weight='bold')
                
    # Check if the reference line percentage is provided and draw the line
    if reference_line_percent is not None:
        # Draw a horizontal dotted line across the plot
        plt.axvline(x=reference_line_percent, color='black', linestyle='--', linewidth=2)

        # Optionally, add text to indicate the percentage
        plt.text(reference_line_percent +0.5, plt.gca().get_ylim()[1]+.3, f'Fail Frequency ({reference_line_percent}%)',
                 va='center', ha='center', color='black', fontsize=12)

    
    # Remove the spines (box around the plot)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove the legend
    ax.legend().remove()

    # # Add total count at the end of each bar
    # for i, value in enumerate(top_words[column]):
    #     ax.text(value - 4,  # Slightly offset from the end of the bar
    #             i,  # Y-coordinate (aligned with each bar)
    #             str(f'{int(round(value))}%'),  # The total count for the word
    #             va='center',  # Vertically align in the center of the bar
    #             fontsize=10,
    #             color='white',
    #             weight='bold')  # Consistent font size

    plt.savefig('filename.png', transparent=True)
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_legend_only(custom_colors, figsize=(4, 2), fontsize=12, title_fontsize=14, title="Legend"):
    """
    Creates a plot that only displays a legend based on the custom colors provided.

    Parameters:
    custom_colors (dict): A dictionary containing color codes to use for each column.
    figsize (tuple): Width, height in inches of the figure. Defaults to (4, 2).
    fontsize (int): The font size of the legend labels. Defaults to 12.
    title_fontsize (int): The font size of the legend title. Defaults to 14.
    title (str): The title of the legend. Defaults to "Legend".
    """
    # Create figure and axis objects with a smaller figure size
    fig, ax = plt.subplots(figsize=figsize)

    # Create a list of patches to add to the legend
    patches = [mpatches.Patch(color=color, label=label) for label, color in custom_colors.items()]

    # Add the legend to the plot with an increased font size
    legend = ax.legend(handles=patches, loc='center', fontsize=fontsize, title=title)
    plt.setp(legend.get_title(), fontsize=title_fontsize)

    # Hide the axis
    ax.axis('off')

    # Tight layout to minimize white space
    plt.tight_layout()

    # Show only the legend
    plt.show()

# # Example usage:
# custom_colors = {'A': 'green', 'B': 'orange', 'C': 'red', 'All': 'grey'}
# plot_legend_only(custom_colors)



def plot_legend_onlyv2(custom_colors, figsize=(4, 2), fontsize=12, title_fontsize=14, title="Legend"):

    fig, ax = plt.subplots(figsize=figsize)
    patches = [mpatches.Patch(color=color, label=label) for label, color in custom_colors.items()]
    legend = ax.legend(handles=patches, loc='center', fontsize=fontsize, title=title, frameon=False)
    plt.setp(legend.get_title(), fontsize=title_fontsize)
    ax.axis('off')
    plt.tight_layout()
    
    plt.savefig('filename.png', transparent=True)
    
    plt.show()

# # Example usage:
# custom_colors = {'A': '#1a472a', 'B': '#CCCCCC', 'C': '#B7B7B7'}
# plot_legend_only(custom_colors)
