import warnings
warnings.filterwarnings("ignore")

# import libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats




# ==============================================STATS TEST EVALUATION FUNCTION==============================================

def eval_t_test_result(p_value, a=0.05):
    if p_value < a:
        result = "Reject the null hypothesis!\nThere is no significant difference in health inspection scores between the Bronx and Queens."
    else:
        result = "Failed to reject the null hypothesis!\nThere is a significant difference in health inspection scores between the Bronx and Queens."
    
    return result

def eval_correlation_result(p_value, a=0.05):
    if p_value < a:
        result = "Reject the null hypothesis!\nThere is a statistically significant correlation between inspection scores and inspection dates."
    else:
        result = "Failed to reject the null hypothesis!\nThere is no statistically significant correlation between inspection scores and inspection dates."
    
    return result


# ==============================================DATA DISTRIBUTION FUNCTION==============================================
def data_distribution(data):
    '''
    Create a histogram plot to visualize the distribution of data.

    Args:
        data (DataFrame): The DataFrame containing the data.
    '''

    # set the font and style
    sns.set(font_scale=1, style="white")

    # histplot visual binned by 25
    sns.histplot(data, x='score', bins=25)
    
    # Edits to histplot for a more appealing view of data
    plt.title('Distribution of data by Scores', fontsize=15)
    plt.xlabel('Score', labelpad=20)
    plt.ylabel('Count', rotation=0, labelpad=30)
    plt.xlim(0, 80)
    plt.ylim(0,28000)
    plt.show()

# ==============================================TOP 20 BUSINESSES BY COUNT FUNCTION==============================================

def visual_1(data):
    '''
    Create a barplot to visualize the Top 20 businesses by count

    Args:
        data (Dataframe): The DataFrame containing the data.
    '''

    # set the font and style
    sns.set(font_scale=4, style='white')
    
    top_n = 20  # Adjust the number of top values to display
    
    # Create a figure and specify the desired figure size
    plt.figure(figsize=(36, 20))
    
    # Create the countplot with y as the 'dba' and x as the count
    countplot = sns.countplot(data, y='cuisine_description', order=data['cuisine_description'].value_counts().iloc[:top_n].index)
    
    # Increase the font size for the title and y-axis labels
    countplot.set_yticklabels(countplot.get_yticklabels(), fontsize=25)  # Adjust the fontsize as needed
    plt.title(f'Top {top_n} Cuisine by count', fontsize=50)  # Adjust the fontsize as needed
    plt.ylabel('')
    plt.xlabel('')
    plt.show()

# ============================================== TOP 20 FAILINING BUSINESSES SCORE/ACTION BY BORO FUNCTION==============================================

# Note: input markdown of why bronx and queens was choosen

def visual_2(data):
    '''
    Create a barplot to visualize the Top 20 businesses poorly scored on health inspection 
    and actions taken by borough in New York

    Args:
        data (Dataframe): The DataFrame containing the data.
    '''

    # Choose the two boroughs you want to visualize
    selected_boros = ['Queens', 'Bronx']
    
    # Filter the DataFrame to include only data for the selected boroughs
    filtered_data = data[data['boro'].isin(selected_boros)]
        
    # assign he top 20 businesses by largest score
    top_20_scores = filtered_data.nlargest(20, 'score')

    # barplot visual
    sns.set(font_scale=1.5, style="white")
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.barplot(data=top_20_scores, x='boro', y='score', hue='action', color='red', ci=False)
    plt.xlabel('')
    plt.ylabel('Score')
    plt.title('Bar Plot of Top 20 Scores by Borough')
    plt.xticks(rotation=0)  # Rotate x-axis labels for better visibility
    plt.show()

    # scores for the bronx and queens assigned to respective variable
    scores_bronx = pd.Series(top_20_scores.boro == 'Bronx')
    scores_queens = pd.Series(top_20_scores.boro == 'Queens')

    # Perform the t-test
    t_statistic, t_test_p_value = stats.ttest_ind(scores_bronx, scores_queens)
    
    # Print the results
    print(f"t-statistic: {t_statistic}")
    print(f"P-value: {t_test_p_value}\n")
    eval_t_test_result(t_test_p_value)

def visual_3(data):

    # Convert inspection_date to numeric format (e.g., seconds since the epoch)
    ny['inspection_date_numeric'] = ny['inspection_date'].view('int64') // 10**9  # Convert to seconds
    
    # set font and style
    sns.set(font_scale=1.25, style="white")
    
    # color corrected the visual
    custom_palette = ["red", "orange", "blue", "mediumseagreen"]
    
    # Specify the order and palette for 'action' categories
    hue_order = ['Closed', 'Violations cited', 'Re-opened', 'No violations']  # Specify your category names
    sns.scatterplot(data=ny, x='score', y='inspection_date', hue='action', hue_order=hue_order, palette=custom_palette)
    plt.xlabel('Score', labelpad=20)
    plt.ylabel('Years')
    plt.title('Health Inspection Scores and Action over time')
    plt.show()

    # Calculate the Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = stats.pearsonr(ny['inspection_date_numeric'], ny['score'])
    
    print(f"Pearson Correlation Coefficient: {correlation_coefficient}\n")
    print(f"P-value: {p_value}")

    eval_t_test_result(t_test_p_value)

    # Perform t-test and evaluate the result
t_test_p_value = perform_t_test()  # Replace with your actual t-test function
t_test_result = eval_t_test_result(t_test_p_value)
print("T-Test Result:")
print(t_test_result)

# Perform correlation test and evaluate the result
correlation_p_value = perform_correlation_test()  # Replace with your actual correlation test function
correlation_result = eval_correlation_result(correlation_p_value)
print("Correlation Test Result:")
print(correlation_result)


    print(f"\nThere is a statistically significant correlation between inspection scores and inspection dates.")
