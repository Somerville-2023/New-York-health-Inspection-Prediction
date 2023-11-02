import pandas as pd
import numpy as np
import acquire as a

def remove_columns(ny, trash_columns=['bin', 'bbl', 'nta', 'census_tract', 'council_district', 'community_board',
                                      'grade_date', 'critical_flag', 'inspection_type']):
    ny = ny.drop(columns=trash_columns)
    return ny


def clean_phones(ny):
    # Clean phone numbers by removing non-digit characters and dropping nulls
    ny.phone = ny.phone.str.replace(' ', '')
    ny.phone = ny.phone.str.replace('_', '')
    ny = ny[ny.phone.notna()]
    return ny


def clean_zipcodes(ny):
    # Clean zipcodes by filling nulls with 0 and then converting to integers
    ny.zipcode = ny.zipcode.fillna(0)
    ny.zipcode = ny.zipcode.astype(int)
    ny = ny[ny.zipcode.notna()]  # Drop nulls
    return ny


def clean_streets(ny):
    # Remove nulls from street
    ny = ny[ny.street.notna()]
    return ny


def clean_scores(data):
    ny = data.copy()
    ny = ny[ny.inspection_date != '1900-01-01T00:00:00.000']  # Remove all values with no inspections done

    # Create a new list of scores that replaces null scores for no violation for 0s
    new_scores = []  # Empty list
    for score, rep in zip(ny.score, ny.action.str.contains('No violation')):  # Loop through 2 iterable values
        if rep:  # If no violation, append score 0
            new_scores.append(0)
        else:  # Else keep score the same
            new_scores.append(score)
    ny.score = new_scores

    ny = ny[ny.score.notna()]
    return ny


def clean_actions(ny):
    # Remove nulls from action
    ny = ny[ny.action.notna()]
    # Rename actions to something more concise
    ny.action = np.where(ny.action == 'Violations were cited in the following area(s).', 'Violations cited', ny.action)
    ny.action = np.where(ny.action == 'Establishment Closed by DOHMH. Violations were cited in the following area(s) '
                                      'and those requiring immediate action were addressed.', 'Closed', ny.action)
    ny.action = np.where(ny.action == 'Establishment re-opened by DOHMH.', 'Re-opened', ny.action)
    ny.action = np.where(ny.action == 'No violations were recorded at the time of this inspection.', 'No violations',
                         ny.action)
    return ny


def clean_grades(data):
    ny = data.copy()  # Create copy of df
    # Create empty list to hold new values for restaurant
    new_grades = []
    # Use scores to determine grades
    for grade, score in zip(ny.grade, ny.score):
        if score <= 13:
            new_grades.append('A')
        elif score <= 27:
            new_grades.append('B')
        elif score > 27:
            new_grades.append('C')
    ny.grade = new_grades
    return ny


def clean_violations(data):
    ny = data.copy()
    # Create empty lists
    new_codes = []
    new_description = []
    # Loop through actions and violations
    for action, code, description in zip(ny.action, ny.violation_code, ny.violation_description):
        if action == 'No violations':  # If there is no violations, append no violations to code and description
            new_codes.append('No violation')
            new_description.append('No violation')
        else:
            new_codes.append(code)
            new_description.append(description)

    # Replace df values with new ones
    ny.violation_code = new_codes
    ny.violation_description = new_description

    return ny  # Return data


def clean_ny(ny):
    """This function just takes in all other cleaning functions for ny data and cleans each element of it"""

    ny = remove_columns(ny)  # Removes useless columns from ny health inspection data

    ny = clean_phones(ny)  # Clean phone numbers

    ny = clean_zipcodes(ny)  # Cleans zip codes

    ny = clean_streets(ny)  # Cleans streets

    ny = clean_scores(ny)  # Cleans scores

    ny = clean_actions(ny)  # Cleans actions

    ny = clean_grades(ny)  # Cleans grades

    ny = clean_violations(ny)  # Cleans violation codes and descriptions

    ny = ny.dropna()  # Drops all remaining null values

    ny = ny.reset_index(drop=True)  # Reset the index and drop the old index

    ny['inspection_date'] = pd.to_datetime(ny['inspection_date'])

    return ny  # Return clean dataframe


def aggregate_violations(ny):
    """This function will aggregate all rows for each inspection for each restaurant into on row by combining the
       violations."""
    # Create aggregated df indexed by camis and inspection_date
    agg_violations = ny.groupby(['camis', 'inspection_date']).agg({'violation_code': lambda x: x.tolist(),
                                                                   'violation_description': lambda x: x.tolist()})
    # Create separate df without code & description
    ny2 = ny.drop(columns=['violation_code', 'violation_description']).copy()
    ny2 = ny2.drop_duplicates()  # Drop duplicates

    # Create empty lists
    agg_data_code = []
    agg_data_description = []

    # Loop through df without duplicates and create lists of aggregated violations
    for cam, date in zip(ny2.camis, ny2.inspection_date):
        agg_data_code.append(agg_violations.loc[(cam, date)][0])
        agg_data_description.append(agg_violations.loc[(cam, date)][1])

    # Insert new, aggregated violations into df
    ny2['violation_code'] = agg_data_code
    ny2['violation_description'] = agg_data_description

    return ny2  # Return df


def clean_code(ny):
    """This function removes 'No violation' from the rows that shouldn't have it. Some rows contained both violation
    codes and 'No violation'."""
    # Create empty lists
    clean_codes = []
    clean_description = []

    # Loop through lists and remove 'No violation' if there are more than one element in each list
    for row1, row2 in zip(ny.violation_code, ny.violation_description):

        code_list1 = row1
        code_list2 = row2

        if len(code_list1) > 1 and 'No violation' in code_list1:
            code_list1.remove('No violation')
            clean_codes.append(code_list1)
        else:
            clean_codes.append(code_list1)

        if len(code_list2) > 1 and 'No violation' in code_list2:
            code_list2.remove('No violation')
            clean_description.append(code_list2)
        else:
            clean_description.append(code_list2)

    # Reassign new data to dataframe
    ny.violation_code = clean_codes
    ny.violation_description = clean_description

    return ny  # Return df


def join_lists(ny):
    """This function joins all the contents of the lists in code, and description into one string."""

    # Create empty lists
    joined_code = []
    joined_description = []

    # Join violation codes with a ' ' between elements
    for row in ny.violation_code:
        joined_code.append(' '.join(row))

    # Join violation description with a ' ' between elements
    for row in ny.violation_description:
        joined_description.append(' '.join(row))

    ny.violation_code = joined_code
    ny.violation_description = joined_description

    return ny  # Return df


def final_ny():
    """This function just combines all the previous functions into one. It will acquire and process the data."""
    ny = a.acquire_ny()  # Acquire data, from local .csv file or api request if no .csv file is present

    ny = clean_ny(ny)  # Cleans the data

    # Aggregates the data into one row per inspection and compiles the violation data into a list per row
    ny = aggregate_violations(ny)

    ny = clean_code(ny)  # Removes 'No violation' from lists it shouldn't be in

    ny = join_lists(ny)  # Unpacks (combines) lists into one string

    ny['inspection_date'] = pd.to_datetime(ny['inspection_date'])


    return ny  # Return df