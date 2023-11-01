import pandas as pd
import numpy as np


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

    return ny  # Return clean dataframe
