import pandas as pd
import numpy as np
import re
import nick_acquire as a


def remove_columns(ny, trash_columns=['bin', 'bbl', 'nta', 'census_tract', 'council_district', 'community_board',
                                      'grade_date', 'critical_flag', 'inspection_type', 'record_date']):
    ny = ny.drop(columns=trash_columns)
    return ny


def clean_phones(data):
    ny = data.copy()

    ny = ny[ny.phone.notna()]

    new_phone = []

    for phone in ny.phone:
        new_phone.append(re.sub(r'\D', '', phone))
    ny.phone = new_phone

    newer_phones = [phone if len(phone) > 1 else '0' for phone in ny.phone]

    ny.phone = newer_phones

    ny['phone'] = pd.to_numeric(ny['phone'], errors='coerce')
    # Convert it to an integer
    ny['phone'] = ny['phone'].astype(int)
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
    # new_scores = []  # Empty list
    # for score, rep in zip(ny.score, ny.action.str.contains('No violation')):  # Loop through 2 iterable values
    #    if rep:  # If no violation, append score 0
    #        new_scores.append(0)
    #    else:  # Else keep score the same
    #        new_scores.append(score)
    # ny.score = new_scores

    ny = ny[ny.score.notna()]

    ny.score = ny.score.astype(int)
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


def combine_address(ny):
    """This function combines the addresses of the restaurants into one single feature."""
    full_addy = ny.building + ' ' + ny.street + ' ' + ny.zipcode.astype(str)  # Concat the address together
    ny['full_address'] = full_addy  # Create new feature
    ny = ny.drop(columns=['building', 'street', 'zipcode'])  # Drop old features
    return ny  # Return df


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

    ny = combine_address(ny)

    ny = ny.reset_index(drop=True)

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

    return ny2


def clean_code(ny):
    clean_codes = []
    clean_description = []

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

    ny.violation_code = clean_codes
    ny.violation_description = clean_description

    return ny


def join_lists(ny):
    joined_code = []
    joined_description = []

    for row in ny.violation_code:
        joined_code.append(' '.join(row))

    for row in ny.violation_description:
        joined_description.append(' '.join(row))

    ny.violation_code = joined_code
    ny.violation_description = joined_description

    return ny


def final_ny():
    ny = a.acquire_ny()
    ny = clean_ny(ny)

    ny = aggregate_violations(ny)
    ny = clean_code(ny)
    ny = join_lists(ny)
    return ny
