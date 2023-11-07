#imported libs
import acquire as a
import prepare as p

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

from sklearn.metrics import classification_report as class_rep







# baseline function

# ===========================================================================================================================================

def baseline():

    grademap = {'A': 'Pass', 'B': 'Pass', 'C': 'Fail'}

    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.dropna()
    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    X = ny_reviews.reviews  # Features
    y = ny_reviews.grade  # Target labels

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    train_baseline_acc = y_train.value_counts().min() / y_train.shape[0] * 100
    val_baseline_acc = y_val.value_counts().min() / y_val.shape[0] * 100

    # Get the minority class label
    minority_class = y_train.value_counts().idxmin()

    # Predict using the minority class for both training and validation
    y_train_pred = [minority_class] * len(y_train)
    y_val_pred = [minority_class] * len(y_val)

    train_classification_report = class_rep(y_train, y_train_pred)
    val_classification_report = class_rep(y_val, y_val_pred)

    print(f'\nBaseline Model (Minority Class)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_baseline_acc:.2f}%\n')
    print(f'\nValidation Accuracy: {val_baseline_acc:.2f}%\n')
    print(f'\nClassification Report for Training Set:\n{train_classification_report}\n')
    print(f'\nClassification Report for Validation Set:\n{val_classification_report}\n')










# logistic regression function function

# ===========================================================================================================================================

def model_1():

    grademap = {'A': 'Pass', 'B': 'Pass', 'C': 'Fail'}

    
    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.dropna()
    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)



    X = ny_reviews.reviews # add new features
    y = ny_reviews.grade

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    # Train a logistic regression model
    lm = LogisticRegression(random_state=42)
    lm.fit(X_train_tfidf, y_train)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': lm.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': lm.predict(X_val_tfidf)})

    y_train_pred = lm.predict(X_train_tfidf)
    y_val_pred = lm.predict(X_val_tfidf)

    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    train_classification_report = class_rep(y_train, y_train_pred)
    val_classification_report = class_rep(y_val, y_val_pred)

    print(f'\nLogisitic Regression Model')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nClassification Report for Training Set:\n{train_classification_report}\n')
    print(f'\nClassification Report for Validation Set:\n{val_classification_report}\n')








# knn regressor function

# ===========================================================================================================================================

def model_2():
    grademap = {'A': 'Pass', 'B': 'Pass', 'C': 'Fail'}

    
    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.dropna()
    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    X = ny_reviews.reviews# add new features
    y = ny_reviews.grade

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    # Train KNN Model
    knn = KNeighborsClassifier(
    n_neighbors=2,  
    weights='distance',  # distance
    p=2,  # Euclidean distance
    algorithm='auto',  # 'ball_tree', 'kd_tree', or 'brute'
    leaf_size=30,  
    metric='euclidean'  # You can choose other metrics or provide custom ones
    )
    knn.fit(X_train_tfidf, y_train)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': knn.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': knn.predict(X_val_tfidf)})

    y_train_pred = knn.predict(X_train_tfidf)
    y_val_pred = knn.predict(X_val_tfidf)
    
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])
    
    train_classification_report = class_rep(y_train, y_train_pred)
    val_classification_report = class_rep(y_val, y_val_pred)

    print(f'\nKNearest Neighbors (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nClassification Report for Training Set:\n{train_classification_report}\n')
    print(f'\nClassification Report for Validation Set:\n{val_classification_report}\n')



# xgbclassifier function

# ===========================================================================================================================================

target_names = ['Fail', 'Pass']

def model_3():

    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.dropna()

    grademap = {'A': 'Pass', 'B': 'Pass', 'C': 'Fail'}


    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)
    
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    
    # Encode the target labels
    y_encoded = label_encoder.fit_transform(ny_reviews.grade)
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(ny_reviews.reviews, y_encoded, train_size=0.7, random_state=42) # add new features
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize and fit the TfidfVectorizer on the training data
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Create the XGBoost classifier instance
    bst = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.25, objective='multi:softprob', num_class=len(label_encoder.classes_))
    
    # Fit the XGBoost model on the training data
    bst.fit(X_train_tfidf, y_train)
    
    # Predict the classes on the validation data
    preds = bst.predict(X_val_tfidf)
    
    # If you want to decode the predicted labels back to their original class names:
    # Convert one-hot encoded labels back to original class labels
    preds_decoded = label_encoder.inverse_transform(preds.argmax(axis=1))

    # Calculate scores
    # Flatten the predictions to a 1D array
    train_preds = bst.predict(X_train_tfidf)
    train_preds_flattened = train_preds.argmax(axis=1)
    
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': train_preds_flattened})
    
    val_preds = bst.predict(X_val_tfidf)
    val_preds_flattened = val_preds.argmax(axis=1)
    
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': val_preds_flattened})
    
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])
    
    train_classification_report = class_rep(y_train, y_train_res['preds'], target_names=target_names)
    val_classification_report = class_rep(y_val, y_val_res['preds'], target_names=target_names)

    print(f'\nXGBClassifier Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')

    print(f'\nClassification Report for Training Set:\n{train_classification_report}\n')
    print(f'\nClassification Report for Validation Set:\n{val_classification_report}\n')








# logistic regression w/hyperparamter tuning function

# ===========================================================================================================================================

def model_4():
    
    grademap = {'A': 'Pass', 'B': 'Pass', 'C': 'Fail'}

    
    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.dropna()
    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    X = ny_reviews.reviews # add new features
    y = ny_reviews.grade


    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(use_idf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    lm = LogisticRegression(
    penalty='l2',  # L2 regularization (Ridge)
    C=1.0,  # Inverse of regularization strength
    fit_intercept=False,  # Include an intercept
    class_weight='balanced',  # You can set class weights if needed
    solver='liblinear',  # Choose a solver appropriate for your data
    max_iter=100,  # You may need to increase this if the model doesn't converge
    random_state=42  # For reproducibility
    )
    
    lm.fit(X_train_tfidf, y_train)

     # Calculate scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': lm.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': lm.predict(X_val_tfidf)})

    y_train_pred = lm.predict(X_train_tfidf)
    y_val_pred = lm.predict(X_val_tfidf)

    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    train_classification_report = class_rep(y_train, y_train_pred)
    val_classification_report = class_rep(y_val, y_val_pred)

    print(f'\nLogisitic Regression Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nClassification Report for Training Set:\n{train_classification_report}\n')
    print(f'\nClassification Report for Validation Set:\n{val_classification_report}\n')







# rf classifier/Gradient Boosting function  ===========================================================================================================================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def model_5():
    grademap = {'A': 'Pass', 'B': 'Pass', 'C': 'Fail'}

    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.dropna()
    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    X = ny_reviews.reviews
    y = ny_reviews.grade

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(use_idf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    # Define the hyperparameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }

    # Create a Random Forest model
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Perform grid search
    rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, scoring='accuracy')
    rf_grid_search.fit(X_train_tfidf, y_train)

    # Get the best Random Forest model
    rf = rf_grid_search.best_estimator_
    
    # Define the hyperparameter grid for Gradient Boosting
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3],
        'subsample': [0.8, 0.9, 1.0],
    }

    # Create a Gradient Boosting model
    gb_model = GradientBoostingClassifier(random_state=42)

    # Perform grid search
    gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=3, scoring='accuracy')
    gb_grid_search.fit(X_train_tfidf, y_train)

    # Get the best Gradient Boosting model
    gb = gb_grid_search.best_estimator

    # Calculate scores for Random Forest
    y_train_pred_rf = rf.predict(X_train_tfidf)
    y_val_pred_rf = rf.predict(X_val_tfidf)

    train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
    val_accuracy_rf = accuracy_score(y_val, y_val_pred_rf)

    # Calculate scores for Gradient Boosting
    y_train_pred_gb = gb.predict(X_train_tfidf)
    y_val_pred_gb = gb.predict(X_val_tfidf)

    train_accuracy_gb = accuracy_score(y_train, y_train_pred_gb)
    val_accuracy_gb = accuracy_score(y_val, y_val_pred_gb)

    # Print classification reports
    print(f'\nRandom Forest Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy_rf:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy_rf:.2f}\n')
    print(f'\nClassification Report for Random Forest (Training Set):\n{classification_report(y_train, y_train_pred_rf)}\n')
    print(f'\nClassification Report for Random Forest (Validation Set):\n{classification_report(y_val, y_val_pred_rf)}\n')

    print(f'\nGradient Boosting Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy_gb:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy_gb:.2f}\n')
    print(f'\nClassification Report for Gradient Boosting (Training Set):\n{classification_report(y_train, y_train_pred_gb)}\n')
    print(f'\nClassification Report for Gradient Boosting (Validation Set):\n{classification_report(y_val, y_val_pred_gb)}\n')

