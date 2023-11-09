#imported libs
import acquire as a
import prepare as p

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.rename(columns={'concatenated_reviews': 'reviews'})
    ny_reviews = ny_reviews.dropna()
    
    X = ny_reviews.reviews # add new features
    y = ny_reviews.grade
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    train_baseline_acc = y_train.value_counts().max() / y_train.shape[0] * 100    
    val_baseline_acc = y_val.value_counts().max() / y_val.shape[0] * 100

    print(f'\nBaseline Accuracy')
    print(f'==================================================')
    print(f'\n\nTrain baseline accuracy: {round(train_baseline_acc)}%\n')
    print(f'\nValidation baseline accuracy: {round(val_baseline_acc)}%\n')









# logistic regression function function

# ===========================================================================================================================================

def model_1():
    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.rename(columns={'concatenated_reviews': 'reviews'})
    ny_reviews = ny_reviews.dropna()

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
    lm = LogisticRegression(
        penalty='l2',
        C=1.0,
        fit_intercept=False,
        class_weight='balanced',
        solver='liblinear',
        max_iter=100,
        random_state=42
    )
    lm.fit(X_train_tfidf, y_train)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': lm.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': lm.predict(X_val_tfidf)})
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    train_classification_report = class_rep(y_train, y_train_pred)
    val_classification_report = class_rep(y_val, y_val_pred)

    print(f'\nLogisitic Regression Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nClassification Report for Training Set:\n\n{train_classification_report}\n')
    print(f'\nClassification Report for Validation Set:\n\n{val_classification_report}\n')








# knn regressor function

# ===========================================================================================================================================

def model_2():
    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.rename(columns={'concatenated_reviews': 'reviews'})
    ny_reviews = ny_reviews.dropna()

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
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    print(f'\nKNearest Neighbors (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')



# xgbclassifier function

# ===========================================================================================================================================

def model_3():

    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.rename(columns={'concatenated_reviews': 'reviews'})
    ny_reviews = ny_reviews.dropna()
    
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
    preds_decoded = label_encoder.inverse_transform(preds)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': bst.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': bst.predict(X_val_tfidf)})
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    print(f'\nXGBClassifier Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')










# logistic regression w/hyperparamter tuning function

# ===========================================================================================================================================

def model_4():
    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews.csv', index_col=0)
    ny_reviews = ny_reviews.rename(columns={'concatenated_reviews': 'reviews'})
    ny_reviews = ny_reviews.dropna()

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

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': lm.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': lm.predict(X_val_tfidf)})
    y_test_res = pd.DataFrame({'actual': y_test, 'preds': lm.predict(X_test_tfidf)})
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])
    test_accuracy = accuracy_score(y_test_res['actual'], y_test_res['preds'])

    print(f'\nFinal Model Logisitic Regression with Hyperparameter tuning')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nTest Accuracy: {test_accuracy:.2f}\n')
