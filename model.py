#Hush warning/note messages
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


#imported libs
import acquire as a
import prepare as p

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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





# Visual functions for model:

def plot_accuracy_comparison(train_accuracy, val_accuracy, test_accuracy, baseline_train_accuracy, baseline_val_accuracy):
    sns.set_theme(font_scale=1.25, style="white")

    # Data
    models = ['Baseline Train', 'Baseline Validation', 'Final Train', 'Final Validation', 'Final Test']
    accuracy_scores = [baseline_train_accuracy, baseline_val_accuracy, train_accuracy, val_accuracy, test_accuracy]

    # Custom colors for each bar
    custom_colors = ['darkgray', 'darkgray', '#1F77B4', '#1F77B4', '#1F77B4']

    # Create bar chart with custom colors
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x=models, y=accuracy_scores, palette=custom_colors, edgecolor='black')
    plt.ylim(0, 1)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    
    # Remove y-axis spines, ticks, and labels
    sns.despine(left=True)
    plt.yticks([])

    # Add percentage labels within the bars
    for i, v in enumerate(accuracy_scores):
        barplot.text(i, v/2, f'{v*100:.0f}%', ha='center', va='center', color='black')

    plt.savefig('models_plot.png', transparent=True)
    plt.show()


    # # Save the figure with transparent background
    # plt.savefig('models_plot.png', transparent=True)




# baseline function

# ===========================================================================================================================================

def baseline():

    grademap = {'A': 'Pass', 'B': 'Fail', 'C': 'Fail'}

    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews_sentiment_ratings.csv', index_col=0)
    ny_reviews = ny_reviews.rename(columns={'neg' : 'negative',
                                            'neu' : 'neutral',
                                            'pos' : 'positive',})
    ny_reviews = ny_reviews[['grade', 'avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'reviews', 'negative', 'neutral', 'positive', 'compound']]
    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    X = ny_reviews.drop(columns=["grade"]) # Features
    y = ny_reviews["grade"] # Target labels

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    tfidf = TfidfVectorizer()
    X_train_reviews_tfidf = tfidf.fit_transform(X_train["reviews"])
    X_val_reviews_tfidf = tfidf.transform(X_val["reviews"])
    X_test_reviews_tfidf = tfidf.transform(X_test["reviews"])

    # Combine TF-IDF vectors with sentiment features
    X_train = hstack([X_train_reviews_tfidf, X_train[['negative', 'neutral', 'positive', 'compound']].values]) #'negative', 'neutral', 'positive', 'compound'
    X_val = hstack([X_val_reviews_tfidf, X_val[['negative', 'neutral', 'positive', 'compound']].values]) #'avg_service', 'avg_atmosphere', 'avg_food', 'avg_price'
    X_test = hstack([X_test_reviews_tfidf, X_test[['negative', 'neutral', 'positive', 'compound']].values])

    train_baseline_acc = y_train.value_counts().max() / y_train.shape[0] * 100    
    val_baseline_acc = y_val.value_counts().max() / y_val.shape[0] * 100

    # train_classification_report = class_rep(y_train, y_train_pred)
    # val_classification_report = class_rep(y_val, y_val_pred)

    print(f'\nBaseline Model')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_baseline_acc:.2f}%\n')
    print(f'\nValidation Accuracy: {val_baseline_acc:.2f}%\n')

    # print(f'\nClassification Report for Training Set:\n\n{train_classification_report}\n\n\n')
    # print(f'\nClassification Report for Validation Set:\n\n{val_classification_report}\n')








# logistic regression function function

# ===========================================================================================================================================

def model_1():
    grademap = {'A': 'Pass', 'B': 'Fail', 'C': 'Fail'}

    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews_sentiment_ratings.csv', index_col=0)
    ny_reviews = ny_reviews.rename(columns={'neg': 'negative', 'neu': 'neutral', 'pos': 'positive'})
    ny_reviews = ny_reviews[['grade', 'avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'reviews', 'negative', 'neutral', 'positive', 'compound']]
    
    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    X = ny_reviews.drop(columns=["grade"])  # Features
    y = ny_reviews["grade"]  # Target labels

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    tfidf = TfidfVectorizer()
    X_train_reviews_tfidf = tfidf.fit_transform(X_train["reviews"])
    X_val_reviews_tfidf = tfidf.transform(X_val["reviews"])
    X_test_reviews_tfidf = tfidf.transform(X_test["reviews"])

    # Combine TF-IDF vectors with sentiment features
    X_train = hstack([X_train_reviews_tfidf, X_train[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])
    X_val = hstack([X_val_reviews_tfidf, X_val[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])
    X_test = hstack([X_test_reviews_tfidf, X_test[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])

    # Train a logistic regression model
    lm = LogisticRegression(random_state=42, max_iter=1000)
    lm.fit(X_train, y_train)

    # Calculate accuracy scores
    y_train_pred = lm.predict(X_train)
    y_val_pred = lm.predict(X_val)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    train_classification_report = class_rep(y_train, y_train_pred)
    val_classification_report = class_rep(y_val, y_val_pred)

    print('\nLogistic Regression Model')
    print('==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nClassification Report for Training Set:\n\n{train_classification_report}\n\n\n')
    print(f'\nClassification Report for Validation Set:\n\n{val_classification_report}\n')









# # knn regressor function

# # ===========================================================================================================================================

# def model_2():

#     ny_reviews = pd.read_csv('ny_reviews_sentiment_ratings.csv', index_col=0)
#     grademap = {'A': 'Pass', 'B': 'Pass', 'C': 'Fail'}

#     ny_reviews = ny_reviews.rename(columns={'neg' : 'negative',
#                                             'neu' : 'neutral',
#                                             'pos' : 'positive',})

#     ny_reviews = ny_reviews[['grade', 'avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'reviews', 'negative', 'neutral', 'positive', 'compound']]
    
#     # Load and preprocess your data
#     ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

#     X = ny_reviews.drop(columns=["grade"])  # Features
#     y = ny_reviews["grade"]  # Target labels

#     # Split the data into training, validation, and test sets
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     tfidf = TfidfVectorizer()
#     X_train_reviews_tfidf = tfidf.fit_transform(X_train["reviews"])
#     X_val_reviews_tfidf = tfidf.transform(X_val["reviews"])
#     X_test_reviews_tfidf = tfidf.transform(X_test["reviews"])

#     # Combine TF-IDF vectors with sentiment features
#     X_train = hstack([X_train_reviews_tfidf, X_train[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])
#     X_val = hstack([X_val_reviews_tfidf, X_val[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])
#     X_test = hstack([X_test_reviews_tfidf, X_test[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])

#     # Train KNN Model
#     knn = KNeighborsClassifier(
#     n_neighbors=2,  
#     weights='distance',  # distance
#     p=2,  # Euclidean distance
#     algorithm='auto',  # 'ball_tree', 'kd_tree', or 'brute'
#     leaf_size=30,  
#     metric='euclidean'  # You can choose other metrics or provide custom ones
#     )
#     knn.fit(X_train, y_train)

#     # Calculate accuracy scores
#     y_train_res = pd.DataFrame({'actual': y_train, 'preds': knn.predict(X_train)})
#     y_val_res = pd.DataFrame({'actual': y_val, 'preds': knn.predict(X_val)})

#     y_train_pred = knn.predict(X_train)
#     y_val_pred = knn.predict(X_val)
    
#     train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
#     val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])
    
#     train_classification_report = class_rep(y_train, y_train_pred)
#     val_classification_report = class_rep(y_val, y_val_pred)

#     print(f'\nKNearest Neighbors (Hyperparameters Used)')
#     print(f'==================================================')
#     print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
#     print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
#     print(f'\nClassification Report for Training Set:\n\n{train_classification_report}\n\n\n')
#     print(f'\nClassification Report for Validation Set:\n\n{val_classification_report}\n')



# xgbclassifier function

# ===========================================================================================================================================

target_names = ['Fail', 'Pass']

def model_2():
    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews_sentiment_ratings.csv', index_col=0)
    grademap = {'A': 'Pass', 'B': 'Pass', 'C': 'Fail'}
    
    ny_reviews = ny_reviews.rename(columns={'neg': 'negative', 'neu': 'neutral', 'pos': 'positive'})
    ny_reviews = ny_reviews[['grade', 'avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'reviews', 'negative', 'neutral', 'positive', 'compound']]
    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Encode the target labels
    y_encoded = label_encoder.fit_transform(ny_reviews.grade)

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(ny_reviews[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'reviews', 'negative', 'neutral', 'positive', 'compound']], y_encoded, train_size=0.7, random_state=42)  # Combine features
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Initialize and fit the TfidfVectorizer on the training data
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train['reviews'])
    X_val_tfidf = tfidf.transform(X_val['reviews'])
    X_test_tfidf = tfidf.transform(X_test['reviews'])

    # Combine TF-IDF vectors with sentiment features for training data
    X_train_combined = hstack([X_train_tfidf, X_train[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']]])

    # Combine TF-IDF vectors with sentiment features for validation data
    X_val_combined = hstack([X_val_tfidf, X_val[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']]])

    # Create the XGBoost classifier instance
    bst = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.25, objective='multi:softprob', num_class=len(label_encoder.classes_))

    # Fit the XGBoost model on the combined features of the training data
    bst.fit(X_train_combined, y_train)

    # Predict the classes on the validation data
    preds = bst.predict(X_val_combined)

    # If you want to decode the predicted labels back to their original class names:
    # Convert one-hot encoded labels back to original class labels
    preds_decoded = label_encoder.inverse_transform(preds.argmax(axis=1))

    # Calculate scores
    # Flatten the predictions to a 1D array
    train_preds = bst.predict(X_train_combined)
    train_preds_flattened = train_preds.argmax(axis=1)

    y_train_res = pd.DataFrame({'actual': y_train, 'preds': train_preds_flattened})

    val_preds = bst.predict(X_val_combined)
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

    print(f'\nClassification Report for Training Set:\n\n{train_classification_report}\n')
    print(f'\nClassification Report for Validation Set:\n\n{val_classification_report}\n')









# logistic regression w/hyperparamter tuning function

# ===========================================================================================================================================

def model_3():
    
    grademap = {'A': 'Pass', 'B': 'Fail', 'C': 'Fail'}

    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews_sentiment_ratings.csv', index_col=0)
    ny_reviews = ny_reviews.rename(columns={'neg': 'negative', 'neu': 'neutral', 'pos': 'positive'})
    ny_reviews = ny_reviews[['grade', 'avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'reviews', 'negative', 'neutral', 'positive', 'compound']]

    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    X = ny_reviews.drop(columns=["grade"])  # Features
    y = ny_reviews["grade"]  # Target labels

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    tfidf = TfidfVectorizer()
    X_train_reviews_tfidf = tfidf.fit_transform(X_train["reviews"])
    X_val_reviews_tfidf = tfidf.transform(X_val["reviews"])
    X_test_reviews_tfidf = tfidf.transform(X_test["reviews"])

    # Combine TF-IDF vectors with sentiment features
    X_train = hstack([X_train_reviews_tfidf, X_train[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])
    X_val = hstack([X_val_reviews_tfidf, X_val[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])
    X_test = hstack([X_test_reviews_tfidf, X_test[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])

    # set log reg hyperparameters
    lm = LogisticRegression(
    penalty='l2',  # L2 regularization (Ridge)
    C=1.0,  # Inverse of regularization strength
    fit_intercept=False,  # Include an intercept
    class_weight='balanced',  # You can set class weights if needed
    solver='liblinear',  # Choose a solver appropriate for your data
    max_iter=100,  # You may need to increase this if the model doesn't converge
    random_state=42  # For reproducibility
    )
    
    lm.fit(X_train, y_train)

     # Calculate scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': lm.predict(X_train)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': lm.predict(X_val)})

    y_train_pred = lm.predict(X_train)
    y_val_pred = lm.predict(X_val)

    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    train_classification_report = class_rep(y_train, y_train_pred)
    val_classification_report = class_rep(y_val, y_val_pred)

    print(f'\nLogisitic Regression Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nClassification Report for Training Set:\n\n{train_classification_report}\n\n\n')
    print(f'\nClassification Report for Validation Set:\n\n{val_classification_report}\n')







# rf classifier

# ===========================================================================================================================================

from sklearn.ensemble import RandomForestClassifier

def model_4():
    grademap = {'A': 'Pass', 'B': 'Fail', 'C': 'Fail'}

    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews_sentiment_ratings.csv', index_col=0)
    ny_reviews = ny_reviews.rename(columns={'neg': 'negative', 'neu': 'neutral', 'pos': 'positive'})
    ny_reviews = ny_reviews[['grade', 'avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'reviews', 'negative', 'neutral', 'positive', 'compound']]

    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    X = ny_reviews.drop(columns=["grade"])  # Features
    y = ny_reviews["grade"]  # Target labels

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    tfidf = TfidfVectorizer()
    X_train_reviews_tfidf = tfidf.fit_transform(X_train["reviews"])
    X_val_reviews_tfidf = tfidf.transform(X_val["reviews"])
    X_test_reviews_tfidf = tfidf.transform(X_test["reviews"])

    # Combine TF-IDF vectors with sentiment features
    X_train_combined = hstack([X_train_reviews_tfidf, X_train[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])
    X_val_combined = hstack([X_val_reviews_tfidf, X_val[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])
    X_test_combined = hstack([X_test_reviews_tfidf, X_test[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']].values])

    # Define the hyperparameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }

    # Create a Random Forest model
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Perform grid search
    rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, scoring='accuracy')
    rf_grid_search.fit(X_train_combined, y_train)

    # Get the best Random Forest model
    rf = rf_grid_search.best_estimator_
    rf.fit(X_train_combined, y_train)

    # Calculate scores for Random Forest
    y_train_pred_rf = rf.predict(X_train_combined)
    y_val_pred_rf = rf.predict(X_val_combined)

    train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
    val_accuracy_rf = accuracy_score(y_val, y_val_pred_rf)

    # Print classification reports
    print(f'\nRandom Forest Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy_rf:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy_rf:.2f}\n')
    print(f'\nClassification Report for Random Forest (Training Set):\n\n{class_rep(y_train, y_train_pred_rf)}\n\n\n')
    print(f'\nClassification Report for Random Forest (Validation Set):\n\n{class_rep(y_val, y_val_pred_rf)}\n')






# Final model

# ===================================================================================================================================



target_names = ['Fail', 'Pass']

def final_model():
    # Load and preprocess your data
    ny_reviews = pd.read_csv('ny_reviews_sentiment_ratings.csv', index_col=0)
    grademap = {'A': 'Pass', 'B': 'Pass', 'C': 'Fail'}
    
    ny_reviews = ny_reviews.rename(columns={'neg': 'negative', 'neu': 'neutral', 'pos': 'positive'})
    ny_reviews = ny_reviews[['grade', 'avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'reviews', 'negative', 'neutral', 'positive', 'compound']]
    ny_reviews['grade'] = ny_reviews['grade'].map(grademap)

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Encode the target labels
    y_encoded = label_encoder.fit_transform(ny_reviews.grade)

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(ny_reviews[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'reviews', 'negative', 'neutral', 'positive', 'compound']], y_encoded, train_size=0.7, random_state=42)  # Combine features
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Initialize and fit the TfidfVectorizer on the training data
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train['reviews'])
    X_val_tfidf = tfidf.transform(X_val['reviews'])
    X_test_tfidf = tfidf.transform(X_test['reviews'])

    # Combine TF-IDF vectors with sentiment features for training data
    X_train_combined = hstack([X_train_tfidf, X_train[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']]])

    # Combine TF-IDF vectors with sentiment features for validation data
    X_val_combined = hstack([X_val_tfidf, X_val[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']]])

    # Combine TF-IDF vectors with sentiment features for test data
    X_test_combined = hstack([X_test_tfidf, X_test[['avg_service', 'avg_atmosphere', 'avg_food', 'avg_price', 'negative', 'neutral', 'positive', 'compound']]])

    # Create the XGBoost classifier instance
    bst = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.25, objective='multi:softprob', num_class=len(label_encoder.classes_))

    # Fit the XGBoost model on the combined features of the training data
    bst.fit(X_train_combined, y_train)

    # Predict the classes on the validation data
    preds = bst.predict(X_val_combined)

    # If you want to decode the predicted labels back to their original class names:
    # Convert one-hot encoded labels back to original class labels
    preds_decoded = label_encoder.inverse_transform(preds.argmax(axis=1))

    # Calculate scores
    # Flatten the predictions to a 1D array
    train_preds = bst.predict(X_train_combined)
    train_preds_flattened = train_preds.argmax(axis=1)

    y_train_res = pd.DataFrame({'actual': y_train, 'preds': train_preds_flattened})

    val_preds = bst.predict(X_val_combined)
    val_preds_flattened = val_preds.argmax(axis=1)

    y_val_res = pd.DataFrame({'actual': y_val, 'preds': val_preds_flattened})

    test_preds = bst.predict(X_test_combined)
    test_preds_flattened = test_preds.argmax(axis=1)

    y_test_res = pd.DataFrame({'actual': y_test, 'preds': test_preds_flattened})

    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])
    test_accuracy = accuracy_score(y_test_res['actual'], y_test_res['preds'])


    train_classification_report = class_rep(y_train, y_train_res['preds'], target_names=target_names)
    val_classification_report = class_rep(y_val, y_val_res['preds'], target_names=target_names)
    test_classification_report = class_rep(y_test, y_test_res['preds'], target_names=target_names)


    print(f'\nXGBClassifier Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nTest Accuracy: {test_accuracy:.2f}\n')


    print(f'\nClassification Report for Training Set:\n\n{train_classification_report}\n')
    print(f'\nClassification Report for Validation Set:\n\n{val_classification_report}\n')
    print(f'\nClassification Report for Validation Set:\n\n{test_classification_report}\n')

    # Plot accuracy comparison
    baseline_train_accuracy = 0.55  # Replace with the actual baseline train accuracy
    baseline_val_accuracy = 0.57  # Replace with the actual baseline validation accuracy
    plot_accuracy_comparison(train_accuracy, val_accuracy, test_accuracy, baseline_train_accuracy, baseline_val_accuracy)




