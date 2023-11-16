<!--Created Anchor links to navigate read me better-->

- [Project Description](#project-description)
- [Project Goal](#project-goal)
- [Initial Thoughts](#initial-thoughts)
- [Plan](#the-plan)
- [Data Dictionary](#data-dictionary)
- [Steps to Reproduce](#steps-to-reproduce) 
- [Conclusions](#conclusions)
	- [Takeaway and Key Findings](#takeaways-and-key-findings)
	- [Reccomendations](#recommendations)
	- [Next Steps](#next-steps)

----------------------------------

# **Project: New York City Health Inspection Prediction**

<p align="center">
  <img src="https://www.wsav.com/wp-content/uploads/sites/75/2022/12/Food-inspection.jpg?w=612" width="750" alt="Image">
</p>

### Predict NYC Health Inspection Scores for restaturants based on 10 years of recorded data from Open Data via Socrata API.  

### Project Description

In today's culinary landscape, making informed decisions about dining out is challenging with the multitude of options available. Our project leverages New York Open data to integrate health inspection results and Google reviews from Google Maps. By analyzing this information, we predict restaurant health inspection outcomes and report sentiment based on posted reviews, offering valuable insights for safety and informed choices. Whether you're a foodie, a concerned parent, or a health-conscious individual, our platform assists in making better decisions about where to dine in New York City.

### Project Goal

1. Identify Drivers of Health Inspection Score outcomes.

2. Develop Machine Learning Models: Utilize the identified drivers to develop machine learning models capable of predicting NYC Health inspections accurately. The project will explore multiple classification-based models, including RandomForestClassifier, Decision Tree, XGBClassifier, to determine the best-performing approach.

### Initial Thoughts

My initial hypothesis is that drivers of health inspection scores will be boroughs and reviews text data.

## The Plan

* Acquire historical Health Inspection data via Socrata API from the Open Data Website and Google Reviews from Google Maps.
* Prepare data
* Explore data in search of drivers of Health Inspection Scores
  * Answer the following initial questions
     * What are the top 20 businesses in the NY Health Inspections dataset? 
     * What were the top 20 cuisine descriptions listed on inspections?
     * Based on the Top 20 failing business for the Bronx and Queens, is there a significant difference between the Bronx and Queens in terms of scores received and results ending with a citation or close actions?  
     * Is there a correlation between health inspection scores and health inspection dates over time?
* Develop a Model to predict Health Inspection Scores
  * Use drivers identified in exploration to help build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on Accuracy Score
  * Evaluate the best model on test data
* Draw conclusions

### Data Dictionary

| **Feature**        | **Data Type** | **Definition**                                       |
|--------------------|---------------|-----------------------------------------------------|
| `compound`        | Float         | aggregated score of positive, neutral and negative score of reviews    |
| `positive`        | Float         | positive score of google reviews.    |
| `neutral`        | Float         | neutral score of google reviews.    |
| `negative`         | Float         | negative score of google reviews.     |
| `reviews`       | Float         | lemmatized text data from google reviews    |
| `avg_price`      | Integer       | average price google review rating     |
| `avg_food`            | String        | average food google review rating         |
| `avg_atmosphere`      | String        | average atmosphere google review rating       |
| `avg_service`             | Integer       | average service google review rating.         |
| `grade` | Float         | **(Target Variable)** Health Inspection score grade (pass and **fail**)|


## Steps to Reproduce

1. Clone this project repository to your local machine.

2. You need to pip install Selenium and Tor and follow instruction on set-up within scraper_gmaps folder in repository. (You may download using homebrew if installed)

4. Install project dependencies by running pip install -r requirements.txt in your project directory.
	
5. Obtain an API key and API Token from the Socrata website.

6. Create a config.py file in your project directory with your API key using the following format:

> API_KEY = "YOUR_API_KEY"
> API_TOKEN = "YOUR_API_TOKEN"
 
6. Ensure that config.py is added to your .gitignore file to protect your API key.

7. Run the acquire.py script to fetch Health Inspection data from the Alpha Vantage API:

> python acquire.py

8. Execute the prepare.py script for data preprocessing and splitting:

> python prepare.py

9. Explore the dataset and answer initial questions using the explore.py script:

> python explore.py

10. Develop machine learning models by running the model.py script:

> python model.py

11. Evaluate the models, select the best-performing one, and draw conclusions based on the results of the model.py script.


# Conclusion

## Takeaways and Key Findings

- **Dunkin Donuts are among the highest counted businesses with health inspections.**  
- **In cuisine-description for New York is mostly composed of "American". There are descriptions without a clear unique description which may make this feature weak.**  
- **There is no significant difference between the top 20 businesses with high (failing scores) that were associated with a closed or violation cited action**  
- **There is a statistical correlation between health inspection scores and dates.**
- **Google review text counts for certain words are overrepresented and underrepresented for fail and pas. For example ramen, crepe and crab are underrepresented for failing inspections and words like Doughnut, Pastrami, and subway are overrepresented in passing inspections.**
- **Sentiment analysis proved it can accurately capture sentiment if reviews were long reviews with more text, versus short reviews, slang words were unable to be captured during this analysis. Manhattan had highest sentiment score on average and had one of the lower violation score on average showing a clear correlation that if you open a restaurant in manhattan ensure you are passing, becuase people will leave meaningful reviews.**
- ** The data was heavily imbalanced when we highlighted and split our target variable within the distribution.**


## Model Improvement
- The XGBclassifier model did well with balancing data. Model could be improved with a larger dataset and rich text data in feature space.

## Recommendations and Next Steps

- I would recommend maybe gaining additional sentiment data, and other forms of unstructured data that can be used with deep learning methodoligies which may help in improved accuracy as well as model classification report metrics like precision, recall, and f1 score to produce a robust model with statistical support in predictive analysis.

- Given more time, the following actions could be considered:
  - Web scrape more data to improve model performance.
  - Revisit the data exploration phase to gain a more comprehensive dataset.
    - Improve sentiment Analysis in order to have better float values in model feature space that account for slang and other important nunaces in google reviews data.
      - Utilizing classification/Regression models and tune hyperparameters like:
        - Logisitic regression
        - Random Forest




<!-- ## Questions to answer

Can we predict whether or not a restaurant will fail inspection?

Does type of restaruant coorelate to neighboorhood demogrphic?

Is there a specefic type of cuisine that tends to fail inspection more often?

Do demographics effect ratings of specific restaurants? Inception of specefic restaurnatns?
- Can compare across cities.

Do certain demogrphics inflate restuarant ratings?

Does age of restarutant effect health inspection rating?

Which customer groups give the highest/lowest review scores?


## Scope of Project 

We will be looking primarily with New York data.

We will be looking at all current restaurants in New York City.

We will be sourcing data from Yelp, Google places API, City of New york open data, US cencus bureau. 

We will be doing a superfilcail analysis of multiple cities to correlate ratings and demographics. -->


### [Daily Standup Document](https://docs.google.com/document/d/10oYE42-RaEuOSvOpb1OFBhJRB8KRiCj7kR0OBwPf1r4/edit?usp=sharing)

A repository on data science work on New York Health Inspection predictor/
