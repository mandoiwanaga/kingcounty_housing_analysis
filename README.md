# King County Housing Analysis

## Business Understanding
For this project I will look to accurately predict housing prices in King County, Washington. This model may be utilized as a predictor for individuals or companies looking to discover opportunities in the real-estate market.

## Data Understanding
The dataset used in this project is the <a href="https://www.kaggle.com/harlfoxem/housesalesprediction">King County Housing data set</a> from Kaggle. 
<br/>
It includes houses sold between 2014 and 2015. 
<br/>
The dataset contains 21613 observations of homes sold with 19 house features along with prices. The feature descriptions can be found [here.](https://github.com/mandoiwanaga/kingcounty_housing_analysis/blob/master/data/column_names.md) 

## Data Preperation
After conducting exploratory data analysis, several tasks needed to be completed before training a model with this dataset.  
<br/>
Tasks included:

- dealing with missing values
- dealing with placeholder values
- dealing with datatypes
- dealing with categorical variables
- dealing with multicollinearity
- feature selection 
- feature engineering
- feature scaling
- fulfilling Linear Regression Assumptions

<br/>
Assumptions of Linear Regression:

- Linearity 
- Normality (Residuals)
- Homoscedasticity


<br/>
This workflow can be viewed in eda_model.ipynb.


## Modeling
Machine Learning model used was Multiple Linear Regression. I utilized CRISP-DM process in this project. Reiterating the process until satisfactory evaluation results were achieved. 
<br/>
![CRISP-DM](https://github.com/mandoiwanaga/kingcounty_housing_analysis/blob/master/images/crispdm.png)  
- Image from Learn.co

## Evaluating
For this model I've decided to use MAE (Mean Absolute Error) as my metric for evaluation.  

## Future Work
- Incorporate location better, conduct more eda to discover other locations, other than downtown Seattle, that influence housing prices
- Deploy as FlaskApp