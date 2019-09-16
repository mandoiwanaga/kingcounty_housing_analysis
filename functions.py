import numpy as np 

#linear regression (sklearn and statsmodels)
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

#model evaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.model_selection import cross_val_score, cross_val_predict

#normality check
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')




def normality_check(df):
    """
    
    Fit Linear Regression model on each independent variable
    Return Jarque Bera, P-Value, Skew, and Kurtosis measures
    Return QQ-Plot and Histogram of residuals
    
    
    """
    predictors = df.drop('price', axis=1)

    for i in predictors:
    
        f = 'price~' + i
    
        model = smf.ols(formula=f, data=df).fit()
        resid = model.resid
    
        name = ['Jarque-Bera','Prob','Skew', 'Kurtosis']
        test = sms.jarque_bera(model.resid)
        print(i)
        print(list(zip(name, test)))
        print(f"Redisuals MIN: {round(resid.min(), 2)}")
        print(f"Redisuals MAX: {round(resid.max(), 2)}")
    
        plt.figure() 
        sm.graphics.qqplot(resid, 
                           dist=stats.norm, 
                           line='45', 
                           fit=True)
        plt.title(i)
        plt.show()
        plt.close()
    
        plt.figure()
        resid.hist(bins=(50), 
                   edgecolor = 'black', 
                   range=(resid.min(), resid.max()),
                   figsize=(10, 5))
        plt.show()
        plt.close()




def linreg_summary(df):
    
    """
    
    Define X and y variables
    Train-test split
    Fit LinearRegression model
    Return R^2, MSE, RMSE, MAE
    
    """

    y = df['price']
    X = df.drop(['price'], axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    # Instantiate and fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_hat_test = model.predict(X_test)
    y_hat_train = model.predict(X_train)
    
    # R Squared score
    r2 = r2_score(y_test, y_hat_test)
    print(f"R^2: {r2}")

    # Mean Squared Error
    test_mse = mean_squared_error(y_test, y_hat_test)
    print(f"MSE: {test_mse}")
    
    # Root Mean Squared Error
    test_rmse = np.sqrt(test_mse)
    print(f"RMSE: {test_rmse}")
    
    # Mean Absolute Error
    test_mae = mean_absolute_error(y_test, y_hat_test)
    print(f"MAE: {test_mae}")
    
    # Mean Squared Error for train data to be used for comparison
    train_mae = mean_absolute_error(y_train, y_hat_train)
    print(f"TRAIN_MAE: {train_mae}")
    
    
    
# Statsmodels OLS version to see p-values
def ols_linreg_summary(df):
    
    """
    Return Statsmodels OLS model summary

    """
    
    X = df.drop(['price'], axis=1)
    y = df['price']

    predictors_int = sm.add_constant(X)
    model = sm.OLS(y, predictors_int).fit()
    
    
    return model.summary()



def k_folds_cv(df):
    
    """
    
    Return Absolute Value K-Folds Cross Validation Results
    Return K-Folds visualization of predictions obtained from model
    
    """
    
    model = LinearRegression()
    X = df.drop(['price'], axis=1)
    y = df['price']
    
    
    cv_5_results = np.mean(abs(cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")))
    cv_10_results = np.mean(abs(cross_val_score(model, X, y, cv=10, scoring="neg_mean_absolute_error")))
    cv_20_results = np.mean(abs(cross_val_score(model, X, y, cv=20, scoring="neg_mean_absolute_error")))
    
    print(f"CV 5-Fold MAE: {cv_5_results}")
    print(f"CV 10-Fold MAE: {cv_10_results}")
    print(f"CV 20-Fold MAE: {cv_20_results}")
    
    
    
    predictions_5 = cross_val_predict(model, X, y, cv=5)
    predictions_10 = cross_val_predict(model, X, y, cv=10)
    predictions_20 = cross_val_predict(model, X, y, cv=20)

    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].scatter(y, predictions_5, edgecolors=(0, 0, 0))
    ax[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax[0].set_title('K-Folds (5) test')
    ax[0].set_xlabel('Measured')
    ax[0].set_ylabel('Predicted')
    ax[1].scatter(y, predictions_10, edgecolors=(0, 0, 0))
    ax[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax[1].set_title('K-Folds (10) test')
    ax[1].set_xlabel('Measured')
    ax[1].set_ylabel('Predicted')
    ax[2].scatter(y, predictions_20, edgecolors=(0, 0, 0))
    ax[2].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax[2].set_title('K-Folds (20) test')
    ax[2].set_xlabel('Measured')
    ax[2].set_ylabel('Predicted')
    plt.show()
    
    
    
    
    
    
    