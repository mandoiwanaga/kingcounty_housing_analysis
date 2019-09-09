import numpy as np 

#linear regression (sklearn and statsmodels)
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

#model evaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

#normality check
import scipy.stats as stats



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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
    train_mse = mean_squared_error(y_train, y_hat_train)
    print(f"TRAIN_MSE: {train_mse}")
    
    
    
# Statsmodels OLS version to see p-values
def ols_linreg_summary(df):
    """
    Return Statsmodels OLS model summary
    Return Residuals plot
    
    """
    
    X = df.drop(['price'], axis=1)
    y = df['price']

    predictors_int = sm.add_constant(X)
    model = sm.OLS(y, predictors_int).fit()
    
    residuals = model.resid
    fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
    fig.show()
    
    return model.summary()