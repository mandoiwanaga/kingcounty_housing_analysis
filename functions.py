from sklearn.metrics import mean_squared_error
from math import sqrt



def rmse(y_actual,y_predicted):
    """Return RootMeanSquared Error"""
    return sqrt(mean_squared_error(y_actual, y_predicted))


def mae(y_actual,y_predicted):
    """Return MeanAbsolute Error"""
    return abs(y_actual-y_predicted).mean()