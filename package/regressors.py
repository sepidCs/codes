import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import xgboost as xg
from scipy.fft import dct
from sklearn import metrics

def xgboost_reg_error(data_without_target,target,testSize):
    """
    while splitting data shuffle is false due to keep time series sampales in order
    """
    x_train, x_test, y_train, y_test = train_test_split(data_without_target, target, test_size = testSize,shuffle=False)

    model =  xg.XGBRegressor()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    # rmse=np.sqrt(metrics.mean_squared_error(y_test, predictions))
    rmse=np.linalg.norm(predictions-y_test)/np.linalg.norm(y_test)
    # print("MSE  is:{:.2f}".format(pca_mse))
    return rmse

def linear_reg_error(data_without_target,target,testSize):
    """
    while splitting data shuffle is false due to keep time series sampales in order
    """
    x_train, x_test, y_train, y_test = train_test_split(data_without_target, target, test_size = testSize,shuffle=False)

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse=np.sqrt(metrics.mean_squared_error(y_test, predictions))
    # print("MSE  is:{:.2f}".format(pca_mse))
    return rmse


def compute_pca(data,num_comp=None):
    """
    input:
    num_comp=Number of components to keep. if n_components is not set all components are kept 
        """
    pca = PCA(n_components=num_comp).fit(data)
    return pca.transform(data),pca.explained_variance_

def energy(eigenvalues,tol):

    energy_list=[sum(eigenvalues[:i+1])/sum(eigenvalues)  for i in range(len(eigenvalues))]
    num_comp=next(energy_list.index(x) for x in energy_list if x>=tol) +1 

    return num_comp
    