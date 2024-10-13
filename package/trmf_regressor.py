import numpy as np
import random

from package.trmf import trmf

def normalized_data(data, T_train, T_start, normalize=True):
    N = len(data)
    # split on train and test
    train = data[:, T_start:T_start+T_train].copy()

    # normalize data
    if normalize:
        mean_train = np.array([])
        std_train = np.array([])
        for i in range(len(train)):
            if (~np.isnan(train[i])).sum() == 0:
                mean_train = np.append(mean_train, 0)
                std_train = np.append(std_train, 0)
            else:
                mean_train = np.append(mean_train, train[i][~np.isnan(train[i])].mean())
                std_train = np.append(std_train, train[i][~np.isnan(train[i])].std())
        
        std_train[std_train == 0] = 1.

        train -= mean_train.repeat(T_train).reshape(N, T_train)
        train /= std_train.repeat(T_train).reshape(N, T_train)
    return train

def get_slice(data, T_train, T_test, T_start, normalize=True):
    N = len(data)
    # split on train and test
    train = data[:, T_start:T_start+T_train].copy()
    test = data[:, T_start+T_train:T_start+T_train+T_test].copy()

    # normalize data
    if normalize:
        mean_train = np.array([])
        std_train = np.array([])
        for i in range(len(train)):
            if (~np.isnan(train[i])).sum() == 0:
                mean_train = np.append(mean_train, 0)
                std_train = np.append(std_train, 0)
            else:
                mean_train = np.append(mean_train, train[i][~np.isnan(train[i])].mean())
                std_train = np.append(std_train, train[i][~np.isnan(train[i])].std())
        
        std_train[std_train == 0] = 1.

        train -= mean_train.repeat(T_train).reshape(N, T_train)
        train /= std_train.repeat(T_train).reshape(N, T_train)
        test -= mean_train.repeat(T_test).reshape(N, T_test)
        test /= std_train.repeat(T_test).reshape(N, T_test)
    
    return train, test

def trmf_reg_error(data_without_target,target,testSize,lags = [1,25],K = 4,lambda_f = 1.,lambda_x = 1.,lambda_w = 1.,alpha = 1000.,eta = 1., max_iteration=1000):
    data_without_target["target"]=target.values
    # display(data_without_target)

    data=data_without_target.to_numpy().T
    # display(data)

    T_train = data.shape[1]-testSize
    T_test = testSize
    train, test =get_slice(data, T_train, T_test, 0, normalize=True)
    # display(train)
    # display(test)

    model = trmf(lags, K, lambda_f, lambda_x, lambda_w, alpha, eta)
    model.fit(train, max_iter = max_iteration)
    # train_preds = np.dot(model.F, model.X)
    test_preds = model.predict(T_test)[-1]
    y_test = test[-1]
    rmse=np.linalg.norm(test_preds-y_test)/np.linalg.norm(y_test)
    return rmse 


def normalized_deviation(prediction, Y):
    return abs(prediction - Y).sum() / abs(Y).sum()

