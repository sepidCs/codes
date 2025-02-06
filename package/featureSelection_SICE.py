# SICE
import numpy as np
import pandas as pd
import xgboost as xg
import operator
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def normr(a, replace_zero_rows=True):
    a = normalize(a, axis=1, norm="l2")
    if replace_zero_rows:
        for i in range(a.shape[0]):
            if np.sum(np.abs(a[i, :])) == 0:
                a[i, :] = 1 / np.sqrt(a.shape[1])
    return a


def window_moving(data_without_target, window_size):
    """
    input :
    data_without_target = pandas dataset without target column
    window_size = number of samples in each segment
    output :
    win_index_dic = a list of dictionaries which includes start index and end index of each window

    num_windows = number of windows

    """
    num_windows = data_without_target.shape[0] // window_size
    win_index_dic = []
    for i in range(num_windows):
        # (extra rows ignored)
        #  last window may differ in size (may be larger)
        if i == num_windows - 1:
            start_id = i * window_size
            end_id = data_without_target.shape[0]
            window = {"start_index": start_id, "end_index": end_id}
            win_index_dic.append(window)
            break
        start_id = i * window_size
        end_id = start_id + (window_size - 1)
        window = {"start_index": start_id, "end_index": end_id}
        win_index_dic.append(window)
    # print("number of windows=",num_windows,"\nsize of windows =",window_size,data_without_target.shape[1],"\nlast window size =",end_id-start_id,data_without_target.shape[1])
    print("number of windows=", num_windows, "\nsize of windows =", window_size)
    # print(win_index_dic)
    return win_index_dic, num_windows


def cosine_sim(data_without_target, windows, num_windows):
    """
    input:
    data_without_target = pandas dataset without target column
    windows = list of dictionaries which contain start and end index of each window in order
    num_windows = total number of segments
    output:
    sim :i-th frontal slice of sim tensor is a similarity matrix of i-th window's features
    """
    sim = np.zeros(
        (data_without_target.shape[1], data_without_target.shape[1], num_windows)
    )
    for i in range(num_windows):
        segmented_window = data_without_target.iloc[
            windows[i]["start_index"] : windows[i]["end_index"] + 1, :
        ].T.to_numpy()
        temp = normr(segmented_window)
        cos_matrix = np.abs(temp.dot(temp.T))
        # cos_matrix=cosine_similarity(segmented_window,dense_output=False)
        sim[:, :, i] = cos_matrix - np.diag(np.diag(cos_matrix))
    return sim


def buildHistogram(matrix):
    plt.hist(matrix, bins=30)
    plt.show()


def H_matrix(sim):
    """
    input:
    sim :i-th frontal slice of tensor is a similarity matrix of i-th window's features
    output:
    H =  the second order correlation matrix ( of [i,j,:] fibers similarity)
    from feature similarity tensor"""

    [n1, n2, n3] = sim.shape
    feature_similarity_matrix = sim.reshape(n1 * n2, n3)
    temp = normr(feature_similarity_matrix, False)

    cos_matrix = np.abs(temp.dot(temp.T))
    Q = cos_matrix - np.diag(np.diag(cos_matrix))
    T = np.zeros(Q.shape)
    T[:, np.sum(np.abs(Q), 0) == 0] = 1 / Q.shape[0]
    Q = Q + T
    Q = normalize(Q, axis=0, norm="l1")
    return Q


# def page_rank_vec(sim):
#     """
#     input:
#     sim = i-th frontal slice of tensor is a similarity matrix of i-th window's features
#     output:
#     pr = returns a list of pageranks of second order correlation matrix
#     """
#     tol=10**-6
#     alpha=0.85
#     Q=H_matrix(sim)
#     n=Q.shape[0]
#     e=np.ones((n,1))
#     A=alpha*Q+(1-alpha)/n*(e.dot(e.T))
#     [Lambda,V]=np.linalg.eig(A)
#     pr=np.abs(V[:,0])
#     return pr


def page_rank_vec(sim):
    """
    Computes the PageRank vector using the power iteration method.

    Args:
    sim (numpy.ndarray): Similarity matrix.
    tol (float): Tolerance for convergence.
    alpha (float): Damping factor.
    max_iter (int): Maximum number of iterations.

    Returns:
    numpy.ndarray: PageRank vector.
    """
    max_iter = 100
    tol = 10**-7
    alpha = 0.85
    Q = H_matrix(sim)
    n = Q.shape[0]
    e = np.ones((n, 1))
    A = alpha * Q + (1 - alpha) / n * (e.dot(e.T))

    pr = np.ones(n) / n  # Initial PageRank vector
    for _ in range(max_iter):
        pr_new = A.dot(pr)
        pr_new /= np.sum(pr_new)  # Normalize the PageRank vector
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break
        pr = pr_new

    return pr


def page_rank_to_feature(pr):
    """
    input:
    pr = a list of pageranks of second order correlation matrix
    output:
    selected_features_index = indices of most important features
    """
    pr[pr < np.mean(pr)] = 0
    number_of_feature = int(np.sqrt(len(pr)))
    pr_reshaped = pr.reshape((number_of_feature, number_of_feature))
    # print("P=\n",pr_reshaped)

    sum_list = [
        np.sum(pr_reshaped[:, i]) + np.sum(pr_reshaped[i, :])
        for i in range(pr_reshaped.shape[0])
    ]
    sum_list_indice = sorted(
        range(len(sum_list)), key=lambda k: sum_list[k], reverse=True
    )
    print(sum_list, sum_list_indice)
    return sum_list_indice


def featureSelection(data_without_target, window_size):
    win_index_dic, num_windows = window_moving(data_without_target, window_size)
    feature_similarity_tensor = cosine_sim(
        data_without_target, win_index_dic, num_windows
    )  # sim
    page_rank_vector = page_rank_vec(feature_similarity_tensor)
    top_ranked_pairs = len(page_rank_vector)  # keep all pairs
    selected_features_index = page_rank_to_feature(page_rank_vector)
    return selected_features_index


# def xgboost_reg_error(data_without_target,target,testSize):
#     """
#     while splitting data shuffle is false due to keep time series sampales in order
#     """
#     x_train, x_test, y_train, y_test = train_test_split(data_without_target, target, test_size = testSize,shuffle=False)

#     model =  xg.XGBRegressor()
#     model.fit(x_train, y_train)
#     predictions = model.predict(x_test)
#     # rmse=np.sqrt(metrics.mean_squared_error(y_test, predictions))
#     rmse=np.linalg.norm(predictions-y_test)/np.linalg.norm(y_test)
#     # print("MSE  is:{:.2f}".format(pca_mse))
#     return rmse

# def linear_reg_error(data_without_target,target,testSize):
#     """
#     while splitting data shuffle is false due to keep time series sampales in order
#     """
#     x_train, x_test, y_train, y_test = train_test_split(data_without_target, target, test_size = testSize,shuffle=False)

#     model = LinearRegression()
#     model.fit(x_train, y_train)
#     predictions = model.predict(x_test)
#     rmse=np.sqrt(metrics.mean_squared_error(y_test, predictions))
#     # print("MSE  is:{:.2f}".format(pca_mse))
#     return rmse

# def compute_pca(data,num_comp=None):
#     """
#     input:
#     num_comp=Number of components to keep. if n_components is not set all components are kept
#         """
#     pca = PCA(n_components=num_comp).fit(data)
#     return pca.transform(data)


def page_rank_to_weight_matrix_seed_iv(pr):
    # pr[pr<np.mean(pr)]=0
    number_of_feature = int(np.sqrt(len(pr)))
    pr_reshaped = pr.reshape(number_of_feature, number_of_feature)
    return pr_reshaped


def weight_matrix_seed_iv(data_without_target, window_size):
    win_index_dic, num_windows = window_moving(data_without_target, window_size)
    feature_similarity_tensor = cosine_sim(
        data_without_target, win_index_dic, num_windows
    )  # sim
    page_rank_vector = page_rank_vec(feature_similarity_tensor)
    pr_reshaped = page_rank_to_weight_matrix_seed - iv(page_rank_vector)
    return pr_reshaped
