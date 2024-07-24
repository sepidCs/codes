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

def tensorizing(window_size,no_target_data):
    '''
    input:
    window_size = number of records in each segment
    no_target_data = data withot target
    output:
    tensor_data= stacked segments with step 1 frontly(mode3)
    '''
    #this func transform the 2D data to a 3D tensor
    # first dimension is timeseries samples , second is features and third dimension is time
    tensor_data=np.zeros((window_size,no_target_data.shape[1],no_target_data.shape[0]-window_size+1))
    for i in range(no_target_data.shape[0]-window_size+1):
        tensor_data[:,:,i]=no_target_data.iloc[i:i+window_size].to_numpy()
    print ("data without target size is",no_target_data.shape)
    print ("tensorized data size is",tensor_data.shape)
    return tensor_data

def svd_via_QR(data):
    '''
    input:
    data is a 2d array
    output:
    svd factors calculated using QR decomposition
    '''
    if data.shape[0] <= data.shape[1]:
        Q,R=np.linalg.qr(data.T, mode='reduced')
        u, s, vh =np.linalg.svd(R.T, full_matrices=True)
        return u,s,vh.dot(Q.T)
    else:
        Q,R=np.linalg.qr(data, mode='reduced')
        u, s, vh =np.linalg.svd(R, full_matrices=True)
        return Q.dot(u),s,vh 


def energy(s1,s2,tol_u1,tol_u2):
    """
    calculating energy of u1 and u2 
    using s1 and s2 tensor with given tolerances
    output:
    num_comp= number of features which meet the energy tol (tol_u1) condition 
    reduced_window_size = size of window which meet the tol_u2 condition
    """


    # energy list of u1
    e_u1=[sum(s1[:i+1])/sum(s1) for i in range(len(s1))]
    # select first energy which meet the condition
    reduced_window_size=next(e_u1.index(x) for x in e_u1 if x>=tol_u1) +1


    #energy list of U2 (features)
    e_u2=[sum(s2[:i+1])/sum(s2)  for i in range(len(s2))]
    # select first energy which meet the condition
    num_comp=next(e_u2.index(x) for x in e_u2 if x>=tol_u2) +1 
    # reduced_window_size=4
    # num_comp=5
    return reduced_window_size,num_comp

def tensor_data_reduced(tensor_data,u0,u1,reduced_window_size,num_comp):
    '''
    input:
    tensor_data: 3D array
    u0:factors[0] in hosvd decomposition -> factors,core=tucker(tensor_data)
    u1:factors[1] in hosvd decomposition -> factors,core=tucker(tensor_data)
    reduced_window_size:number of first columns to keep from u0 matrix
    num_comp:number of first columns to keep from u1 matrix
    output:
    reduced tensor calculated by mode i product 
    '''
    u0=u0[:,:reduced_window_size]
    u1=u1[:,:num_comp]
    new_data=tl.tenalg.mode_dot(tensor_data, u1.T, mode=1, transpose=False)
    new_data=tl.tenalg.mode_dot(new_data, u0.T, mode=0, transpose=False)
    print("data size is",new_data.shape)
    return new_data


def data_unfold(new_data):
    '''
    input:
    new_data=reduced tensor data
    output:
    unfolded_arr=unfolded tensor to 2d array from mode 3
    '''
    n_dim=new_data.shape
    unfolded_arr=new_data.reshape(n_dim[0]*n_dim[1],n_dim[2]).T
    print("unfolded data size is",unfolded_arr.shape)
    return unfolded_arr

def generate_data_HOSVD(window_size,data_without_target,tol_u0,tol_u1):    
    tensor_data=tensorizing(window_size,data_without_target)
    # unfold data by first dimension 
    tensor_data_reshaped_0= tensor_data.reshape(tensor_data.shape[0], tensor_data.shape[1]*tensor_data.shape[2],order='F')
    # print ("size of reshaped data from axis 0  =",tensor_data_reshaped_0.shape)
    u0,s0,_= svd_via_QR(tensor_data_reshaped_0)
    # print ("u0={} , s0={}".format(u0.shape,s0.shape))

    # unfold data by second dimension 
    tensor_data_reshaped_1= np.transpose(tensor_data, (1, 0, 2)).reshape(tensor_data.shape[1], tensor_data.shape[0]*tensor_data.shape[2],order='c')
    # print ("size of reshaped data from axis 1  =",tensor_data_reshaped_1.shape)
    u1,s1,_= svd_via_QR(tensor_data_reshaped_1)
    # print ("u1={} , s1={}".format(u1.shape,s1.shape))
    reduced_window_size,num_comp=energy(s0,s1,tol_u0,tol_u1)
    # print("reduced window size is {} out of {}".format(reduced_window_size,window_size))
    # print("number of features  is {} out of {}".format(num_comp,tensor_data.shape[1]))
    new_data=tensor_data_reduced(tensor_data,u0,u1,reduced_window_size,num_comp)
    unfolded_arr=data_unfold(new_data)
    return num_comp,unfolded_arr



###############################################################################
import math

def generateData_svdOnFrequencySpace(data_without_target,window_size,num_comp,col_percent) :   
    # reshape data to tensor
    tensor_data=tensorizing(window_size,data_without_target).transpose(1,0,2)
    # calculate cosine matrix
    DCT_matrix= dct(np.eye(tensor_data.shape[1]), axis=0)
    # mode 1 product of tensor and cosine matrix
    ctensorized=tl.tenalg.mode_dot (tensor_data,DCT_matrix, mode=1, transpose=False)

    # number of front slice's columns after removing last columns
    num_del= math.floor((1-col_percent)*ctensorized.shape[1])
    cut_ctensorized=ctensorized[:,:num_del,:]

    # computing svd for each frontal slice
    m,n,l=cut_ctensorized.shape
    new_data=np.zeros((n, num_comp,l))
    # print(m,n,l)
    for i in range(l):
        u, _, _= np.linalg.svd(cut_ctensorized[:,:,i], full_matrices=True)
        # comuting u.T * x for each slice
        new_data[:,:,i]=np.dot(u[:,:num_comp].T,cut_ctensorized[:,:,i]).T
    # reshape data to 2d array 
    n_dim=new_data.shape
    # new_data=np.random.rand(2,3,4)
    # print(new_data[:,:,0])
    print("reduced tensorized data size =",n_dim)

    new_data=new_data.reshape(n_dim[0]*n_dim[1],n_dim[2],order="C").T
    # print(new_data[0,:])
    return new_data



###########################
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

