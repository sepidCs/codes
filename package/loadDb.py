import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def load_stock_market_dataframe(address):
    dateparse = lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce')
    data = pd.read_csv(address,index_col=["date","permno"], parse_dates=['date'], date_parser=dateparse, skipinitialspace=True)
    return data

def load_kaggle_dataframe(address):
    dateparse = lambda x: pd.to_datetime(x, format = "%Y-%m-%d %H:00:00", errors='coerce')
    data= pd.read_csv(address ,index_col=["datetime"], parse_dates=['datetime'], date_parser=dateparse, skipinitialspace=True)
    return data


def add_target(data,address):
    # generate a dataset with target column from sample_normalized.csv

    ret = data["ret"]
    # use next months (e. g. july) return as this months target (e. g. in june)
    data['TARGET'] = data.groupby(by='permno')['ret'].shift(-1)

    # data.groupby(level='permno')

    # see indices without consecutive month
    inds = np.where(data['TARGET'].isnull())[0]
    # print(inds)


    # drop final observation as there is no target per permno
    data.dropna(subset=["TARGET"], inplace=True)
    data.shape

    data.to_csv(address+'\sample_normalized_with_target.csv', index=True)

def generate_Pca(data,num_comp,address):
    pca     = PCA(n_components=num_comp).fit(data)
    columns = ['pca_%i' % i for i in range(num_comp)]
    data_pca  = pd.DataFrame(pca.transform(data), columns=columns, index=data.index)
    data_pca.head()
    # to generate pca_data file with target column

    ret = data["ret"]
    data_pca= data_pca.join(ret)
    # use next months (e. g. july) return as this months target (e. g. in june)
    data_pca['TARGET'] = data_pca.groupby(by='permno')['ret'].shift(-1)

    # data.groupby(level='permno')

    # see indices without consecutive month
    inds = np.where(data_pca['TARGET'].isnull())[0]
    print(inds)


    # drop final observation as there is no target per permno
    data_pca.dropna(subset=["TARGET"], inplace=True)
    data_pca.drop('ret', inplace=True, axis=1)

    data_pca.to_csv(address+'\data_pca_{}_with_target.csv'.format(num_comp), index=True)