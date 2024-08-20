import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
def load_dynamic_gas_mixtures(address,filename):
    date_str = "20200101"  # Extract the "20160930" part from the file name

    # Convert the extracted date to a Timestamp object
    start_time = pd.Timestamp(date_str)

    # Read the CSV file without parsing the "Time_(seconds)" column as a date
    data = pd.read_csv(address + filename, skipinitialspace=True)

    # Convert the "Time_(seconds)" column to a timedelta (time difference)
    data['Time_(seconds)'] = pd.to_timedelta(data['Time_(seconds)'], unit='s')

    # Create the actual datetime by adding the timedelta to the start_time
    data['Time_(seconds)'] = start_time + data['Time_(seconds)']

    # Optionally, set the new 'Datetime' column as the index of the DataFrame
    data.set_index('Time_(seconds)', inplace=True)
    return data
def load_temperature_modulation(address,filename)
    date_str = filename[:8]  # Extract the "20160930" part from the file name

    # Convert the extracted date to a Timestamp object
    start_time = pd.Timestamp(date_str)

    # Read the CSV file without parsing the "Time (s)" column as a date
    data = pd.read_csv(address + "//" + filename, skipinitialspace=True)

    # Convert the "Time (s)" column to a timedelta (time difference)
    data['Time (s)'] = pd.to_timedelta(data['Time (s)'], unit='s')

    # Create the actual datetime by adding the timedelta to the start_time
    data['Time (s)'] = start_time + data['Time (s)']

    # Optionally, set the new 'Datetime' column as the index of the DataFrame
    data.set_index('Time (s)', inplace=True)
    return data

def load_energydata_complete_dataframe(address):
    dateparse = lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:00", errors='coerce')
    data = pd.read_csv(address, index_col="date", parse_dates=['date'], date_parser=dateparse, skipinitialspace=True)
    return data

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