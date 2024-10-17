import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xg
import networkx as nx
import operator
import scipy.stats
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from sklearn.preprocessing import MinMaxScaler


def window_moving(data,window_size):
    '''
    input:
    data = pd dataframe
    window_size=number of records in each segment
    output:
    windows= a list of dictionaries . each dictionary include star and end index of its
        correspondig segment
    num_windows = number of segments
    '''
    num_dates = data.shape[0]

    num_windows = num_dates // window_size

    windows = []
    for i in range(num_windows):
        start_id = i* window_size
        end_id = start_id + (window_size - 1)
        window = {'start_index': start_id,'end_index': end_id}
        windows.append(window)
    return windows

def cosine_sim(data,windows):
    '''
    input:
    data = pd dataframe
    windows=windows= a list of dictionaries . each dictionary include star and end index of its
        correspondig segment
    output:
    sim:a 3d tensor that its frontal slices are similarity matrix of the segments in order
    '''
    num_windows=len(windows)
    num_features = data.shape[1]
    sim = np.zeros((num_features,num_features,num_windows))
    for i in range(num_windows):
        sim[:,:,i] = cosine_similarity(data.iloc[windows[i]['start_index']:windows[i]['end_index']+1,:].T,dense_output=False)
        u, _, _ = np.linalg.svd(sim[:,:,i], full_matrices=True)
        sim[:,:,i]=np.matmul(u.T,sim[:,:,i]) 
        # np.fill_diagonal(sim[:,:,i], 0)
    # meanSim=np.mean(sim)
    # sim[abs(sim) <meanSim/3 ] = 0
    # sim[abs(sim) > meanSim] = 1
    # print(sim.shape)
    sim=np.abs(sim)

    return sim
def zij(Y,i,j,lam,N):
    if i==j:
        return 0.0
    else:
        numerator = 2 * np.exp(-(np.square(np.linalg.norm(Y[:,i]-Y[:,j],2)))/lam)
        #print(numerator)
        sum_i=0
        sum_j=0
        for h in range(N):
            if h!=i:
                sum_i += np.exp(-(np.square(np.linalg.norm(Y[:,i]-Y[:,h],2)))/lam)
            if h!=j:
                sum_j += np.exp(-(np.square(np.linalg.norm(Y[:,j]-Y[:,h],2)))/lam)
        return numerator/(sum_i+sum_j)


def Z_sim(data,windows):
    num_windows=len(windows)
    data_X=data.to_numpy()
    #preparing input for z function
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(data_X)
    N=input_data.shape[1]
    sim =np.zeros((N,N,num_windows), dtype='float64')
    cz=0
    for k in range(num_windows):
        # print(windows[k]['start_index'],windows[k]['end_index']+1)
        for i in range(N):
            for j in range(N):
                sim[i,j,k] = zij(input_data[windows[k]['start_index']:windows[k]['end_index']+1,:],i,j,0.01,N)
                if sim[i,j,k] < 1e-5 and sim[i,j,k] > -1e-5:
                    cz += 1
    print("number of zeros = ",cz)
    return sim
    
    
def add_time(sim):
    """
    input:
    sim =cosine similarity tensor (output of cosine_sim function)
    num_windows= 
    """
    arr=[]
    date=[datetime.datetime(2000, 1, 1) + datetime.timedelta(days=idx) for idx in range(sim.shape[2])]
    for t in range(len(date)):
        for i in range(len(sim)):
            for j in range(len(sim)):
                
                if sim[i,j,t]!=0:
                    arr.append(str("\"{}\"".format(date[t]))+" "+str(i)+" "+str(j)+" "+str(sim[i,j,t]))
    return arr

def create_graph_details(data,window_size):    
    windows=window_moving(data,window_size)

    #here we choose a similarity method
    sim=cosine_sim(data,windows)
    # sim=Z_sim(data,windows)

    arr=add_time(sim)
    return arr

def readRealGraph(arr):
    edgesTS = []

    nodes = set()
    edges = set()
    lookup = {}
    c = 0
    
    for line in arr:

        line = line.strip()
        items = line.split(' ')

        tstamp = ' '.join(items[0:2])

        tstamp = tstamp[1:-1]
        tstamp = datetime.datetime.strptime(tstamp, '%Y-%m-%d %H:%M:%S')

        t = items[2:4]
        t = list(map(int, t))

        if t[0] == t[1]:
            continue
        edgesTS.append((tstamp, tuple(t), items[-1]))
        nodes.add(t[0])
        nodes.add(t[1])
        edges.add(tuple([t[0], t[1]]))
    
    return edgesTS, nodes, edges 

def getGraph(edgesTS):
    G = nx.DiGraph()
    edges = {}
    # get tuple(t) from edgesTS and assign number of that edge as value
    for item in edgesTS:
        edge = item[1]
        # print(item[-1])
        edges[edge] = edges.get(edge, 0.0) + float(item[-1]) ################################

    #nrm = float(sum(edges.values()))
    G.add_edges_from([(k[0], k[1], {'weight': v}) for k, v in edges.items()])
    # G.add_edges_from([tuple(edge)])
    return G

def getSubgraph(G, N):
    Gcc = sorted([G.subgraph(c) for c in nx.connected_components(
        G.to_undirected())], key=len, reverse=True)
    # print("length GCC",len(Gcc))
    nodes = set()
    i = 0

    while len(nodes) < N:
        s = np.random.choice(Gcc[i].nodes())
        # s=0
        # print(s)
        i += 1
        nodes.add(s)
        for edge in nx.bfs_edges(G.to_undirected(), s):
            nodes.add(edge[1])
            if len(nodes) == N:
                break
    return nx.subgraph(G, nodes)


def weighted_DiGraph(arr):

    edgesTS, nodes, edges= readRealGraph(arr)
    G = getGraph(edgesTS)
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    ###################################################################################
    G = getSubgraph(G, len(nodes))
    ###################################################################################
    # print (len(nodes))
    G = G.copy()

    for i in G.nodes():
        if G.out_degree(i) == 0:
            for j in G.nodes():
                if i != j:
                    G.add_edge(i, j, weight=1)

    # print(nx.info(G))
    nrm = float(sum([val for (node, val) in G.degree()]))

    # nrm = float(sum(G.out_degree(weight='weight').values()))
    for i in G.edges(data=True):
        G[i[0]][i[1]]['weight'] = i[-1]['weight']/nrm
    return G


def flowPR(n_nodes,p_prime_nodes, stream, RS, current, iters=100000, alpha=0.85, beta=0.001, gamma=0.9999, normalization=1.0, padding=0):
    if beta == 1.0:
        beta = 0.0

    # tau = []
    # pearson = []
    # spearman = []
    # error = []
    x = []
    i = 0

    # rank_order = [key for (key, value) in sorted(
    #     ref_pr.items(), key=operator.itemgetter(1), reverse=True)]
    # ordered_pr = np.array([ref_pr[k] for k in rank_order])
    # print (stream)
    for e in stream:
        i += 1

        RS[e[0]] = RS.get(e[0], 0.0) * gamma + 1.0 * \
            (1.0 - alpha) * p_prime_nodes[e[0]] * normalization
        RS[e[1]] = RS.get(e[1], 0.0) * gamma + (current.get(e[0], 0.0) +
                                                1.0 * (1.0 - alpha) * p_prime_nodes[e[0]]) * alpha * normalization
        current[e[1]] = current.get(e[1], 0.0) + (current.get(e[0], 0.0) + 1.0 * (
            1.0 - alpha) * p_prime_nodes[e[0]]) * alpha * (1 - beta)
        current[e[0]] = current.get(e[0], 0.0) * beta
        

        # if  len(RS) == n_nodes:
            # if i == iters-1:
            #     print(sum(RS.values()))
        #     RS4 = np.array([RS[k] / sum(RS.values())
        #                           for k in rank_order])
        #     x.append(i+padding)

        # if i == iters-1:
        #     print(sum(RS.values()))

#     sorted_RS4 = np.array([RS[k] / sum(RS.values()) for k in rank_order])
    # return RS, current, tau, spearman, pearson, error, x
    # print("RS",RS)
    return RS

def featureSelection_tpr(arr,num_features,num_comp):
    

    Total_edges= len(arr)
    
    beta =0 # 0.5
    n = num_features
    iters =Total_edges
    alpha = 0.85
    gamma = 1.0

    # weights = 'real'


    G =weighted_DiGraph(arr)
    norm = sum([val for (node, val) in G.out_degree(weight='weight')])
    # sampling_edges = {e[:-1]: e[-1]['weight']/norm for e in G.edges(data=True)}
    sampling_edges = {e[:-1]: e[-1]['weight']/norm for e in G.edges(data=True)}
    
    # print(sampling_edges)
    # stream = [list(sampling_edges.keys())[i] for i in np.random.choice(range(len(sampling_edges)), size=iters, p=list(sampling_edges.values()))]
    stream = list(sampling_edges.keys())
    # print(stream)



    # basic (degree personalization)
    personalization = {k: v / norm for k, v in G.out_degree(weight='weight')}
    p_prime_nodes = {i: personalization[i]/G.out_degree(i, weight='weight') for i in G.nodes()}
    
    # pr_basic = nx.pagerank(G, alpha=alpha, personalization=personalization, weight='weight')
    # print(pr_basic)
    RS4_basic, current_basic = {}, {}

    #???????????????????????
    RS4_basic= flowPR(num_features,p_prime_nodes, stream, RS4_basic, current_basic, iters = iters, beta = beta, gamma = gamma)


    hr=dict(sorted(RS4_basic.items(), key=lambda item: item[1]))
    #             print (i,"   ",hr.keys())
    # print("hr",hr)
    indices=np.array([np.sort(list(hr.keys())[-num_comp:])])
    # print("indices",indices)
    # indices = np.array([x - 1 for x in indices])
    return indices



# def compute_pca(data,num_comp=None):
#     """
#     input:
#     num_comp=Number of components to keep. if n_components is not set all components are kept 
#         """
#     pca = PCA(n_components=num_comp).fit(data)
#     return pca.transform(data)

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
