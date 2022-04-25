# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:16:16 2021

@author: jagtaps
"""


import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csgraph
import logging
import theano
from theano import tensor as T
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef
import pickle
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances,paired_distances
import numpy as np
from sklearn.metrics import precision_recall_curve,recall_score,matthews_corrcoef
from sklearn.metrics import roc_curve,average_precision_score,auc
import scipy.io as sio
from sklearn.metrics import pairwise_distances
import ast
import networkx as nx
import statistics
from sklearn import preprocessing
from scipy.spatial import distance_matrix
import argparse
import random


import glob
import os

pd.options.mode.chained_assignment = None  # default='warn'


logger = logging.getLogger(__name__)
theano.config.exception_verbosity='high'

def PPMI_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        logger.info("Compute matrix %d-th power", i+1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)

def res_dat(emb):
        idx = list(emb.index)
        sim = emb.values @ emb.transpose().values
        min_max_scaler = MinMaxScaler()
        normdf = min_max_scaler.fit_transform(sim)
        np.fill_diagonal(normdf, 0)
        normdf = np.triu(normdf)
        dat = np.asmatrix(np.where(normdf > 0.699, 1, 0))
        sim_dat = pd.DataFrame(dat, index = idx, columns = idx)
        res = sim_dat.stack().reset_index()
        #res = res.sort_values(by = 0, ascending=False)
        res.columns = ['#source','target',0]
        return res
    

def embedd(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def net_recons(emb,ref_net):
    
    emb = emb.dropna(axis=1)
    ref_net = nx.Graph(ref_net.subgraph(np.array(emb.index)))
    emb = emb[emb.index.isin(ref_net.nodes)]
    
    #sim = emb.values @ emb.transpose().values
    
    sim = pairwise_distances(emb,metric='cosine')
    
    min_max_scaler = MinMaxScaler()
    normdf = min_max_scaler.fit_transform(sim)
    np.fill_diagonal(normdf, 0)
    normdf = np.triu(normdf)
    sim_dat = pd.DataFrame(normdf, index = emb.index, columns = emb.index)
    #print(sim_dat)
    res = sim_dat.stack().reset_index()

    a1 = nx.adjacency_matrix(ref_net)
    dd = pd.DataFrame(data = a1.todense(),columns= np.array(ref_net.nodes))
    dat = dd.set_index([np.array(ref_net.nodes)])
    dat = dat.sort_index()
    dat = dat.transpose().sort_index()
    ref_dat = np.triu(dat)
    ref_dat = pd.DataFrame(ref_dat, index = dd.index, columns = dd.index)
    ref_dat = ref_dat.stack().reset_index()

    node2vec_datt = pd.DataFrame({'y': ref_dat[0], 'probs': res[0]})
    node2vec_datt = node2vec_datt.sort_values(["y", "probs"], ascending = (False, False))

    return node2vec_datt


def mcc_t(dat):
    y = dat['y']
    prob_s = dat['probs']
    thres = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    mcc = []
    for i in thres:
        prob = np.array([1 if item > i else 0 for item in prob_s])
        m = matthews_corrcoef(y,prob)
        mcc.append(m)
    p_d = {'thres': thres, 'dat_mcc': mcc}
    pk_d = pd.DataFrame(p_d)
    return(pk_d)


def pr_at_k(dat):
    k = [1,10,50,100,150, 200,250,300,350, 400,450, 500]
    dat_dist = dat.sort_values(ascending=False, by='probs')
    s_dist = []
    for i in k:
        ss_dist = ((sum(dat_dist.y[:i-1].values))/i)
        s_dist.append(ss_dist)
 
    p = {'k': k, 'precision_dist': s_dist}
    pk = pd.DataFrame(p)
    return pk


def aupr(dat):
    y = dat['y']
    prob_s = dat['probs']
    precision, recall, _ = precision_recall_curve(y, prob_s)
    aup = auc(recall,precision)
    return aup



def DEA_g(omics_dat):
    omics_dat = omics_dat.filter(regex='wt')
    omics_dat_t0 = omics_dat.filter(regex='t0').mean(axis=1)
    omics_dat_t = omics_dat.filter(regex=r'(t20|t120)')
    omics_dat_fc = np.log2(omics_dat_t.div(omics_dat_t0,axis = 0))
    omics_dat_fc = omics_dat_fc[~omics_dat_fc.isin([np.nan, np.inf, -np.inf]).any(1)]
    omics_up = list(omics_dat_fc.index[omics_dat_fc[omics_dat_fc > 2].count(axis=1) > 1])
    omics_down = list(omics_dat_fc.index[omics_dat_fc[omics_dat_fc < -2].count(axis=1) > 1])
    omics_sel = omics_up + omics_down
    omics_dat_fc = omics_dat_fc[omics_dat_fc.index.isin(omics_sel)]
    omics_fc = pd.DataFrame(omics_dat_fc.values)
    omics_fc.index = list(omics_dat_fc.index)
    omics_dat_corr = omics_dat_fc.transpose().corr()
    omics_dat_corr = omics_dat_corr.stack().reset_index()
    omics_dat_corr.columns = ['node1','node2','Value']
    omics_dat_corr = omics_dat_corr[abs(omics_dat_corr['Value']) > 0.8]
    omics_dat_corr = omics_dat_corr[['node1','node2']].reset_index()
    del omics_dat_corr['index']
    up = pd.DataFrame({'label': omics_up, 'color': 'UP', 'shape': 'circle','type' : 'gene'},index = omics_up)
    down = pd.DataFrame({'label': omics_down, 'color': 'DOWN','shape': 'circle','type' : 'gene'},index = omics_down)
    diff = pd.concat([up,down])     
    return[diff,omics_fc,omics_dat_corr]

def DEA_tf(omics_dat):
    omics_dat = omics_dat.filter(regex='wt')
    omics_dat_t0 = omics_dat.filter(regex='t0').mean(axis=1)
    omics_dat_t = omics_dat.filter(regex=r'(t20|t120)')
    omics_dat_fc = np.log2(omics_dat_t.div(omics_dat_t0,axis = 0))
    omics_dat_fc = omics_dat_fc[~omics_dat_fc.isin([np.nan, np.inf, -np.inf]).any(1)]
    omics_up = list(omics_dat_fc.index[omics_dat_fc[omics_dat_fc > 1].count(axis=1) > 1])
    omics_down = list(omics_dat_fc.index[omics_dat_fc[omics_dat_fc < -1].count(axis=1) > 1])
    omics_sel = omics_up + omics_down
    omics_dat_fc = omics_dat_fc[omics_dat_fc.index.isin(omics_sel)]
    omics_fc = pd.DataFrame(omics_dat_fc.values)
    omics_fc.index = list(omics_dat_fc.index)
    omics_dat_corr = omics_dat_fc.transpose().corr()
    omics_dat_corr = omics_dat_corr.stack().reset_index()
    omics_dat_corr.columns = ['node1','node2','Value']
    omics_dat_corr = omics_dat_corr[abs(omics_dat_corr['Value']) > 0.8]
    omics_dat_corr = omics_dat_corr[['node1','node2']].reset_index()
    del omics_dat_corr['index']
    up = pd.DataFrame({'label': omics_up, 'color': 'UP', 'shape': 'circle','type' : 'tf'},index = omics_up)
    down = pd.DataFrame({'label': omics_down, 'color': 'DOWN','shape': 'circle','type' : 'tf'},index = omics_down)
    diff = pd.concat([up,down])     
    return[diff,omics_fc,omics_dat_corr]


def tf_gene(tf_fc, ntf_fc):
    tf = tf_fc.index
    ntf = ntf_fc.index
    
    tf_ntf_data = pd.concat([tf_fc,ntf_fc])  
    
    dat_corr = tf_ntf_data.transpose().corr()
    dat_corr = dat_corr.stack().reset_index()
    dat_corr.columns = ['node1','node2','Value']
    dat_corr = dat_corr[abs(dat_corr['Value']) > 0.8]
    dat_corr = dat_corr[['node1','node2']].reset_index()
    del dat_corr['index']
    
    dat_corr = dat_corr[dat_corr['node1'].isin(tf)]
    dat_corr = dat_corr[dat_corr['node2'].isin(ntf)]

    return dat_corr


def DEA_m(omics_dat):
    omics_dat = omics_dat.filter(regex='wt')
    omics_dat_t0 = omics_dat.filter(regex='t0').mean(axis=1)
    omics_dat_t = omics_dat.filter(regex=r'(t20|t120)')
    omics_dat_fc = omics_dat_t.div(omics_dat_t0,axis =0)
    omics_dat_fc = omics_dat_fc[~omics_dat_fc.isin([np.nan, np.inf, -np.inf]).any(1)]
    omics_up = list(omics_dat_fc.index[omics_dat_fc[omics_dat_fc > 1].count(axis=1) > 1])
    omics_down = list(omics_dat_fc.index[omics_dat_fc[omics_dat_fc < -1].count(axis=1) > 1])
    omics_sel = omics_up + omics_down
    omics_dat_fc = omics_dat_fc[omics_dat_fc.index.isin(omics_sel)]
    omics_fc = pd.DataFrame(omics_dat_fc.values)
    omics_fc.index = list(omics_dat_fc.index)
    omics_dat_corr = omics_dat_fc.transpose().corr()
    omics_dat_corr = omics_dat_corr.stack().reset_index()
    omics_dat_corr.columns = ['node1','node2','Value']
    omics_dat_corr = omics_dat_corr[abs(omics_dat_corr['Value']) > 0.8]
    omics_dat_corr = omics_dat_corr[['node1','node2']].reset_index()
    del omics_dat_corr['index']
    up = pd.DataFrame({'label': omics_up, 'color': 'UP', 'shape': 'square','type' : 'metabolite'},index = omics_up)
    down = pd.DataFrame({'label': omics_down, 'color': 'DOWN','shape': 'square','type' : 'metabolite'},index = omics_down)
    diff = pd.concat([up,down])     
    return[diff,omics_fc,omics_dat_corr]


#network properties
import networkx as nx
import pandas as pd

def node_prop(filename, d, c):
    
    BRANet_R = nx.read_weighted_edgelist(filename)
    BRANet_R.degree()
    #nx.clustering(BRANet_R)

    deg = pd.DataFrame(BRANet_R.degree())
    deg.columns = (["nodes", "degree"])

    clus = nx.clustering(BRANet_R)
    clus_dat = pd.DataFrame({'nodes': clus.keys(), 'clustering_coefficient' : clus.values()})

    node_proterties = pd.merge(deg,clus_dat, on = "nodes")
    node_proterties = node_proterties[node_proterties['clustering_coefficient'] > c]
    node_proterties = node_proterties[node_proterties['degree'] > d]
    
    node_proterties = node_proterties.sort_values(by = "degree", ascending = False)
    outfile = filename.split(".")[0] + "_node_properties.txt"
    
    node_proterties.to_csv(outfile, sep = "\t", index = False)
    
    return node_proterties


def split_tf_target(G,tf,gene):
    
    test_dat = pd.DataFrame(columns=['source', 'target'])
    train_dat = pd.DataFrame(columns=['source', 'target'])

    for t in tf:
        nei1 = list(G.neighbors(t))
        nei = list(set(nei1) & set(gene))
        random.shuffle(nei)
        nei_split = np.array_split(nei, 2)
        test_nei = nei_split[0]
        train_nei = nei_split[1]

        test_dat1 = pd.DataFrame({'source':t, 'target':test_nei })
        train_dat1 = pd.DataFrame({'source':t, 'target':train_nei })

        test_dat = pd.concat([test_dat,test_dat1])
        train_dat = pd.concat([train_dat,train_dat1])
        
    tf_target_testnet = nx.from_pandas_edgelist(test_dat)
    tf_target_trainnet = nx.from_pandas_edgelist(train_dat)
    
    test_cc = max(nx.connected_components(tf_target_testnet), key=len)
    train_cc = max(nx.connected_components(tf_target_trainnet), key=len)
    
    tf_target_traincc = nx.Graph(tf_target_trainnet.subgraph(train_cc))
    tf_target_testcc = nx.Graph(tf_target_testnet.subgraph(test_cc))
    
    nodes = list(set(list(tf_target_traincc)).intersection(set(list(tf_target_testcc))))
    nodes_dict = dict(zip(nodes,range(len(nodes))))
    
    tf_target_testnet = nx.Graph(tf_target_testnet.subgraph(nodes))
    tf_target_testcc = nx.relabel_nodes(tf_target_testcc,nodes_dict)
    
    tf_target_trainnet = nx.Graph(tf_target_trainnet.subgraph(nodes))
    tf_target_traincc = nx.relabel_nodes(tf_target_traincc,nodes_dict)
    
    print(tf_target_traincc)
    
    return(tf_target_traincc,tf_target_testcc,nodes)
