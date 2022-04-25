import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from networkx.algorithms.community import greedy_modularity_communities
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="performs PPI prediction and evaluation")

    parser.add_argument('--emb', nargs='?',type = str,
                        help='embedding file')
    parser.add_argument('--thres', nargs='?', type = float,
                        help= 'threshold to select edges')
  
    return parser.parse_args()


def main(args):
    
    emb = pd.read_csv(args.emb,index_col=0,header = None,skiprows = 1,sep=' ').sort_index()
    name = os.path.split(args.emb)[1]
    name = name.split(".")[0]
    idx = list(emb.index)
    sim = emb.values @ emb.transpose().values
    min_max_scaler = MinMaxScaler()
    normdf = min_max_scaler.fit_transform(sim)
    np.fill_diagonal(normdf, 0)
    normdf = np.triu(normdf)
    #dat = np.asmatrix(np.where(normdf > 0.799, 1, 0))
    sim_dat = pd.DataFrame(normdf, index = idx, columns = idx)
    res = sim_dat.stack().reset_index()
    #res = res.sort_values(by = 0, ascending=False)
    res.columns = ['#source','target','edge_score']
    res = res[res['edge_score']>args.thres]
    print(res)
    net_file = name + "_NetworFile.txt"
    res.to_csv(net_file, index = False,sep = ' ' )
    
    net = nx.read_weighted_edgelist(net_file)
    c = list(greedy_modularity_communities(net))
    
    #print(c)

    for m in range(len(c)):
        if len (c[m]) > 10:
            filename = name + '_Module_' + str(m+1)
            textfile = open(filename, "w")
            for element in sorted(c[m]):
                textfile.write(element + "\n")
            textfile.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)