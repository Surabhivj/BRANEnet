import pandas as pd
import networkx as nx
import numpy as np
from node2vec import Node2Vec
import gensim
from gensim.models import Word2Vec
import functions as f
import argparse
import os
import time



from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
pd.options.mode.chained_assignment = None  # default='warn'

#reading multilayer network

def parse_args():
    parser = argparse.ArgumentParser(description="performs PPI prediction and evaluation")

    parser.add_argument('--multilayer_networkfile', nargs='?',
                        help='multilayer_networkfile')
    parser.add_argument('--annotation_file', nargs='?',
                        help= 'annotation file')
  
    return parser.parse_args()


def main(args):
    start = time.time()
    print("Spliting train and test data....")
    
    G = nx.read_gml(args.multilayer_networkfile)
    largest_cc = max(nx.connected_components(G), key=len)
    G_cc = nx.Graph(G.subgraph(largest_cc))
    nx.write_gml(G_cc,"EdgeList_R_largestcc.gml")

    anno = pd.read_csv(args.annotation_file, sep = "\t")

    tf = list(anno[anno['shape']== 'triangle'].label)
    gene = list(anno[anno['shape']== 'circle'].label)
    metab = list(anno[anno['shape']== 'square'].label)
    
    out_folder = "BRANEnet_TF_target_pred"
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for n in range(10):

        train, test, nodes = f.split_tf_target(G,tf,gene)
        
        file_test = "EdgeList_R_test" + str(n) + ".txt"
        file_train = "EdgeList_R_train" + str(n) + ".txt"
        
        outfile_test = os.path.join(out_folder, file_test)
        outfile_train = os.path.join(out_folder, file_train)

        file_nodes = "nodes" + str(n) + ".txt"
        
        outfile_nodes = os.path.join(out_folder, file_nodes)

        nx.write_edgelist(test,outfile_test, data=False)
        nx.write_edgelist(train,outfile_train, data=False)

        textfile = open(outfile_nodes, "w")
        for element in range(len(nodes)):
            textfile.write(str(element) + "\n")
        textfile.close()
        
        os.system("python BRANEnet.py --multilayer_networkfile {} --T {} --d {} --type {}".format(outfile_train, 3, 128,'edgelist'))
        
        print("-------Negative sampling  (split: " + str(n+1)+ ") --------")
        
        test_graph = nx.read_edgelist(outfile_test)
        train_graph = nx.read_edgelist(outfile_train)

        pos_test_edges =list(test_graph.edges)
        pos_train_edges =list(train_graph.edges)

        non_edges_refnet = list(nx.non_edges(test_graph))
        non_edges_input = list(nx.non_edges(train_graph))

        non_edges = list(set(non_edges_refnet) & set(non_edges_input))

        np.random.shuffle(non_edges)

        neg_train_edges = non_edges[:len(pos_train_edges)]
        neg_test_edges = non_edges[len(pos_train_edges):(len(pos_train_edges)+ len(pos_test_edges))]

        # Prepare the positive and negative samples for training set
        train_samples = pos_train_edges + neg_train_edges
        train_labels = [1 for _ in pos_train_edges] + [0 for _ in neg_train_edges]
        # Prepare the positive and negative samples for test set
        test_samples = pos_test_edges + neg_test_edges
        test_labels = [1 for _ in pos_test_edges] + [0 for _ in neg_test_edges]

        #save_file_path = "train.pkl"
    
        train = {'edges':train_samples, 'labels': train_labels}
        test = {'edges':test_samples, 'labels': test_labels}
        
        test_pkl = "EdgeList_R_test" + str(n) + ".pkl"
        train_pkl = "EdgeList_R_train" + str(n) + ".pkl"
        
        outfile_test_pkl = os.path.join(out_folder, test_pkl)
        outfile_train_pkl = os.path.join(out_folder, train_pkl)

        pickle.dump(train, open(outfile_train_pkl, "wb"))
        pickle.dump(test, open(outfile_test_pkl, "wb"))
    '''
    with open(save_file_path, 'wb') as f:
        pickle.dump({'train': {'edges':train_samples, 'labels': train_labels }},f)
        pickle.dump({'test': {'edges':test_samples, 'labels': test_labels}}, f)
        f.close()
    '''
        
        
    print("Done!")
    end = time.time()
    print("Time in Seconds:" + str(format(end - start,".2f")))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    