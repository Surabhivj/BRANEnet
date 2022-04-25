import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, RawTextHelpFormatter
import glob
import pickle
from sklearn.metrics import roc_auc_score, roc_curve,precision_recall_curve,auc
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, model_selection, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import preprocessing
import numpy as np
from scipy.spatial import distance
import argparse
import scipy.stats as stats
import glob
import ast
import time



def parse_args():
    parser = argparse.ArgumentParser(description="performs PPI prediction and evaluation")

    parser.add_argument('--op', nargs='?',
                        help='operator to compute edge features. e.g, average, l2')
      
    return parser.parse_args()



def extract_feature_vectors_from_embeddings(edges,lab, embeddings, binary_operator):
    features = []
    labels = []
    lab = lab.copy()
    for i in range(len(edges)):
        edge = edges[i]
        labs = lab[i]
        if all (k in embeddings for k in (str(edge[0]),str(edge[1]))):
            vec1 = np.asarray(embeddings[str(edge[0])],dtype=np.float64)
            vec2 = np.asarray(embeddings[str(edge[1])],dtype=np.float64)
            value = 0
            if binary_operator == "average":
                value = 0.5 * (vec1 + vec2)
            if binary_operator == "l2":
                value = abs(vec1 - vec2)**2
                
            features.append(value)
            labels.append(labs)
    features = np.asarray(features)
    labels = np.asarray(labels)
    return features,labels

def main(args):
    start = time.time()
    print("Operator: " + args.op)
    #print("....................training model ................")
    
    scores = {args.op: {'AUPR': [], 'AUROC': []}}
    
    output_file =  'TF_target_' + args.op + '_pred_scores'
    with open(output_file, "w") as fp:
        fp.write("{} {} {} {} {}\n".format('Train_data','Embedding_file', 'Operator', 'AUPR','AUROC'))
        for t in range(10):
            emb_folder = "BRANEnet_emb"
            f = "BRANet_EdgeList_R_train" + str(t) + "_w_3_d_128.emb"
            emb = os.path.join(emb_folder, f)

            tr =  "EdgeList_R_train" + str(t) + ".pkl"
            train = os.path.join("BRANEnet_TF_target_pred", tr)

            ts = "EdgeList_R_test" + str(t) + ".pkl"
            test = os.path.join("BRANEnet_TF_target_pred", ts)

            train = pickle.load(open(train, "rb"))
            train_samples, train_labels = train['edges'], train['labels']
            test = pickle.load(open(test, "rb"))    
            test_samples, test_labels = test['edges'], test['labels']

            embedding_file = emb

            embeddings = {}
            with open(embedding_file, 'r') as fin:
                num_of_nodes, dim = fin.readline().strip().split()
                for line in fin.readlines():
                    tokens = line.strip().split()
                    embeddings[tokens[0]] = [float(v) for v in tokens[1:]]

            for i in embeddings.copy() :
                if np.sum(embeddings[i]) == 0:
                    embeddings.pop(i)

            train_features,train_labels = extract_feature_vectors_from_embeddings(edges=train_samples.copy(),lab=train_labels.copy(), embeddings=embeddings, binary_operator=args.op)

            test_features,test_labels = extract_feature_vectors_from_embeddings(edges=test_samples.copy(),lab=test_labels.copy(),embeddings=embeddings,binary_operator=args.op)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(train_features, train_labels)

            #print("....................testing performance ................")
            train_preds = clf.predict_proba(train_features)[:, 1]
            test_preds = clf.predict_proba(test_features)[:, 1]
            zipped = list(zip(test_labels, test_preds))
            #output_file1 =  embedding_file  + "_"+ str(op)+ "_"+ str('_lp_preds')
            #np.savetxt(output_file1, zipped, fmt='%i,%i')
            train_roc = roc_auc_score(y_true=train_labels, y_score=train_preds)
            test_roc = roc_auc_score(y_true=test_labels, y_score=test_preds)
            pr, re, thresholds = precision_recall_curve(test_labels, test_preds)
            aupr = auc(re,pr)
            print("Train: "+ str(t+1) + "; AUPR = " + str(aupr) + "; AUROC = " + str(test_roc))
            #scores[args.op]['AUPR'].append(aupr)
            #scores[args.op]['AUROC'].append(test_roc)

            fp.write("{} {} {} {} {}\n".format(str(t+1) ,embedding_file, args.op, aupr,test_roc))    
    
    print("Done!")
    end = time.time()
    print("Time in Seconds:" + str(format(end - start,".2f")))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
