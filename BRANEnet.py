import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import pandas as pd
import os
import argparse
import time




pd.options.mode.chained_assignment = None  # default='warn'

#reading multilayer network

def parse_args():
    parser = argparse.ArgumentParser(description="performs PPI prediction and evaluation")

    parser.add_argument('--multilayer_networkfile', nargs='?',
                        help='multilayer_networkfile',type=str)
    parser.add_argument('--T', nargs='?',
                        help= 'window size',type=int)
    parser.add_argument('--d', nargs='?',
                        help= 'embedding dimension',type=int)
    parser.add_argument('--type', default = 'gml',
                        help= 'network file type (gml or edgelist)',type=str)
  
    return parser.parse_args()


def main(args):
    start = time.time()
    import functions as f

    w = args.T
    d = args.d
    
    typ = args.type

    print("Computing embedding for: w=" + str(w) + ";d=" + str(d))


    out_folder = "BRANEnet_emb"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # read multilayer network
    if typ == 'gml':
        G = nx.read_gml(args.multilayer_networkfile)
    
    elif typ == 'edgelist':
        G = nx.read_edgelist(args.multilayer_networkfile)
    
    else:
        print("Incorrect file format")
    
    A = nx.adjacency_matrix(G,nodelist = G.nodes)
    
    name = os.path.split(args.multilayer_networkfile)[1]
    name = name.split(".")[0]

    #compute PPMI matrix
    M = f.PPMI_matrix(A,w,1)

    #learn embedding
    emb = f.embedd(M,d)
    emb = pd.DataFrame(data=emb,index = G.nodes)
    idx = emb.index
    emb = emb.sort_index()

    outfile = os.path.join(out_folder, 'BRANet_' + name + '_w_' + str(w) + '_d_' +
                           str(d) + '.emb')
    emb.to_csv(outfile,sep = ' ')

    with open(outfile) as f:
        lines = f.readlines()
    txt = str(len(G)) + ' ' + str(d) +'\n'
    lines[0] = txt 
    with open(outfile, "w") as ff:
        ff.writelines(lines)
    ff.close()

    print("Done!")
    end = time.time()
    print("Time in Seconds:" + str(format(end - start,".2f")))
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
    