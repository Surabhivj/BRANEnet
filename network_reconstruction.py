#LOAD PACKAGES
import pandas as pd
import networkx as nx
import numpy as np
import functions as f
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
pd.options.mode.chained_assignment = None  # default='warn'
import pickle


node_dict = pd.read_pickle("node_labels_all.pickle")
node_dict2 = dict((v,k) for k,v in node_dict.items())
gene_sel = list(node_dict2.values())
gene_sel = [i.replace('"', '') for i in gene_sel]


branet_emb = pd.read_csv("BRANEnet_emb/BRANet_Edgelist_R_w_3_d_128.emb", sep=' ',index_col=0,header = None, skiprows =1).sort_index()
branet_emb = pd.DataFrame(branet_emb.values, index = list(branet_emb.index))

multinet_emb = pd.read_csv("baseline_emb/MultiNet_R.emb", sep=' ',index_col=0,header = None, skiprows =1).sort_index()
multinet_emb = pd.DataFrame(multinet_emb.values, index = list(multinet_emb.index))

branexp_emb = pd.read_csv("baseline_emb/BRANE_exp_R.emb", sep=' ',index_col=0,header = None, skiprows =1).sort_index()
branexp_emb = pd.DataFrame(branexp_emb.values, index = list(branexp_emb.index))

deepnf_emb = pd.DataFrame(pd.read_pickle("baseline_emb/deepNF_R.pckl"))
deepnf_emb = deepnf_emb.rename(index=node_dict2)

ohmnet_emb = pd.read_csv("baseline_emb/ohmnet_R.emb", sep=' ',index_col=0,header = None, skiprows =1).sort_index()
ohmnet_emb = pd.DataFrame(ohmnet_emb.values, index = list(ohmnet_emb.index))
ohmnet_emb = ohmnet_emb.rename(index=node_dict2)

branet_emb = branet_emb[branet_emb.index.isin(gene_sel)].sort_index()
multinet_emb = multinet_emb[multinet_emb.index.isin(gene_sel)].sort_index()
deepnf_emb = deepnf_emb[deepnf_emb.index.isin(gene_sel)].sort_index()
branexp_emb = branexp_emb[branexp_emb.index.isin(gene_sel)].sort_index()
ohmnet_emb = ohmnet_emb[ohmnet_emb.index.isin(gene_sel)].sort_index()

#integrated_emb = pd.concat([h4k12_emb, rnaseq_emb,meta_emb], axis=1)
string_ref_net = nx.read_edgelist('string_file.txt')
string_ref_net = nx.Graph(string_ref_net.subgraph(gene_sel))
biogrid_ref_net = nx.read_edgelist('biogrid_refnet.txt')
biogrid_ref_net = nx.Graph(biogrid_ref_net.subgraph(gene_sel))
tf_ref_net = nx.read_edgelist('tf_deg_sgd.txt')
tf_ref_net = nx.Graph(tf_ref_net.subgraph(tf_ref_net))
ref_net = nx.compose_all([string_ref_net,biogrid_ref_net,tf_ref_net])
#ref_net = biogrid_ref_net   

#ref_net = biogrid_ref_net
emb_wt_dat_branet_biogrid = f.net_recons(branet_emb,ref_net)
emb_wt_dat_branexp_biogrid = f.net_recons(branexp_emb,ref_net)
emb_wt_dat_multinet_biogrid = f.net_recons(multinet_emb,ref_net)
emb_wt_dat_deepnf_biogrid = f.net_recons(deepnf_emb,ref_net)
emb_wt_dat_ohmnet_biogrid = f.net_recons(ohmnet_emb,ref_net)


branet_mcc_at_k = f.mcc_t(emb_wt_dat_branet_biogrid)
branexp_mcc_at_k = f.mcc_t(emb_wt_dat_branexp_biogrid)
multinet_mcc_at_k = f.mcc_t(emb_wt_dat_multinet_biogrid)
deepnf_mcc_at_k = f.mcc_t(emb_wt_dat_deepnf_biogrid)
ohmnet_mcc_at_k = f.mcc_t(emb_wt_dat_ohmnet_biogrid)

k = branet_mcc_at_k['thres']

branet_mcc_at_k = branet_mcc_at_k['dat_mcc']
branexp_mcc_at_k = branexp_mcc_at_k['dat_mcc']
multinet_mcc_at_k = multinet_mcc_at_k['dat_mcc']
deepnf_mcc_at_k = deepnf_mcc_at_k['dat_mcc']
ohmnet_mcc_at_k = ohmnet_mcc_at_k['dat_mcc']

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#params = {'text.usetex': True, 'text.latex.preamble' : [r'\usepackage{siunitx}', r'\usepackage{amsmath}', r'\usepackage{palatino}', r'\usepackage{amssymb}'], 'font.family':'serif','font.serif':'Palatino'}
#sns.set(params)
#plt.rcParams.update(params)


from matplotlib.ticker import NullFormatter  # useful for `logit` scale
plt.rcParams['font.family'] = 'sans-serif'

fontsize=25

plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rc('lines', markersize=6)
plt.rc('lines', markeredgewidth=1)
plt.rc('markers', fillstyle='full')
plt.rc('lines', linewidth=1)
plt.rc('grid', linewidth=0.5, linestyle=':')

# Fixing random state for reproducibility
np.random.seed(19680801)


########### Global Settings ###########
y1_bar_color='blue'
y2_bar_color='black'
y3_bar_color='red'
seaborn_plot_dict={'axes.spines.bottom': True,
				  'axes.spines.left': True,
				  'axes.spines.top': False,
				  'axes.spines.right': False,
				  'axes.edgecolor': '.005',
				  'xtick.bottom': True,
 				  'ytick.left': True,
 				  'xtick.direction': 'out',
 				   'ytick.direction': 'out',
				  }

flatui = ["#A51C30", "#3498db",  "#739c3e" , "#cc9764", "#9b59b6"] #"#934f28", "#34495e"]
flatui.reverse()
sns.set_palette(sns.color_palette(sns.color_palette(flatui)))
#sns.set_palette(sns.color_palette(sns.color_palette("muted", 7), n_colors=7,))

#######################################
fig = plt.figure(figsize=(8,6))
#######################################

#integration_added_val


with sns.axes_style("whitegrid", seaborn_plot_dict):
    '''
    line1, = plt.plot(emb_wt_dat_integrated_biogrid_r, emb_wt_dat_integrated_biogrid_p, '-', linewidth=3,  label='BRANet')
    line2, = plt.plot(emb_wt_dat_branexp_biogrid_r, emb_wt_dat_branexp_biogrid_p, '-', linewidth=3,  label='BraneExp')
    line3, = plt.plot(emb_wt_dat_multinet_biogrid_r, emb_wt_dat_multinet_biogrid_p, '-', linewidth=3,  label='MultiNet')
    '''
    line1, = plt.plot(k, ohmnet_mcc_at_k, 'o-', linewidth=3,  label='OhmNet',markersize=12)
    line2, = plt.plot(k, multinet_mcc_at_k, 's-', linewidth=3,  label='MultiNet',markersize=12)
    line3, = plt.plot(k, deepnf_mcc_at_k, 'p-', linewidth=3,  label='deepNF',markersize=12)
    line4, = plt.plot(k, branexp_mcc_at_k, '^-', linewidth=3,  label='BRANE-Exp',markersize=12)
    line5, = plt.plot(k, branet_mcc_at_k, '*-', linewidth=3,  label='BRANEnet',markersize=12)
    
    plt.grid(True)
    plt.ylabel('MCC@threshold', fontsize=fontsize,  labelpad=10)
    plt.xlabel('threshold', fontsize=fontsize, labelpad=10)
    plt.grid(True)
    
#pad=4, h_pad=0.2,rect= (0, 0.5, 1, 0.5))
leg=fig.legend([line1,line2,line3,line4, line5], 
               ["OhmNet" ,
                'MultiNet','deepNF','BRANE-Exp','BRANEnet' ], bbox_to_anchor=(0.5, -0.20, 0, 0),
               loc='lower center', ncol=3, prop={'size': 25}, frameon=False, columnspacing=0.20)
plt.tight_layout(rect= (0, 0, 1, 1))
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.show()
fig.savefig('mcc.pdf',bbox_inches='tight')


branet_mcc_at_k = f.pr_at_k(emb_wt_dat_branet_biogrid)
branexp_mcc_at_k = f.pr_at_k(emb_wt_dat_branexp_biogrid)
multinet_mcc_at_k = f.pr_at_k(emb_wt_dat_multinet_biogrid)
deepnf_mcc_at_k = f.pr_at_k(emb_wt_dat_deepnf_biogrid)
ohmnet_mcc_at_k = f.pr_at_k(emb_wt_dat_ohmnet_biogrid)

k = branet_mcc_at_k['k']

branet_mcc_at_k = branet_mcc_at_k['precision_dist']
branexp_mcc_at_k = branexp_mcc_at_k['precision_dist']
multinet_mcc_at_k = multinet_mcc_at_k['precision_dist']

deepnf_mcc_at_k = deepnf_mcc_at_k['precision_dist']
ohmnet_mcc_at_k = ohmnet_mcc_at_k['precision_dist']

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functions as f

sns.set()
#params = {'text.usetex': True, 'text.latex.preamble' : [r'\usepackage{siunitx}', r'\usepackage{amsmath}', r'\usepackage{palatino}', r'\usepackage{amssymb}'], 'font.family':'serif','font.serif':'Palatino'}
#sns.set(params)
#plt.rcParams.update(params)


from matplotlib.ticker import NullFormatter  # useful for `logit` scale
plt.rcParams['font.family'] = 'sans-serif'

fontsize=25

plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rc('lines', markersize=6)
plt.rc('lines', markeredgewidth=1)
plt.rc('markers', fillstyle='full')
plt.rc('lines', linewidth=1)
plt.rc('grid', linewidth=0.5, linestyle=':')

# Fixing random state for reproducibility
np.random.seed(19680801)


########### Global Settings ###########
y1_bar_color='blue'
y2_bar_color='black'
y3_bar_color='red'
seaborn_plot_dict={'axes.spines.bottom': True,
				  'axes.spines.left': True,
				  'axes.spines.top': False,
				  'axes.spines.right': False,
				  'axes.edgecolor': '.005',
				  'xtick.bottom': True,
 				  'ytick.left': True,
 				  'xtick.direction': 'out',
 				   'ytick.direction': 'out',
				  }

flatui = ["#A51C30", "#3498db",  "#739c3e" , "#cc9764", "#9b59b6"]# "#934f28", "#34495e"]
flatui.reverse()
sns.set_palette(sns.color_palette(sns.color_palette(flatui)))
#sns.set_palette(sns.color_palette(sns.color_palette("muted", 7), n_colors=7,))

#######################################
fig = plt.figure(figsize=(8,6))
#######################################

#integration_added_val


with sns.axes_style("whitegrid", seaborn_plot_dict):
    '''
    line1, = plt.plot(emb_wt_dat_integrated_biogrid_r, emb_wt_dat_integrated_biogrid_p, '-', linewidth=3,  label='BRANet')
    line2, = plt.plot(emb_wt_dat_branexp_biogrid_r, emb_wt_dat_branexp_biogrid_p, '-', linewidth=3,  label='BraneExp')
    line3, = plt.plot(emb_wt_dat_multinet_biogrid_r, emb_wt_dat_multinet_biogrid_p, '-', linewidth=3,  label='MultiNet')
    '''
    line1, = plt.plot(k, ohmnet_mcc_at_k, 'o-', linewidth=3,  label='OhmNet',markersize=10)
    line2, = plt.plot(k, multinet_mcc_at_k, 's-', linewidth=3,  label='MultiNet',markersize=10)
    line3, = plt.plot(k, deepnf_mcc_at_k, 'p-', linewidth=3,  label='deepNF',markersize=10)
    line4, = plt.plot(k, branexp_mcc_at_k, '^-', linewidth=3,  label='BRANE-Exp',markersize=10)    
    line5, = plt.plot(k, branet_mcc_at_k, '*-', linewidth=3,  label='BRANEnet',markersize=10)
    
    plt.grid(True)
    plt.ylabel('Precision@k', fontsize=fontsize,  labelpad=10)
    plt.xlabel('k', fontsize=fontsize, labelpad=10)
    plt.grid(True)
    

plt.xticks([1,100,200,300,400,500],['1','100','200','300','400','500'])    
plt.tight_layout(pad=4, h_pad=0.2,rect=  (0, 0, 1, 1))


leg=fig.legend([line1,line2,line3,line4, line5], 
               ["OhmNet" ,
                'MultiNet','deepNF','BRANE-Exp','BRANEnet' ], bbox_to_anchor=(0.5, -0.20, 0, 0),
               loc='lower center', ncol=3, prop={'size': 25}, frameon=False, columnspacing=0.20)
plt.tight_layout(rect= (0, 0, 1, 1))
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.show()
fig.savefig('pr_at_k.pdf',bbox_inches='tight')
