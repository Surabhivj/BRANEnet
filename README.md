# BRANEnet
Embedding Multilayer Networks for Omics Data Integration
<img width="842" alt="ION" src="https://user-images.githubusercontent.com/47250394/164945504-d8f743f6-00dc-4964-8b37-b37d9fa694bf.png">


# To reproduce the results, please run BRANEnet.ipynb 

# To compute embedding for user input network, Please run following command. 

multilayer_networkfile = multilayer/single layer network file in 'gml' or 'edgelist' format

T = widow size 

d = embedding dimension

type = 'gml' or 'edgelist'

 ```
 python BRANEnet.py --multilayer_networkfile network_file.gml --T 3 --d 128 --type 'gml'
 
 ```
