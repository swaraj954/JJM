from fetch_data import *
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon,squareform
from scipy.cluster.hierarchy import linkage, dendrogram


resolution=1000

Train_data , labels = fetch_the_data_1d(file_list_f1)


Train_data=[x for y in Train_data for x in y]



Train_data=[x for x in Train_data]

data = pd.DataFrame({'value':Train_data,'label':Labels})

kde_grid = np.linspace(data['value'].min() - 1,data['value'].max()+1,resolution)
unique_labels = sorted(data['label'].unique())
kde_dict={}

#calc kde for sampled 1000 points
for label in unique_labels:
    values=data[data['label'] == label]['value'].astype(float).values
    kde= gaussian_kde(values)
    kde_vals = kde(kde_grid)
    kde_vals /= kde_vals.sum()
    kde_dict[label]=kde_vals

print(kde_dict)
#create a 16x 16 matrix init to zeroo
n=len(unique_labels)
dist_matrix = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        p=kde_dict[unique_labels[i]]
        q=kde_dict[unique_labels[j]]
        d= jensenshannon(p,q)
        dist_matrix[i,j] = dist_matrix[j,i]=d


#print(dist_matrix)

linked=linkage(squareform(dist_matrix),method='average')

plt.figure(figsize=(12,6))
denddrogram(linked,labels=uniques_labels,leaf_rotation=90)

plt.title("hira clustering")
plt.xlabel("label")
plt.ylabel("JS DIST")
plt.tight_layout()
plt.show()
    


    
    










