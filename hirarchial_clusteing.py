import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import fetch_data
import simple_svm_code
import clean_data
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering


def cluster_based_on_mean_varience():
    # Sample 2D data: 16 points with 2 features each
    data = np.array([
        [24.63224025974026, 0.010730144041626127],
        [24.740452586206896, 0.015607777684782166],
        [24.79863064396743, 0.021924567919510925],
        [24.829684278350516, 0.044774303866145566],
        [24.81512230919765, 0.03654384210606981],
        [24.855197026022303, 0.047276588165194575],
        [24.887356396866842, 0.05007665256762045],
        [24.918990398603434, 0.27744086050149996],
        [24.94158807035432, 0.4239790429173107],
        [24.961728229236623, 0.37624497493736975],
        [25.01452584531675, 0.4963404651239209],
        [25.05539385376192, 0.700876854214116],
        [ 25.06857599730685, 0.9139138641308445],
        [25.078969977642924, 0.8874356902021717],
        [25.09949970131422, 0.8382729535473342],
        [25.122667981405833, 0.8002617735181986],
    ])

    labels=['00','01','02','03','10','11','12','13','20','21','22','23','30','31','32','33']

    # Perform hierarchical clustering (Ward's method is good for Euclidean data)
    linkage_matrix = linkage(data, method='single')
    

    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=[i for i in labels])
    plt.title("Hierarchical Clustering Dendrogram (2D Data)")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

def get_mean_by_label(some_2dim_list,labels):
    mean_cord=[]
    corresponding_label=[]
    
    for every_label in (set(labels)):
        sum_x=0
        sum_y=0
        count=0
        for i in range(len(some_2dim_list)):
            if labels[i]==every_label:
                count=count+1
                sum_x=sum_x+some_2dim_list[i][0]
                sum_y=sum_y+some_2dim_list[i][1]

        mean_x_cord=(sum_x/count)
        mean_y_cord=(sum_y/count)
        mean_cord.append([mean_x_cord,mean_y_cord])
        corresponding_label.append(every_label)

    return [mean_cord,corresponding_label]

        
            
        
                   
    

def clusterize_2d_data():
    coordinates,labels=fetch_data.get_2d_data()
    mean_coordinates,labels=get_mean_by_label(coordinates,labels)
    mean_coordinates=np.array(mean_coordinates)
    linkage_matrix=linkage(mean_coordinates,method='single')
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix,labels=labels)
    plt.show()
    plt.figure(figsize=(10,5))
    plt.scatter(mean_coordinates[:,0],mean_coordinates[:,1],c=labels)
    
    plt.show()
    

def task_aware_clustering():
    classifier=simple_svm_code.get_model()
    Train_data,labels=clean_data.get_cleaned_2d_data()
    probabilities=classifier.predict_proba(Train_data)
    classwise_probs=defaultdict(list)
    for p, label in zip(probabilities,labels):
        classwise_probs[label].append(p)

    mean_probs={
        label:np.mean(classwise_probs[label],axis=0)
        for label in classwise_probs
        }

    print(mean_probs)
    sorted_labels=sorted(mean_probs.keys())
    #vectors = np.array(list(mean_probs.values()))
    vectors = np.array([mean_probs[label] for label in sorted_labels])

    kmeans=KMeans(n_clusters=4,random_state=42)
    group_ids=kmeans.fit_predict(vectors)
    label_to_group = dict(zip(sorted_labels, group_ids))
    print(label_to_group)





def task_aware_clustering_sample_level():
    # Get model and data
    classifier = simple_svm_code.get_model()
    Train_data, labels = clean_data.get_cleaned_2d_data()

    # Get per-sample predicted probability vectors
    probabilities = classifier.predict_proba(Train_data)  # shape: (n_samples, n_classes)

    # Perform KMeans on the probability vectors
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_ids = kmeans.fit_predict(probabilities)

    # Count label occurrences in each cluster
    cluster_label_count = defaultdict(lambda: defaultdict(int))

    for cluster_id, label in zip(cluster_ids, labels):
        cluster_label_count[cluster_id][label] += 1

    # Print counts in a readable way
    for cluster_id in sorted(cluster_label_count.keys()):
        print(f"\nCluster {cluster_id}:")
        for label, count in sorted(cluster_label_count[cluster_id].items()):
            print(f"  Label {label}: {count} samples")

    return cluster_ids, labels, cluster_label_count


def task_aware_clustering_sample_level_DBSCAN():
    # Get model and data
    classifier = simple_svm_code.get_model()
    Train_data, labels = clean_data.get_cleaned_2d_data()

    # Get per-sample predicted probability vectors
    probabilities = classifier.predict_proba(Train_data)  # shape: (n_samples, n_classes)

    dbscan = DBSCAN(eps=0.01, min_samples=5)  # Tune eps!
    cluster_ids = dbscan.fit_predict(probabilities)
    

    # Count label occurrences in each cluster
    cluster_label_count = defaultdict(lambda: defaultdict(int))

    for cluster_id, label in zip(cluster_ids, labels):
        cluster_label_count[cluster_id][label] += 1

    # Print counts in a readable way
    for cluster_id in sorted(cluster_label_count.keys()):
        print(f"\nCluster {cluster_id}:")
        for label, count in sorted(cluster_label_count[cluster_id].items()):
            print(f"  Label {label}: {count} samples")

    return cluster_ids, labels, cluster_label_count


def task_aware_clustering_sample_level_Spectral_Clustering():
    # Get model and data
    classifier = simple_svm_code.get_model()
    Train_data, labels = clean_data.get_cleaned_2d_data()

    # Get per-sample predicted probability vectors
    probabilities = classifier.predict_proba(Train_data)  # shape: (n_samples, n_classes)

    spectral = SpectralClustering(n_clusters=4, affinity='rbf', random_state=42)
    cluster_ids = spectral.fit_predict(probabilities)

    # Count label occurrences in each cluster
    cluster_label_count = defaultdict(lambda: defaultdict(int))

    for cluster_id, label in zip(cluster_ids, labels):
        cluster_label_count[cluster_id][label] += 1

    # Print counts in a readable way
    for cluster_id in sorted(cluster_label_count.keys()):
        print(f"\nCluster {cluster_id}:")
        for label, count in sorted(cluster_label_count[cluster_id].items()):
            print(f"  Label {label}: {count} samples")

    return cluster_ids, labels, cluster_label_count


        

task_aware_clustering_sample_level()       
    
    
    

    
