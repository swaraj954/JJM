import fetch_data
from sklearn.svm import OneClassSVM
import numpy as np


def get_cleaned_2d_data(cluster_mapping=''):
    Data_points,labels = fetch_data.get_2d_data(cluster_mapping)
    Data_points=np.array(Data_points)
    labels=np.array(labels)
    model=OneClassSVM(kernel='rbf', gamma='auto',nu=0.01)
    model.fit(Data_points)
    preds=model.predict(Data_points)

    Clean_Data = Data_points[preds == 1]
    corrosponding_labels=labels[preds == 1]

    return [Clean_Data, corrosponding_labels]
    

    
    
    
