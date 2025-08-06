import fetch_data
import clean_data
from fetch_data import cluster1_mapping_data as clus1
from fetch_data import cluster2_mapping_data as clus2
from fetch_data import cluster3_mapping_data as clus3
from fetch_data import cluster4_mapping_data as clus4
from fetch_data import cluster5_mapping_data as clus5
from fetch_data import cluster6_mapping_data as clus6
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score







#X_train, X_test,y_train,y_test=train_test_split(Train_data,labels,test_size=0.2,random_state=42)

#alt fetch
#X_train,X_test,y_train,y_test=fetch_data.fetch_the_data_1d_80_20(fetch_data.file_list_f1,fetch_data.kclus4)


#alt fetch 2
#X_train,X_test,y_train,y_test=fetch_data.fetch_the_data_1d_20_80(fetch_data.file_list_f1,fetch_data.kclus4)

def get_model(cluster_mapping=''):
    #2D fetch
    #Train_data,labels = fetch_data.get_2d_data(cluster_mapping)
    Train_data,labels=clean_data.get_cleaned_2d_data(cluster_mapping)
    
    #print(len(labels))
    X_train,X_test,y_train,y_test=train_test_split(Train_data,labels,test_size=0.2,random_state=42)


    
    clf=SVC(probability=True,kernel='rbf',C=1000,gamma=100,decision_function_shape='ovr')
    clf.fit(X_train,y_train)
    #print("1")
    #y_pred=clf.predict(X_test)


    #print("Confusion Matrix:")
    #print(confusion_matrix(y_test,y_pred))

    #print("\nClassification Report:")
    #print(classification_report(y_test,y_pred))

    #print("Accuracy:", accuracy_score(y_test,y_pred))
    return clf


def train_svm_1():
    #2D fetch
    #Train_data , labels = fetch_data.fetch_the_data_1d(fetch_data.file_list_f1)
    Train_data,labels = fetch_data.get_2d_data(fetch_data.TA_clus4_spec)
    #Train_data,labels=clean_data.get_cleaned_2d_data(fetch_data.TA_clus4)
    #print(len(labels))
    X_train,X_test,y_train,y_test=train_test_split(Train_data,labels,test_size=0.2,random_state=42)


    print("1")
    clf=SVC(kernel='rbf',C=1000,gamma=100,decision_function_shape='ovr')
    clf.fit(X_train,y_train)
    print("1")
    y_pred=clf.predict(X_test)


    print("Confusion Matrix:")
    print(confusion_matrix(y_test,y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test,y_pred))

    print("Accuracy:", accuracy_score(y_test,y_pred))

  
    


      





