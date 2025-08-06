from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import fetch_data
import clean_data
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

Global_Cluster=''



mppng = {
    0:  "00",
    1:  "01",
    2:  "02",
    3:  "03",
    4:  "10",
    5:  "11",
    6:  "12",
    7:  "13",
    8:  "20",
    9:  "21",
    10: "22",
    11: "23",
    12: "30",
    13: "31",
    14: "32",
    15: "33"
}


def label_confusion_stats(cm,label_idx):
    row=cm[label_idx]
    total = row.sum()
    percentages=(row/total)*100

    lbl=mppng[label_idx]
    print(f" True Label {lbl} - Total Samples : {total}")

    for pred_label, pct in enumerate(percentages):
        count = row[pred_label]
        print(f" Predicted as {mppng[pred_label]}: {count} samples ({pct:.2f}%)")

    return total, percentages
        
    

def normal_KNN():
    #flow_readings,labels=fetch_data.get_2d_data()
    flow_readings,labels=clean_data.get_cleaned_2d_data()
    
   
    

##     Best Parameters:
##{'metric': 'manhattan', 'n_neighbors': 18, 'weights': 'uniform'}

    X_train, X_test,y_train,y_test = train_test_split(flow_readings,labels,test_size=0.2,random_state=42)


    knn=KNeighborsClassifier(n_neighbors=18,weights= 'uniform',metric= 'manhattan')
    knn.fit(X_train,y_train)

    predictions=knn.predict(X_test)

    accuracy = accuracy_score(y_test,predictions)
    print(classification_report(y_test,predictions))
    print(f"Accuracy: {accuracy:.2f}")

    class_names=np.unique(labels)
    cm= confusion_matrix(y_test,predictions,labels=class_names)

    cm_df = pd.DataFrame(cm,index=class_names,columns=class_names)
    print(cm_df)

    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
    plt.xlabel('predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


    for label_idx in range(cm.shape[0]):
        percentages=label_confusion_stats(cm,label_idx)
        print("--"*40)



def get_best_model():
    print("IN GET BEST MODEL")
    #flow_readings,labels=fetch_data.get_2d_data()
    flow_readings,labels=clean_data.get_cleaned_2d_data()
##    flow_readings=flow_readings[(1192-300):(1492)]
##    labels=labels[(1192-300):(1492)]

    X_train, X_test,y_train,y_test = train_test_split(flow_readings,labels,test_size=0.2,random_state=42)

    param_grid = {
        'n_neighbors':[10,15,18],
        'weights':['uniform','distance'],
        'metric':['euclidean','manhattan']
        }

    grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5, return_train_score=True)
    grid.fit(X_train,y_train)

    print("Results for all combinations")
    results=grid.cv_results_
    for mean_score,params in zip(results['mean_test_score'],results['params']):
        print(f"Accuracy:{mean_score:.4f}  Params:{params}")

    print("\n Best Parameters:")
    print(grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")

    best_model=grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Test set performance:")
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    unique_labels=np.unique(labels)
    cm=confusion_matrix(y_test,y_pred,labels=unique_labels)
    cm_df=pd.DataFrame(cm,index=unique_labels,columns=unique_labels)
    print(cm_df)


    for label_idx in range(cm.shape[0]):
            percentages=label_confusion_stats(cm,label_idx)
            print("--"*40)


    return best_model

def get_best_models_per_class():
    flow_readings,labels=clean_data.get_cleaned_2d_data(Global_Cluster)
    X_train, X_test,y_train,y_test = train_test_split(flow_readings,labels,test_size=0.2,random_state=42)

    knn_models={}

    unique_labels=np.unique(labels)
    print(unique_labels)

    param_grid = {
        'n_neighbors':list(range(1,20)),
        'weights':['uniform','distance'],
        'metric':['euclidean','manhattan']

        }

    print("Training individual models....")

    for every_label in unique_labels:
        print(f"Processing label :{every_label}")

        grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
        grid.fit(X_train, y_train)

        best_model= None
        best_score = -1
        best_stats= None

        for i in tqdm(range(len(grid.cv_results_['params']))):

            params= grid.cv_results_['params'][i]
            model = KNeighborsClassifier(**params)
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            prec_arr,rec_arr,f1_arr,support_arr=precision_recall_fscore_support(y_test,y_pred,labels=[every_label],average=None,zero_division=0)

            prec=prec_arr[0]
            rec=rec_arr[0]
            f1=f1_arr[0]

            if f1> best_score:
                best_score = f1
                best_model=model
                best_stats = {
                    'precision':prec,
                    'recall': rec,
                    'f1':f1,
                    'params':params
                    }

        if best_model:
            knn_models[every_label] = best_model
            print(f"Best model for {every_label} ->{best_stats['params']}")
            print(f"   F1: {best_stats['f1']:.4f} | Precision: {best_stats['precision']:.4f} | Recall: {best_stats['recall']:.4f}")
        else:
            print(f" No valid model found for label {every_label} (maybe no test samples?)")

    return knn_models




                    
def get_data_for_adaptive_KNN():
    MoDeLs = get_best_models_per_class()
    flow_readings,labels=clean_data.get_cleaned_2d_data(Global_Cluster)
    new_flow_readings=[]
    print("Generating new data")
    for every_flow_reading in tqdm(flow_readings):
        representation=[]
        for every_model in MoDeLs.values():
            representation.append(every_model.predict([every_flow_reading])[0])

        new_flow_readings.append(representation)

    
    print(new_flow_readings[0:5])
    print(labels[0:5])
    return (new_flow_readings,labels)
        
            
    
        
def KNN_on_KNN():
    flow_readings,labels=get_data_for_adaptive_KNN()
    encoder = OrdinalEncoder()
    flow_readings = encoder.fit_transform(flow_readings)

    X_train, X_test,y_train,y_test = train_test_split(flow_readings,labels,test_size=0.2,random_state=42)

    param_grid = {
        'n_neighbors':[10,15,18],
        'weights':['uniform','distance'],
        'metric':['euclidean','manhattan']
        }

    grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5, return_train_score=True)
    grid.fit(X_train,y_train)

    print("Results for all combinations")
    results=grid.cv_results_
    for mean_score,params in zip(results['mean_test_score'],results['params']):
        print(f"Accuracy:{mean_score:.4f}  Params:{params}")

    print("\n Best Parameters:")
    print(grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")

    best_model=grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Test set performance:")
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))






def RandomForest_on_KNN():
    flow_readings, labels = get_data_for_adaptive_KNN()

    
    encoder = OrdinalEncoder()
    flow_readings = encoder.fit_transform(flow_readings)

   
    X_train, X_test, y_train, y_test = train_test_split(
        flow_readings, labels, test_size=0.2, random_state=42
    )

    # Random Forest hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }

    print("Training Random Forest with GridSearchCV...")
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, return_train_score=True)
    grid.fit(X_train, y_train)

    print("\nResults for all combinations:")
    results = grid.cv_results_
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f"Accuracy: {mean_score:.4f}  Params: {params}")

    print("\nBest Parameters:")
    print(grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nTest set performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    
    
KNN_on_KNN()    
        
#RandomForest_on_KNN()



        
         

        
    
    
    







    






