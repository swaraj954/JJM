import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

sys.stdout=open("redundant_data_detection(1)",'w')
highest_min_timestamp=60




file_list = [
    "f2\predicted_digits_00_f2.csv",
    "f2\predicted_digits_01_f2.csv",
    "f2\predicted_digits_02_f2.csv",
    "f2\predicted_digits_03_f2.csv",
    "f2\predicted_digits_10_f2.csv",
    "f2\predicted_digits_11_f2.csv",
    "f2\predicted_digits_12_f2.csv",
    "f2\predicted_digits_13_f2.csv",
    "f2\predicted_digits_20_f2.csv",
    "f2\predicted_digits_21_f2.csv",
    "f2\predicted_digits_22_f2.csv",
    "f2\predicted_digits_23_f2.csv",
    "f2\predicted_digits_30_f2.csv",
    "f2\predicted_digits_31_f2.csv",
    "f2\predicted_digits_32_f2.csv",
    "f2\predicted_digits_33_f2.csv"
]


print("program Starting...")



for min_timestamp in range(0,highest_min_timestamp):
    print("For Timestamp:"+str(min_timestamp))

    Train_data=[]
    Labels=[]


    for each_file in file_list:
        #print("processing:"+str(each_file))
        
        dataframe=pd.read_csv(each_file)
        dataframe = dataframe[['frame' , 'predicted_digit' ,'last_4_digits']]
        dataframe=dataframe[dataframe['frame']>min_timestamp]
        #Extracting the 21,22 char from each file name
        label=each_file[20:22]
        for each_point in dataframe['last_4_digits'].tolist():
            temp_list=[each_point]
            Train_data.append(temp_list)
            Labels.append(label)
        #print("-----------------------")


    #print("First list in train data:")
    #print(len(Train_data))
    #print("labels")
    #print(len(Labels))
    X_train,X_test, y_train, y_test = train_test_split(Train_data, Labels,test_size=0.2,random_state=42)
    clf=SVC(kernel='rbf',C=1000,gamma=100,decision_function_shape='ovr')
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test,y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test,y_pred))
    print("Accuracy:", accuracy_score(y_test,y_pred))
    print("------------------------------------------------------")
    print("--------------------------------------------------------")
    


        

        

        
        
        
        
        
        

    
    
