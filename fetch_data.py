import pandas as pd



file_list_f1 = [
    "f1\predicted_digits_00_f1.csv",
    "f1\predicted_digits_01_f1.csv",
    "f1\predicted_digits_02_f1.csv",
    "f1\predicted_digits_03_f1.csv",
    "f1\predicted_digits_10_f1.csv",
    "f1\predicted_digits_11_f1.csv",
    "f1\predicted_digits_12_f1.csv",
    "f1\predicted_digits_13_f1.csv",
    "f1\predicted_digits_20_f1.csv",
    "f1\predicted_digits_21_f1.csv",
    "f1\predicted_digits_22_f1.csv",
    "f1\predicted_digits_23_f1.csv",
    "f1\predicted_digits_30_f1.csv",
    "f1\predicted_digits_31_f1.csv",
    "f1\predicted_digits_32_f1.csv",
    "f1\predicted_digits_33_f1.csv"
]


file_list_f2 = [
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

file_list_combined = [
    "Combined Data\\00.csv",
    "Combined Data\\01.csv",
    "Combined Data\\02.csv",
    "Combined Data\\03.csv",
    "Combined Data\\10.csv",
    "Combined Data\\11.csv",
    "Combined Data\\12.csv",
    "Combined Data\\13.csv",
    "Combined Data\\20.csv",
    "Combined Data\\21.csv",
    "Combined Data\\22.csv",
    "Combined Data\\23.csv",
    "Combined Data\\30.csv",
    "Combined Data\\31.csv",
    "Combined Data\\32.csv",
    "Combined Data\\33.csv",
   



    ]

Train_data=[]
Test_Data=[]
Train_Labels=[]
Test_Labels=[]
Labels=[]
Threshold=0.8

cluster1_mapping_data={
    'cluster 1':['00','01','10'],
    'cluster 2':['02','03','11'],
    'cluster 3':['22','23','31','32','33'],
    'cluster 4':['12','13','20','21','30']
    }

cluster2_mapping_data={
    'cluster 1':['00','01','02'],
    'cluster 2':['11','03','12'],
    'cluster 3':['22','23'],
    'cluster 4':['31','32','33'],
    'cluster 5':['20','21'],
    'cluster 6':['10'],
    'cluster 7':['13'],
    'cluster 8':['30']
    }

cluster3_mapping_data={
    'cluster 1':['00','01','02'],
    'cluster 2':['10','03','11','12'],
    'cluster 3':['22','23','21','20','13'],
    'cluster 4':['31','32','33','30']
    }

cluster4_mapping_data={
    'cluster 1':['30','31'],
    'cluster 2':['32','33'],
    'cluster 3':['23'],
    'cluster 4':['20','21'],
    'cluster 5':['22'],
    'cluster 6':['13'],
    'cluster 7':['11','12'],
    'cluster 8':['02','03','10'],
    'cluster 9':['00','01']
    }


cluster5_mapping_data={
    'cluster 1':['00','01','12', '11','02','03','10'],
    'cluster 2':['13','22'],
    'cluster 3':['20','21'],
    'cluster 4':['30','31'],
    'cluster 5':['23'],
    'cluster 6':['32','33'],
    
    }

cluster6_mapping_data={
    'cluster 1':['00','01','12', '11','02','03','10'],
    'cluster 2':['13','22','30','31'],
    'cluster 3':['20','21'],
    'cluster 4':['23'],
    'cluster 5':['32','33'],
    
    }

clus7={
    'cluster 1':['00','10','30','23','13','20'],
    'cluster 2':['01','02','03','11','12','21','22'],
    'cluster 3':['31','32','33']
    }

clus8={
    'cluster 1':['00'],
    'cluster 2':['01','02','03','10','13','11','12'],
    'cluster 3':['31','32','33'],
    'cluster 4':['20','21','22','23'],
    'cluster 5':['30']
    }




kclus2_f2={'Cluster 0': ['00', '01', '02', '03', '10', '11', '12', '13', '20', '21', '23', '30'],
        'Cluster 1': ['22', '31', '32', '33']
        }


    

kclus3_f2={'Cluster 2': ['00', '01', '10', '21'],
        'Cluster 0': ['02', '03', '11', '12', '13', '20', '23', '30'],
        'Cluster 1': ['22', '31', '32', '33']
        }

kclus4_f2={'Cluster 2': ['00'],
        'Cluster 3': ['01', '10', '21'],
        'Cluster 0': ['02', '03', '11', '12', '13', '20', '23', '30'],
        'Cluster 1': ['22', '31', '32', '33']
        }

kclus5_f2={'Cluster 2': ['00'],
        'Cluster 3': ['01', '10', '21'],
        'Cluster 0': ['02', '03', '11', '12', '20', '23', '30'],   
        'Cluster 4': ['13', '31'],
        'Cluster 1': ['22', '32', '33']
        }

kclus6_f2={'Cluster 2': ['00'],
        'Cluster 5': ['01', '10'],
        'Cluster 0': ['02', '03', '11', '12', '20', '23', '30'],    
        'Cluster 4': ['13', '31'],
        'Cluster 3': ['21'],
        'Cluster 1': ['22', '32', '33']
        }

kclus7_f2={'Cluster 2': ['00'],
        'Cluster 5': ['01', '10'],
        'Cluster 0': ['02', '03', '11', '20'],
        'Cluster 6': ['12', '13', '23', '30'],
        'Cluster 3': ['21'],
        'Cluster 1': ['22', '32', '33'],
        'Cluster 4': ['31']
        }

kclus8_f2={'Cluster 2': ['00'],
        'Cluster 3': ['01', '10', '21'],
        'Cluster 0': ['02', '03', '11', '20'],
        'Cluster 6': ['12', '23'],
        'Cluster 4': ['13', '30'],
        'Cluster 5': ['22'],
        'Cluster 7': ['31'],
        'Cluster 1': ['32', '33']
        }


kclus2={'Cluster 0': ['00', '01', '02', '03', '10', '11', '12', '13', '20', '21', '30'],
        'Cluster 1': ['22', '23', '31', '32', '33']
        }

kclus3={'Cluster 2': ['00', '01', '10'],
        'Cluster 0': ['02', '03', '11', '12', '13', '20', '21', '30'],
        'Cluster 1': ['22', '23', '31', '32', '33']
        }

kclus4={'Cluster 2': ['00', '01', '10'],
        'Cluster 0': ['02', '03', '11'],
        'Cluster 3': ['12', '13', '20', '21', '30'],
        'Cluster 1': ['22', '23', '31', '32', '33']
        }

kclus5={'Cluster 2': ['00', '01', '10'],
        'Cluster 0': ['02', '03', '11'],
        'Cluster 3': ['12', '13', '20', '21', '30'],
        'Cluster 1': ['22', '32', '33'],
        'Cluster 4': ['23', '31']
        }

kclus6={'Cluster 2': ['00', '01', '10'],
        'Cluster 0': ['02', '11'],
        'Cluster 5': ['03'],
        'Cluster 3': ['12', '13', '20', '21', '30'],
        'Cluster 1': ['22', '32', '33'],
        'Cluster 4': ['23', '31']
        }

kclus7={'Cluster 2': ['00', '01', '10'],
        'Cluster 0': ['02', '11'],
        'Cluster 5': ['03'],
        'Cluster 3': ['12', '13', '20', '21', '30'],
        'Cluster 6': ['22', '23'],
        'Cluster 4': ['31'],
        'Cluster 1': ['32', '33']
        }

kclus8={'Cluster 2': ['00', '01', '10'],
        'Cluster 0': ['02', '11'],
        'Cluster 5': ['03'],
        'Cluster 4': ['12', '13', '21'],
        'Cluster 6': ['20'],
        'Cluster 3': ['22', '23', '31'],
        'Cluster 7': ['30'],
        'Cluster 1': ['32', '33']
        }

kclus9={'Cluster 8': ['00'],
        'Cluster 2': ['01', '10'],
        'Cluster 0': ['02', '11'],
        'Cluster 5': ['03'],
        'Cluster 4': ['12', '13', '21'],
        'Cluster 6': ['20'],
        'Cluster 3': ['22', '23', '31'],
        'Cluster 7': ['30'],
        'Cluster 1': ['32', '33'],
        }

kclus10={'Cluster 8': ['00'],
         'Cluster 2': ['01', '10'],
         'Cluster 0': ['02', '11'],
         'Cluster 5': ['03'],
         'Cluster 4': ['12', '21'],
         'Cluster 9': ['13'],
         'Cluster 6': ['20'],
         'Cluster 3': ['22', '23', '31'],
         'Cluster 7': ['30'],
         'Cluster 1': ['32', '33'],
         }

kclus11={'Cluster 8': ['00'],
         'Cluster 2': ['01', '10'],
         'Cluster 0': ['02', '11'],
         'Cluster 5': ['03'],
         'Cluster 4': ['12', '21'],
         'Cluster 9': ['13'],
         'Cluster 6': ['20'],
         'Cluster 3': ['22', '23'],
         'Cluster 7': ['30'],
         'Cluster 10': ['31'],
         'Cluster 1': ['32', '33'],
         }


twodclus1={'Cluster 1': ['00','01','10'],
         'Cluster 2': ['33','32','22','23','30','31','12','13','20','03','21','11','02']
    }

twodclus2={'Cluster 1': ['00','01','10'],
         'Cluster 2': ['33','32','22'],
         'Cluster 3':['23','30','31','12','13','20','03','21','11','02']      
    }

twodclus2_2={'Cluster 1': ['00','01','10'],
         'Cluster 2': ['33','32','22','23'],
         'Cluster 3':['30','31','12','13','20','03','21','11','02']      
    }


twodclus3={'Cluster 1': ['00','01','10'],
         'Cluster 2': ['33','32','22'],
         'Cluster 3':['23'],
         'Cluster 4':['30','31','12','13','20','03','21','11','02']
    }

twodclus3_2={'Cluster 1': ['00','01','10'],
         'Cluster 2': ['33','32','22'],
         'Cluster 3':['23','30'],
         'Cluster 4':['31','12','13','20','03','21','11','02']
    }


twodclus4={'Cluster 1': ['00','01','10'],
         'Cluster 2': ['33','32','22'],
         'Cluster 3':['23'],
         'Cluster 4':['30'],
         'Cluster 5':['31','12','13','20','03','21','11','02']

    }

twodclus5={'Cluster 1': ['00','01','10'],
         'Cluster 2': ['33','32','22'],
         'Cluster 3':['23'],
         'Cluster 4':['30'],
         'Cluster 5':['31'],
         'Cluster 6':['12','13','20','03','21','11','02']
    }

TA_clus4={'Cluster 1': ['01','02','03','10','11','20','21','30'],
         'Cluster 2': ['12','13','23','31'],
         'Cluster 3':['00'],
         'Cluster 4':['22','32','33'],
    }

TA_clus4_2={'Cluster 1': ['12','13','23','30','31'],
         'Cluster 2': ['03','11','20','21','02'],
         'Cluster 3':['00'],
         'Cluster 4':['22','32','33','01','10'],
    }


TA_clus4_spec={'Cluster 1': ['02','03','11','12','13','20','21','30'],
         'Cluster 2': ['32','33'],
         'Cluster 3':['22','23','31'],
         'Cluster 4':['00','01','10'],
    }

TA_KNN_clus4 = {
    'Cluster 1': ['01'],
    'Cluster 2': ['00'],
    'Cluster 3': ['10', '30', '32', '33'],
    'Cluster 4': ['02', '03', '11', '12', '13', '20', '21', '22', '23', '31']
}


    

def build_cluster(cluster_mapping_x_data):
    print("CLUSTER DATA")
    print(cluster_mapping_x_data)
    actual_cluster = {}
    for cluster_name,labels in cluster_mapping_x_data.items():
        for each_label in labels:
            actual_cluster[each_label]=cluster_name

    return actual_cluster
        
    


    


    

def fetch_the_data_1d(selected_file_list,cluster_mapping=''):
    print("Current Cluster(in ftd1d):")
    print(cluster_mapping)
    for each_file in selected_file_list:
        
        dataframe=pd.read_csv(each_file)
        dataframe=dataframe[['frame','predicted_digit','last_4_digits']]
        label=each_file[20:22]
        for each_point in dataframe['last_4_digits'].tolist():
            temp_list=[each_point]
            Train_data.append(temp_list)
            if cluster_mapping:
                actual_cluster=build_cluster(cluster_mapping)
                Labels.append(actual_cluster[str(label)])
            else:
                Labels.append(label)
       
            
        
    print("Returned from here")
    return (Train_data,Labels)
       
        

def fetch_the_data_1d_80_20(selected_file_list,cluster_mapping=''):
    print("LOTD")
    print(len(Train_data))
    print("Current Cluster:")
    print(cluster_mapping)
    for each_file in selected_file_list:
        print("LOTD")
        print(len(Train_data))
        dataframe=pd.read_csv(each_file)
        dataframe=dataframe[['frame','predicted_digit','last_4_digits']]
        label=each_file[20:22]

        
        
        for each_point in dataframe['last_4_digits'].tolist()[:int(Threshold*len(dataframe))]:
            temp_list=[each_point]
            Train_data.append(temp_list)
            
            if cluster_mapping:
                actual_cluster=build_cluster(cluster_mapping)
                Train_Labels.append(actual_cluster[str(label)])
                
            else:
                Train_Labels.append(label)
                

        for each_point in dataframe['last_4_digits'].tolist()[int(Threshold*len(dataframe)):]:
            temp_list=[each_point]
            Test_Data.append(temp_list)
            if cluster_mapping:
                actual_cluster=build_cluster(cluster_mapping)
                Test_Labels.append(actual_cluster[str(label)])
            else:
                Test_Labels.append(label)

        
            
        
    print("Returned from here")
    
    return (Train_data,Test_Data,Train_Labels,Test_Labels)






def fetch_the_data_1d_20_80(selected_file_list,cluster_mapping=''):
    print("LOTD")
    print(len(Train_data))
    print("Current Cluster:")
    print(cluster_mapping)
    for each_file in selected_file_list:
        print("LOTD")
        print(len(Train_data))
        dataframe=pd.read_csv(each_file)
        dataframe=dataframe[['frame','predicted_digit','last_4_digits']]
        label=each_file[20:22]

        
        
        for each_point in dataframe['last_4_digits'].tolist()[int((1-Threshold)*len(dataframe)):]:
            temp_list=[each_point]
            Train_data.append(temp_list)
            
            if cluster_mapping:
                actual_cluster=build_cluster(cluster_mapping)
                Train_Labels.append(actual_cluster[str(label)])
                
            else:
                Train_Labels.append(label)
                

        for each_point in dataframe['last_4_digits'].tolist()[:int(1-Threshold*len(dataframe))]:
            temp_list=[each_point]
            Test_Data.append(temp_list)
            if cluster_mapping:
                actual_cluster=build_cluster(cluster_mapping)
                Test_Labels.append(actual_cluster[str(label)])
            else:
                Test_Labels.append(label)

        
            
        
    print("Returned from here")
    
    return (Train_data,Test_Data,Train_Labels,Test_Labels)



def get_intersection_of_timeframes_in_all_files(file_list_x):

    
    first_file_data=pd.read_csv(file_list_x[0])
    timeframes=first_file_data['frame'].tolist()
    timeframes=[int(x) for x in timeframes]
    print("timeframes:")
    #print(timeframes)

    for each_file in file_list_x[1:2]:
        
        current_file_data=pd.read_csv(each_file)
        current_file_data=current_file_data['frame'].tolist()
        current_file_data=[int(x) for x in current_file_data]
        timeframes=list(set(current_file_data)&set(timeframes))
        #timeframes=list(set(current_file_data['frame'].tolist())&set(timeframes))

    print("timeframes common in all file, Length is here:"+str(len(timeframes)))
    print(sorted(timeframes))


"Returns a 2d array, each array within contains dataframe and label of class"
def get_all_dataframes(file_list_fx):
    dataframes=[]
    for each_file in file_list_fx:
        df=pd.read_csv(each_file)
        temp_list=[]
        temp_list.append(df)
        temp_list.append(each_file[20:22])
        dataframes.append(temp_list)

    
    return dataframes


def get_2d_data(cluster_mapping=''):
    print("In get 2d data")
    print(str(cluster_mapping))
    labels=[]
    feature_list=[]
    if cluster_mapping:
        actual_cluster=build_cluster(cluster_mapping)
        
    for every_file in file_list_combined:
        
        label=every_file[14:16]
        data=pd.read_csv(every_file)
      
        for every_row in range(len(data)):
            flow1=data.iloc[every_row,1]
            flow2=data.iloc[every_row,2]
           
            temp_list=[flow1,flow2]
            feature_list.append(temp_list)



            if cluster_mapping:
                
                labels.append(actual_cluster[str(label)])
            else:
                labels.append(label)



            
            
            #labels.append(label)
        

        
    
    return [feature_list,labels]
        

''' Training data is three dim with each point containng:[Frame, Flow1, Flow2]'''
def get_3d_data(cluster_mapping=''):

    labels=[]
    feature_list=[]
    if cluster_mapping:
        actual_cluster=build_cluster(cluster_mapping)
        
    for every_file in file_list_combined:
        
        label=every_file[14:16]
        data=pd.read_csv(every_file)
      
        for every_row in range(len(data)):
            frame_num=data.iloc[every_row,0]
            flow1=data.iloc[every_row,1]
            flow2=data.iloc[every_row,2]
           
            temp_list=[frame_num,flow1,flow2]
            feature_list.append(temp_list)



            if cluster_mapping:
                
                labels.append(actual_cluster[str(label)])
            else:
                labels.append(label)


    return [feature_list,labels]
    
    
    
    












        
    
    

    
    
        

    

