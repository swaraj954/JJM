import matplotlib.pyplot as plt
import fetch_data
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches
import simple_svm_code
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
import clean_data
import simple_knn_code
import colorcet as cc
from collections import Counter


def get_kde_plots_of_flow_values():
    data,labels=fetch_data.fetch_the_data_1d(fetch_data.file_list_f1)

    flat_data=[a for x in data for a in x]


    df=pd.DataFrame({'values':flat_data,'label':labels})

    print(df)

    unique_labels=set(labels)

    for each_label in unique_labels:
        subset=df[df['label'] == each_label]
        plt.figure()
        sns.kdeplot(data=subset,x='values',fill=True)
        plt.xlim(24, 27)
        plt.title("Distribution for:"+str(each_label))
        plt.xlabel("val")
        plt.ylabel("Density")
        plt.show()


def get_time_flow_charts():
    all_f2_dataframes=fetch_data.get_all_dataframes(fetch_data.file_list_f2)
    for every_dataframe in all_f2_dataframes:
        data=every_dataframe[0]
        timestamps=data['frame'].tolist()
        flows=data['last_4_digits'].tolist()

        plt.figure()
        plt.plot(timestamps,flows)
        plt.xlabel('Time')
        plt.ylabel("Flow")
        plt.title("Flow vs time for:"+str(every_dataframe[1]))
        plt.grid(True)
        plt.ylim(22.5,24)
        plt.show()


def visualize_2d_data():
    #Points_to_Plots, Labels = fetch_data.get_2d_data()
    Points_to_Plots, Labels = clean_data.get_cleaned_2d_data()
    le=LabelEncoder()
    encoded_labels=le.fit_transform(Labels)


    
    X_cord=[]
    Y_cord=[]
    for every_point in Points_to_Plots:
        X_cord.append(every_point[0])
        Y_cord.append(every_point[1])

    
    s=plt.scatter(X_cord,Y_cord,c=encoded_labels,cmap="tab20",s=50,edgecolors='k')
   

    legend_labels = le.classes_  # ['class1', 'class2', ..., 'class16']
    colors = [s.cmap(s.norm(i)) for i in range(len(legend_labels))]
    patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
    plt.legend(handles=patches, title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')


    #plt.xlim(20, 55)   
    #plt.ylim(22, 24) 

    

    
    
    
    plt.grid(True)
    
    plt.show()


def visualize_svm_boundries():
    X,y=fetch_data.get_2d_data(fetch_data.twodclus3)
    #X,y=clean_data.get_cleaned_2d_data(fetch_data.twodclus3)
    
    X=np.array(X)
    y=np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    step_size=0.01
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,step_size),
                      np.arange(y_min,y_max,step_size))
    
    #Z=simple_svm_code.get_model().predict(np.c_[xx.ravel(),yy.ravel()])

    mesh_grid_coordinates=np.c_[xx.ravel(),yy.ravel()]
    Z=[]
    print("Training Model")
    svm_model=simple_svm_code.get_model(fetch_data.twodclus3)
    print("Generating Plot")
    for i in tqdm(range(len(mesh_grid_coordinates))):
        to_2d=[]
        to_2d.append(mesh_grid_coordinates[i])
        Z.append(svm_model.predict(to_2d)[0])


    print("Predicted clsses")
    print(set(Z))
    Z=le.transform(Z)
   
    Z=Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap='viridis', s=80, edgecolors='k')
    legend_label=le.classes_
    colors=[scatter.cmap(scatter.norm(i)) for i in range(len(legend_label))]
    patches=[mpatches.Patch(color=colors[i],label=legend_label[i]) for i in range(len(legend_label))]
    plt.legend(handles=patches,title="Class",loc="upper right")


    plt.title("Decision Boundary for original Data")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        

'''Returns a 3d view of density of values for (f1,f2)'''
def plot_3d_kde_surface():
    Coordinates=np.array(fetch_data.get_2d_data()[0])
    
    x_Coordinates=Coordinates[:,0]
    y_Coordinates=Coordinates[:,1]

    xy=np.vstack([x_Coordinates,y_Coordinates])

    kde=gaussian_kde(xy)

    x_grid=np.linspace(x_Coordinates.min()-1,x_Coordinates.max()+10,1000)
    y_grid=np.linspace(y_Coordinates.min()-1,y_Coordinates.max()+10,1000)
    X,Y=np.meshgrid(x_grid,y_grid)

    Z=kde(np.vstack([X.ravel(),Y.ravel()])).reshape(X.shape)

    fig = plt.figure(figsize=(10,7))
    ax=fig.add_subplot(111,projection='3d')

    surf = ax.plot_surface(X,Y,Z,cmap='plasma',edgecolor='none',alpha=0.9)
    plt.tight_layout()
    ax.set_xlabel('X',labelpad=10)
    ax.set_ylabel('Y',labelpad=10)
    ax.set_zlabel('Density',labelpad=10)
    ax.set_title('3D KDE SURFACE PLOT')
    #ax.view_init(elev=30,azim=135)
    ax.grid(False)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) 
    plt.show()


def create_grid(Train_data):
    Train_data = np.array(Train_data)
    x_min_coordinate = Train_data[:, 0].min() - 1
    x_max_coordinate = Train_data[:, 0].max() + 1
    y_min_coordinate = Train_data[:, 1].min() - 1
    y_max_coordinate = Train_data[:, 1].max() + 1
    x_cord_matrix, y_cord_matrix = np.meshgrid(
        np.linspace(x_min_coordinate, x_max_coordinate, 1000),
        np.linspace(y_min_coordinate, y_max_coordinate, 1000)
    )
    print("matrix x,y")
    print(x_cord_matrix[0:3], y_cord_matrix[0:3])
    return (x_cord_matrix, y_cord_matrix)


def visualizeKNNBoundaries():
    #Flow_readings, Labels = fetch_data.get_2d_data()
    Flow_readings,Labels=clean_data.get_cleaned_2d_data()
    

##    Flow_readings=Flow_readings[(1192-300):(1492)]
##    Labels=Labels[(1192-300):(1492)]
    
    print("Frequencies of labels in data:")
    freqq=Counter(Labels)
    print(freqq)
    

    knn_model = simple_knn_code.get_best_model()
    Flow_readings = np.array(Flow_readings)
    x_cord_matrix, y_cord_matrix = create_grid(Flow_readings)

    print("Generating Visual.........")
    predictions = knn_model.predict(np.c_[x_cord_matrix.ravel(), y_cord_matrix.ravel()])



    
   
    freq=Counter(predictions)
    print(freq)
    unique_labels = sorted(np.unique(Labels))
    print("------------------")
    print(enumerate(unique_labels))
    print("---------------")
    print(enumerate(unique_labels))
    print("---------------")
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for idx, label in enumerate(unique_labels)}

    numeric_labels = np.array([label_to_int[lbl] for lbl in Labels])
    numeric_predictions = np.array([label_to_int[pred] for pred in predictions])
    numeric_predictions = numeric_predictions.reshape(x_cord_matrix.shape)

    num_classes = len(unique_labels)
    cmap = cc.cm.glasbey
    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)
    print(f"Norm:{norm}")
   
    plt.figure(figsize=(10, 8))

    plt.contourf(x_cord_matrix, y_cord_matrix, numeric_predictions, alpha=1, cmap=cmap, norm=norm)

    scatter = plt.scatter(
        Flow_readings[:, 0],
        Flow_readings[:, 1],
        c=numeric_labels,
        cmap=cmap,
        norm=norm,
        edgecolor='k',
        s=40
    )

    plt.title("KNN Decision Boundary on dataset")
    plt.xlabel("Flowmeter 1")
    plt.ylabel("Flowmeter 2")

   
    legend_patches = [
        mpatches.Patch(color=cmap(norm(i)), label=int_to_label[i])
        for i in range(num_classes)
    ]
    plt.legend(handles=legend_patches, title="Classes", loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.grid(True)
    plt.tight_layout()
    plt.show()



visualizeKNNBoundaries()




#visualize_svm_boundries()
#plot_3d_kde_surface()
#visualize_2d_data()







