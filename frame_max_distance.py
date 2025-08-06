import fetch_data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

max_distance=0

f1=fetch_data.get_all_dataframes(fetch_data.file_list_f1)
f2=fetch_data.get_all_dataframes(fetch_data.file_list_f2)
back=0
frot=0
fname=''
distances=[]
for x in f1:
    list_of_frames=x[0]['frame']
    length=len(list_of_frames)
    for i in range(1,length):
        distances.append(list_of_frames[i]-list_of_frames[i-1])
        
        if max_distance<(list_of_frames[i]-list_of_frames[i-1]):
            back=list_of_frames[i-1]
            frot=list_of_frames[i]
            max_distance=list_of_frames[i]-list_of_frames[i-1]
            fname=str(x[1])


print("maximum_distance"+str(max_distance))
print("Prev value"+str(back))
print("Front name"+str(frot))
print("Name of file:"+str(fname))
print("DISTANCES:")


df=pd.DataFrame({'value':distances,'dummy':['All']*len(distances)})
sns.stripplot(x='dummy',y='value',data=df,jitter=True)
plt.show()
      
    
      
        
