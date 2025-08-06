import fetch_data
import numpy as np
import pandas as pd

f1_data=fetch_data.get_all_dataframes(fetch_data.file_list_f1)
f2_data=fetch_data.get_all_dataframes(fetch_data.file_list_f2)

csv_file={}

for i in range(16):
    label=f1_data[i][1]

    frames=[x for x in range(1,301)]

    known_time_frames_f1=(f1_data[i][0])['frame'].tolist()
    known_time_frames_f2=(f2_data[i][0])['frame'].tolist()

    known_flow_values_f1=(f1_data[i][0])['last_4_digits'].tolist()
    known_flow_values_f2=(f2_data[i][0])['last_4_digits'].tolist()

    interpolated_flow_values_f1=np.interp(frames,known_time_frames_f1,known_flow_values_f1)
    interpolated_flow_values_f2=np.interp(frames,known_time_frames_f2,known_flow_values_f2)

    csv_file['frame']=frames
    csv_file['f1 flow']=interpolated_flow_values_f1
    csv_file['f2 flow']=interpolated_flow_values_f2

    df=pd.DataFrame(csv_file)
    filename=str(label)
    df.to_csv("Combined Data/"+filename+".csv",index=False)
    



    
    
    

        

        


