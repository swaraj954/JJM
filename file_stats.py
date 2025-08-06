import pandas as pd
import statistics

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

X_all = []
y_all = []


for file in file_list:
    df=pd.read_csv(file)
    print("File Name:"+str(file))
    X_all.extend(df["last_4_digits"].tolist())
    print("Max value: "+str(max(X_all)))
    print("Min value: "+str(min(X_all)))
    print("mean: "+str(statistics.mean(X_all)))
    print("Varience: "+str(statistics.variance(X_all)))
    print("-----------------------------------------------------")
  
    
