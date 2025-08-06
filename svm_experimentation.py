import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import glob

# Your list of filenames (you can also use glob to auto-grab)
file_list = [
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

X_all = []
y_all = []

for file in file_list:
    # Extract label XY from filename using regex
    match = re.search(r"predicted_digits_(\w{2})_", file)
    if match:
        label = str(match.group(1))
    else:
        continue  # Skip files that don't match

    df = pd.read_csv(file)
    print(file+":"+str(len(df)))
    
    selected_columns = df.iloc[:, [0, 1, 3]]
    # Get the 'last_4_digits' column and append
    #X_all.extend(df["last_4_digits"].tolist())
    X_all.extend(selected_columns.values.tolist())
    print(X_all[0:10])
    y_all.extend([label] * len(df))  # Assign label XY to all rows

# Convert to DataFrame for SVM
X_df = pd.DataFrame(X_all)
y_df = pd.Series(y_all)

print(y_df)

combined_df = X_df.copy()
combined_df["label"] = y_df

# Save to CSV
combined_df.to_csv("svm_data.csv", index=False)


X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)



# Train SVM
clf = SVC(kernel='rbf',C=1000,gamma=100, decision_function_shape='ovr')
clf.fit(X_train, y_train)

all_classes = [
    "00", "01", "02", "03",
    "10", "11", "12", "13",
    "20", "21", "22", "23",
    "30", "31", "32", "33"
]


# Predict & evaluate
y_pred = clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred,labels=all_classes))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
