# import pandas as pd
# import matplotlib.pyplot as plt
# import glob

# # Since Python file is in same folder
# base_path = "./"

# # Find all CSV files
# csv_files = glob.glob(base_path + "*.csv")
# print("Files found:", csv_files)

# # Create a plot
# plt.figure(figsize=(15, 8))

# for file_path in csv_files:
#     print(f"Reading file: {file_path}")
#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()  # In case there are spaces

#     filename = file_path.split("/")[-1]
#     label = filename.replace(".csv", "")

#     if 'last_4_digits' in df.columns:
#         print(f"Plotting {label} with {len(df['last_4_digits'].dropna())} points.")
#         plt.plot(df['last_4_digits'].values, label=label)
#     else:
#         print(f"Warning: {label} has no 'last_4_digits' column!")

# plt.xlabel('Sample Index')
# plt.ylabel('Flowmeter Reading (last_4_digits)')
# plt.title('Flowmeter Readings Comparison (Multiple Cases)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



### Mean plot
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import glob

# # Since Python file is in the same folder
# base_path = "./"

# # Find all CSV files
# csv_files = glob.glob(base_path + "*.csv")
# print("Files found:", csv_files)

# # Seaborn style setup
# sns.set(style="whitegrid")
# plt.figure(figsize=(15, 8))

# # Store the plot handles for legend interaction
# plot_handles = []

# for file_path in csv_files:
#     print(f"Reading file: {file_path}")
#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()  # In case there are spaces

#     filename = file_path.split("/")[-1]
#     label = filename.replace(".csv", "")

#     if 'last_4_digits' in df.columns:
#         # Calculate the mean of the last_4_digits column
#         mean_value = df['last_4_digits'].mean()
#         print(f"Plotting mean value for {label}: {mean_value}")

#         # Plot the mean value as a horizontal line
#         line, = plt.plot([0, len(df)], [mean_value, mean_value], label=label)  # Horizontal line at the mean value
#         plot_handles.append(line)
#     else:
#         print(f"Warning: {label} has no 'last_4_digits' column!")

# # Add labels, title, grid, and legend
# plt.xlabel('Sample Index')
# plt.ylabel('Flowmeter Reading (Mean of last_4_digits)')
# plt.title('Flowmeter Readings Mean Comparison (Multiple Cases)')
# plt.legend(handles=plot_handles, title="Files", loc='upper left')
# plt.grid(True)

# # Make the legend draggable (this allows real-time hiding/showing of plots)
# plt.legend().set_draggable(True)

# plt.tight_layout()
# plt.show()




# import pandas as pd
# import numpy as np
# import glob
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Initialize Seaborn for better visualization
# sns.set(style="whitegrid")

# # Since Python file is in same folder
# base_path = "./"

# # Find all CSV files
# csv_files = glob.glob(base_path + "*.csv")
# print("Files found:", csv_files)

# # Read all the data, label them based on the filename
# time_series_data = []
# labels = []

# for file_path in csv_files:
#     print(f"Reading file: {file_path}")
#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()  # In case there are spaces

#     # Extract label from filename
#     filename = file_path.split("/")[-1]
#     label = filename.split('_')[2]  # Extracting the '00', '01', ... label from filename

#     if 'last_4_digits' in df.columns:
#         # Taking the first 200 points from each file
#         data_points = df['last_4_digits'].head(200).values
#         time_series_data.append(data_points)
#         labels.append(label)
#     else:
#         print(f"Warning: {label} has no 'last_4_digits' column!")

# # Convert the data to a NumPy array and scale it
# time_series_data = np.array(time_series_data)
# scaler = StandardScaler()
# time_series_data_scaled = scaler.fit_transform(time_series_data)

# # Apply KMeans clustering
# kmeans = KMeans(n_clusters=8, random_state=42)  # 16 clusters based on labels (00-33)
# kmeans.fit(time_series_data_scaled)

# # Map KMeans labels back to the file labels
# # Create a dictionary of clusters and corresponding filenames
# cluster_to_labels = {}

# for idx, file_path in enumerate(csv_files):
#     filename = file_path.split("/")[-1]
#     label = filename.split('_')[2]  # Extracting '00', '01', etc.
#     cluster_label = kmeans.labels_[idx]
    
#     if cluster_label not in cluster_to_labels:
#         cluster_to_labels[cluster_label] = []
#     cluster_to_labels[cluster_label].append(label)

# # Print the cluster and corresponding labels
# for cluster, cluster_labels in cluster_to_labels.items():
#     print(f"Cluster {cluster}: {cluster_labels}")

# # Visualizing the clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=time_series_data_scaled[:, 0], y=time_series_data_scaled[:, 1], 
#                 hue=kmeans.labels_, palette="Set1", s=100, edgecolor="k")

# # Add plot labels
# plt.title("KMeans Clustering of Flowmeter Data")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend(title="Cluster Labels", loc='best')
# plt.show()



# import pandas as pd
# import numpy as np
# import glob
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Initialize Seaborn for better visualization
# sns.set(style="whitegrid")

# # Since Python file is in same folder
# base_path = "./"

# # Find all CSV files
# csv_files = glob.glob(base_path + "*.csv")
# print("Files found:", csv_files)

# # Read all the data, label them based on the filename
# time_series_data = []
# labels = []

# # Creating a list to store custom labels (00, 01, 02,...33) and their corresponding indices
# custom_labels = []

# for file_path in csv_files:
#     print(f"Reading file: {file_path}")
#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()  # In case there are spaces

#     # Extract label from filename
#     filename = file_path.split("/")[-1]
#     label = filename.split('_')[2]  # Extracting the '00', '01', ... label from filename

#     # Add the label to custom labels list
#     custom_labels.append(label)

#     if 'last_4_digits' in df.columns:
#         # Taking the first 200 points from each file
#         data_points = df['last_4_digits'].head(200).values
#         time_series_data.append(data_points)
#         labels.append(label)
#     else:
#         print(f"Warning: {label} has no 'last_4_digits' column!")

# # Convert the data to a NumPy array and scale it
# time_series_data = np.array(time_series_data)
# scaler = StandardScaler()
# time_series_data_scaled = scaler.fit_transform(time_series_data)

# # Apply KMeans clustering
# kmeans = KMeans(n_clusters=16, random_state=42)  # 16 clusters based on labels (00-33)
# kmeans.fit(time_series_data_scaled)

# # Mapping KMeans labels to the custom labels
# # Create a dictionary where cluster label maps to custom labels (00, 01, etc.)
# cluster_to_custom_label = {}

# for idx, cluster_label in enumerate(kmeans.labels_):
#     cluster_to_custom_label[cluster_label] = custom_labels[idx]

# # Visualizing the clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=time_series_data_scaled[:, 0], y=time_series_data_scaled[:, 1], 
#                 hue=[cluster_to_custom_label[label] for label in kmeans.labels_], 
#                 palette="Set1", s=100, edgecolor="k")

# # Add plot labels
# plt.title("KMeans Clustering of Flowmeter Data with Custom Labels")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend(title="Custom Labels", loc='best')
# plt.show()

# # Print cluster labels with their corresponding custom labels
# for cluster, label in cluster_to_custom_label.items():
#     print(f"Cluster {cluster}: {label}")




# ## flowmeter one data 4 clusters
# import pandas as pd
# import numpy as np
# import glob
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Initialize Seaborn for better visualization
# sns.set(style="whitegrid")

# # Since Python file is in same folder
# base_path = "./"

# # Find all CSV files
# csv_files = glob.glob(base_path + "*.csv")
# print("Files found:", csv_files)

# # Read all the data, label them based on the filename
# time_series_data = []
# labels = []
# device_info = []

# for file_path in csv_files:
#     print(f"Reading file: {file_path}")
#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()  # In case there are spaces

#     # Extract label and device info from filename
#     filename = file_path.split("/")[-1]
#     label = filename.split('_')[2]  # Extracting '00', '01', ... label from filename
#     device = filename.split('_')[3][1]  # Extracting f1 or f2 from filename

#     if 'last_4_digits' in df.columns:
#         # Taking the first 200 points from each file
#         data_points = df['last_4_digits'].head(200).values
#         time_series_data.append(data_points)
#         labels.append(label)
#         device_info.append(device)
#     else:
#         print(f"Warning: {label} has no 'last_4_digits' column!")

# # Convert the data to a NumPy array and scale it
# time_series_data = np.array(time_series_data)
# scaler = StandardScaler()
# time_series_data_scaled = scaler.fit_transform(time_series_data)

# # Apply KMeans clustering
# kmeans = KMeans(n_clusters=16, random_state=42)  # 16 clusters based on labels (00-33)
# kmeans.fit(time_series_data_scaled)

# # Map KMeans labels back to the file labels and device info
# cluster_to_labels = {}

# for idx, file_path in enumerate(csv_files):
#     filename = file_path.split("/")[-1]
#     label = filename.split('_')[2]  # Extracting '00', '01', etc.
#     device = filename.split('_')[3][1]  # Extracting f1 or f2 from filename
#     cluster_label = kmeans.labels_[idx]
    
#     if cluster_label not in cluster_to_labels:
#         cluster_to_labels[cluster_label] = []
#     cluster_to_labels[cluster_label].append(f"{label}_f{device}")

# # Print the cluster and corresponding labels with device info
# for cluster, cluster_labels in cluster_to_labels.items():
#     print(f"Cluster {cluster}: {cluster_labels}")

# # Visualizing the clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=time_series_data_scaled[:, 0], y=time_series_data_scaled[:, 1], 
#                 hue=kmeans.labels_, palette="Set1", s=100, edgecolor="k")

# # Add plot labels
# plt.title("KMeans Clustering of Flowmeter Data")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend(title="Cluster Labels", loc='best')
# plt.show()




### SVM for clusters made by kmeans 
import re
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define your cluster mapping for Flowmeter 2
# cluster_mapping = {
#  2: ['00_f2'],
#  3: ['01_f2', '10_f2', '21_f2'],
#  0: ['02_f2', '03_f2', '11_f2', '12_f2', '13_f2', '20_f2', '23_f2', '30_f2'],
#  1: ['22_f2', '31_f2', '32_f2', '33_f2']
# }
cluster_mapping = {
 2: ['00_f2'],
 3: ['01_f2'],
 13: ['02_f2'],
 15: ['03_f2'],
 9: ['10_f2'],
 0: ['11_f2'],
 6: ['12_f2'],
 4: ['13_f2'],
 10: ['20_f2'],
 8: ['21_f2'],
 5: ['22_f2'],
 11: ['23_f2'],
 12: ['30_f2'],
 7: ['31_f2'],
 1: ['32_f2'],
 14: ['33_f2']
}

# 2. List all CSV files in the current directory
files = [f for f in os.listdir() if f.endswith('.csv')]

X_blocks = []
y_blocks = []

# 3. Read each file, map to cluster index, grab 200 samples
for filename in files:
    match = re.match(r'predicted_digits_(\d{2}_f2)_digits\.csv', filename)
    if not match:
        continue
    label_str = match.group(1)   # e.g. "02_f2"

    # Find which cluster this label belongs to
    cluster_id = None
    for cid, members in cluster_mapping.items():
        if label_str in members:
            cluster_id = cid
            break
    if cluster_id is None:
        # Skip files not in any cluster
        continue

    # Load and slice
    df = pd.read_csv(filename)
    df = df.iloc[:200]            # first 200 rows
    X_blocks.append(df.values)    # shape (200, n_features)
    y_blocks.append(np.full(df.shape[0], cluster_id, dtype=int))

# 4. Stack into single arrays
X = np.vstack(X_blocks)          # (number_of_files * 200, n_features)
y = np.concatenate(y_blocks)     # (number_of_files * 200,)

# 5. Drop any feature/column that’s entirely NaN
all_nan_cols = np.isnan(X).all(axis=0)
if all_nan_cols.any():
    print("Dropping all-NaN columns:", np.where(all_nan_cols)[0])
    X = X[:, ~all_nan_cols]

# (Optional) confirm no full-NaN columns remain
print("Any NaNs in X before imputation? ", np.isnan(X).any())
print("NaN count per feature:       ", np.isnan(X).sum(axis=0))

# 6. Split into train and test (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Build a pipeline: impute → scale → SVM
pipeline = Pipeline([
    # ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',  StandardScaler()),
    ('svm',     SVC(kernel='rbf', C=1000, gamma=1)
)
])

# 8. Train
pipeline.fit(X_train, y_train)

# 9. Predict & evaluate
y_pred = pipeline.predict(X_test)

# === Evaluation ===

# 1. Compute Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f}\n")

# 2. Compute and Print Classification Report
target_names = [f"cluster_{i}" for i in sorted(cluster_mapping.keys())]
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 4. Plot the Confusion Matrix as Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)

plt.title("Confusion Matrix Heatmap", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.tight_layout()

# 5. Save the heatmap to file
plt.savefig("confusion_matrix_f2_test.png", dpi=300)
plt.show()

print("\nConfusion matrix heatmap saved as 'confusion_matrix_f2.png'.")
