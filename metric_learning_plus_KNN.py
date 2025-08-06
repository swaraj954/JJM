import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
import clean_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 1. Dataset class for Triplet Loss
class FlowTripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.labels = np.unique(y)
        self.class_to_indices = {
            label: np.where(y == label)[0] for label in self.labels
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor = self.X[idx]
        anchor_label = self.y[idx]

        # Positive sample
        pos_idx = np.random.choice(self.class_to_indices[anchor_label])
        positive = self.X[pos_idx]

        # Negative sample
        neg_label = np.random.choice(self.labels[self.labels != anchor_label])
        neg_idx = np.random.choice(self.class_to_indices[neg_label])
        negative = self.X[neg_idx]

        return torch.tensor(anchor, dtype=torch.float32), \
               torch.tensor(positive, dtype=torch.float32), \
               torch.tensor(negative, dtype=torch.float32)


# 2. Simple Embedding Model
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)


# 3. Triplet Loss
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)


# 4. Training + KNN classifier on learned embeddings

def train_metric_model():
    flow_readings, labels = clean_data.get_cleaned_2d_data()
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        flow_readings, labels, test_size=0.2, random_state=42
    )

    train_dataset = FlowTripletDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = EmbeddingNet(input_dim=2, embedding_dim=16)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(40):
        model.train()
        total_loss = 0
        for anchor, positive, negative in tqdm(train_loader):
            optimizer.zero_grad()
            anc_out = model(anchor)
            pos_out = model(positive)
            neg_out = model(negative)
            loss = triplet_loss_fn(anc_out, pos_out, neg_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Generate embeddings
    model.eval()
    with torch.no_grad():
        X_train_embed = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
        X_test_embed = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # Train KNN on embeddings
    print("\nTraining KNN on embeddings...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_embed, y_train)
    y_pred = knn.predict(X_test_embed)

    print("\nKNN on Embeddings - Test Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test, label_encoder


# Run training only
model, X_test, y_test, encoder = train_metric_model()
