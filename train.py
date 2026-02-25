import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ==========================================
# 1. DÉFINITION DU DATASET
# ==========================================
class ChessDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Chargement des données depuis {csv_file}...")
        # On charge le CSV avec Pandas
        data = pd.read_csv(csv_file, header=None)

        # Les 768 premières colonnes sont l'échiquier (X)
        self.X = torch.tensor(data.iloc[:, :768].values, dtype=torch.float32)

        # La dernière colonne est le score de l'Alpha-Beta (Y)
        # ASTUCE HPC : On divise le score par 100 pour que le réseau
        # prédise un "avantage en pions" (ex: 150 -> 1.5).
        # Les réseaux de neurones détestent prédire des nombres géants.
        self.y = torch.tensor(
            data.iloc[:, 768].values / 100.0, dtype=torch.float32
        ).view(-1, 1)
        print(f"Dataset chargé avec {len(self.X)} positions.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 2. L'ARCHITECTURE DU RÉSEAU (MLP Type NNUE)
# ==========================================
class ChessNNUE(nn.Module):
    def __init__(self):
        super(ChessNNUE, self).__init__()

        # Architecture minimaliste et rapide pour le CPU
        self.layer1 = nn.Linear(768, 256)  # Couche d'accumulation
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(256, 32)  # Couche de réflexion
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(32, 1)  # Couche de sortie (Score)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x


# ==========================================
# 3. LA BOUCLE D'ENTRAÎNEMENT
# ==========================================
def train_model():
    # Paramètres
    batch_size = 512  # On évalue 512 plateaux d'un coup sur la carte graphique
    epochs = 40  # Nombre de passages complets sur le dataset
    learning_rate = 0.001

    # Utilisation du GPU si dispo, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    # Préparation des données
    dataset = ChessDataset("dataset.csv")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialisation du modèle
    model = ChessNNUE().to(device)

    # MSELoss (Mean Squared Error) est parfaite pour prédire un score exact (Régression)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n--- Début de l'entraînement ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 1. Forward pass (Le réseau devine le score)
            predictions = model(inputs)

            # 2. Calcul de l'erreur (Devinette vs Vrai score Alpha-Bêta)
            loss = criterion(predictions, targets)

            # 3. Backpropagation (Correction des poids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(
            f"Epoch [{epoch+1}/{epochs}] - Loss (Erreur moyenne au carré) : {avg_loss:.4f}"
        )

    print("--- Entraînement terminé ---")

    # Sauvegarde des poids du modèle pour les lire en C plus tard !
    torch.save(model.state_dict(), "chess_weights.pth")
    print("Poids du réseau sauvegardés dans 'chess_weights.pth'")


if __name__ == "__main__":
    train_model()
