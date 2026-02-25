import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from rl_model import ChessActorCritic  # On importe l'architecture de ton modèle !


# ==========================================
# 1. LECTURE DU DATASET DE SELF-PLAY
# ==========================================
class RLDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Chargement de l'experience RL depuis {csv_file}...")
        data = pd.read_csv(csv_file, header=None)

        # Les 768 premières colonnes : L'échiquier
        self.states = torch.tensor(data.iloc[:, :768].values, dtype=torch.float32)

        # La colonne 768 : L'index du coup joué (de 0 à 4095)
        self.actions = torch.tensor(data.iloc[:, 768].values, dtype=torch.long)

        # La colonne 769 : Le résultat de la partie (+1, -1, ou 0)
        self.rewards = torch.tensor(data.iloc[:, 769].values, dtype=torch.float32).view(
            -1, 1
        )

        print(f"Experience chargee : {len(self.states)} positions jouees.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx]


# ==========================================
# 2. L'ENTRAÎNEMENT (POLICY GRADIENT)
# ==========================================
def train_rl_agent():
    # Paramètres
    batch_size = 256
    epochs = 5  # En RL, on ne fait pas trop d'epochs sur les mêmes données pour ne pas "sur-apprendre"
    learning_rate = 0.0005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # On charge le modèle (avec tes poids actuels)
    model = ChessActorCritic().to(device)
    (
        model.load_state_dict(torch.load("actor_critic_weights_temp.pth"))
        if False
        else None
    )  # (Placeholder)

    # On doit charger les poids de l'ancien modèle pour continuer l'entraînement !
    # La façon la plus simple est de charger directement l'état PyTorch.
    # Dans la pratique, on va charger depuis les poids de rl_model.py

    dataset_path = (
        "build/rl_dataset.csv"  # Le chemin vers le fichier généré par ton C++
    )
    dataloader = DataLoader(
        RLDataset(dataset_path), batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Fonctions d'erreur
    mse_loss = nn.MSELoss()  # Pour le Critique
    log_softmax = nn.LogSoftmax(
        dim=1
    )  # Pour transformer les sorties de l'Acteur en probabilités log

    print("\n--- Debut de la mutation (Policy Gradient) ---")
    model.train()

    history_value_loss = []
    history_policy_loss = []

    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0

        for states, actions, rewards in dataloader:
            states, actions, rewards = (
                states.to(device),
                actions.to(device),
                rewards.to(device),
            )

            optimizer.zero_grad()

            # Le réseau évalue les plateaux
            policy_logits, values = model(states)

            # 1. PERTE DU CRITIQUE (Value Loss)
            # Le réseau a-t-il bien deviné qui allait gagner ?
            value_loss = mse_loss(values, rewards)

            # 2. PERTE DE L'ACTEUR (Policy Loss)
            # On prend les probabilités logaritmiques de tous les coups
            log_probs = log_softmax(policy_logits)

            # On isole la probabilité du coup qui a *vraiment* été joué
            action_log_probs = log_probs.gather(1, actions.view(-1, 1))

            # L'astuce mathématique du RL : on multiplie par la récompense !
            # Si reward = +1, on veut maximiser cette probabilité (donc minimiser le log négatif)
            # Si reward = -1, la probabilité va chuter.
            policy_loss = -(action_log_probs * rewards).mean()

            # On additionne les deux erreurs et on corrige les poids
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} | Critique Loss: {total_value_loss/len(dataloader):.4f} | Acteur Loss: {total_policy_loss/len(dataloader):.4f}"
        )

        avg_v_loss = total_value_loss / len(dataloader)
        avg_p_loss = total_policy_loss / len(dataloader)

        history_value_loss.append(avg_v_loss)
        history_policy_loss.append(avg_p_loss)
        print(
            f"Epoch {epoch+1}/{epochs} | Critique: {avg_v_loss:.4f} | Acteur: {avg_p_loss:.4f}"
        )

    # --- NOUVEAU : SAUVEGARDE DE LA COURBE ---
    plt.figure(figsize=(10, 5))
    plt.plot(history_value_loss, label="Critique (Deviner le gagnant)", color="blue")
    plt.plot(history_policy_loss, label="Acteur (Choisir le bon coup)", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Erreur)")
    plt.title("Evolution de l'Apprentissage RL")
    plt.legend()
    plt.grid(True)
    plt.savefig("build/loss_curve.png")  # Sauvegarde l'image !
    plt.close()
    # ----------------------------------------

    # ==========================================
    # 3. EXPORTATION POUR LE PROCHAIN CYCLE C++
    # ==========================================
    print("\n--- Exportation du modele mis a jour ---")
    model.eval()
    example_input = torch.zeros(1, 768).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(
        "build/actor_critic_model.pt"
    )  # On ecrase l'ancien modele directement dans build !

    print("Mise a jour terminee. Le nouveau modele est pret dans 'build/' !")
    print("Tu peux relancer ./selfplay pour generer de meilleures parties !")


if __name__ == "__main__":
    train_rl_agent()
