import torch
import torch.nn as nn


# ==========================================
# 1. L'ANCIEN MODÈLE (Pour récupérer tes poids)
# ==========================================
class OldChessNNUE(nn.Module):
    def __init__(self):
        super(OldChessNNUE, self).__init__()
        self.layer1 = nn.Linear(768, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 32)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)


# ==========================================
# 2. LE NOUVEAU MODÈLE (Actor-Critic)
# ==========================================
class ChessActorCritic(nn.Module):
    def __init__(self):
        super(ChessActorCritic, self).__init__()

        # --- TRONC COMMUN (Shared Body) ---
        # Cette couche comprend la position du plateau (Centre, Sécurité, etc.)
        self.shared_layer = nn.Linear(768, 256)
        self.relu_shared = nn.ReLU()

        # --- TÊTE CRITIQUE (Value Head / Ton ancienne IA) ---
        # Prédit le résultat de la partie entre -1 (Défaite) et 1 (Victoire)
        self.value_layer1 = nn.Linear(256, 32)
        self.relu_value = nn.ReLU()
        self.value_layer2 = nn.Linear(32, 1)
        # Nouveauté : On utilise Tanh pour forcer le score entre -1 et 1
        # car en RL, on prédit une probabilité de victoire, pas un score matériel !
        self.value_tanh = nn.Tanh()

        # --- TÊTE ACTEUR (Policy Head / La Nouveauté) ---
        # Prédit la probabilité pour chaque coup possible (64 * 64 = 4096)
        self.policy_layer = nn.Linear(256, 4096)

    def forward(self, x):
        # Le tenseur traverse le tronc commun
        x_shared = self.relu_shared(self.shared_layer(x))

        # Le signal part dans la tête Critique (Valeur)
        v = self.relu_value(self.value_layer1(x_shared))
        v = self.value_tanh(self.value_layer2(v))

        # Le signal part EN MÊME TEMPS dans la tête Acteur (Coups)
        p = self.policy_layer(x_shared)

        return p, v


# ==========================================
# 3. TRANSFERT CHIRURGICAL ET EXPORTATION
# ==========================================
def create_and_export_rl_model():
    print("1. Initialisation du nouveau modele Actor-Critic...")
    rl_model = ChessActorCritic()

    print("2. Chargement de tes anciens poids (Apprentissage Supervise)...")
    old_model = OldChessNNUE()
    # Assure-toi que chess_weights.pth est dans le meme dossier
    old_model.load_state_dict(torch.load("chess_weights.pth"))

    # --- TRANSFERT DES POIDS ---
    print("3. Greffe de l'ancienne intelligence dans le nouveau modele...")
    with torch.no_grad():  # On desactive le calcul du gradient pour copier brutalement
        # Tronc commun
        rl_model.shared_layer.weight.copy_(old_model.layer1.weight)
        rl_model.shared_layer.bias.copy_(old_model.layer1.bias)

        # Tete Critique
        rl_model.value_layer1.weight.copy_(old_model.layer2.weight)
        rl_model.value_layer1.bias.copy_(old_model.layer2.bias)

        rl_model.value_layer2.weight.copy_(old_model.layer3.weight)
        rl_model.value_layer2.bias.copy_(old_model.layer3.bias)

        # NOTE : rl_model.policy_layer (L'Acteur) reste initialise avec des poids
        # aleatoires. Il est aveugle pour l'instant et apprendra via le Self-Play !

    print("   -> Greffe reussie ! Le Critique a conserve ton expertise.")

    # --- EXPORTATION POUR C++ (TorchScript) ---
    print("4. Compilation du modele en TorchScript pour LibTorch (C++)...")

    # TorchScript a besoin d'un "exemple" d'entree pour figer le code en C++
    example_input = torch.zeros(1, 768)

    # On "trace" le modele (on le convertit en un objet C++)
    traced_script_module = torch.jit.trace(rl_model, example_input)

    # Sauvegarde du fichier binaire lisible par C++
    traced_script_module.save("actor_critic_model.pt")
    print("\nSUCCES : Modele exporte sous 'actor_critic_model.pt'.")
    print("Nous sommes prets a basculer en C++ !")


if __name__ == "__main__":
    create_and_export_rl_model()
