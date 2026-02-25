import struct

import torch

from train import ChessNNUE  # Assure-toi que ton fichier précédent s'appelle train.py


def export_to_bin():
    print("Chargement du modèle...")
    model = ChessNNUE()
    model.load_state_dict(torch.load("chess_weights.pth"))
    model.eval()  # Mode évaluation

    # Fichier binaire brut de sortie
    filename = "nnue_weights.bin"

    with open(filename, "wb") as f:
        # On parcourt toutes les couches du réseau
        for name, param in model.named_parameters():
            # On convertit le tenseur PyTorch en un tableau numpy 1D (plat)
            flat_weights = param.detach().numpy().flatten()

            # On écrit chaque nombre flottant (float32) en binaire ('f' en struct)
            for weight in flat_weights:
                f.write(struct.pack("f", weight))

            print(f"Exporté {name}: {len(flat_weights)} valeurs.")

    print(f"\nSuccès ! Poids exportés dans '{filename}'.")
    print("Le code C pourra maintenant lire ce fichier directement !")


if __name__ == "__main__":
    export_to_bin()
