#!/bin/bash
for i in {1..100} # On fait 100 cycles d'entraînement !
do
    echo "======================================"
    echo "      CYCLE RL NUMERO $i"
    echo "======================================"
    
    cd build
    ./selfplay          # 1. Joue 50 parties
    cd ..
    python3 train_rl.py  # 2. Apprend et met à jour le modèle
done
