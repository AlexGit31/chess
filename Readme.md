# ♟️ NeuralBit Chess: HPC C-Engine & Deep RL PyTorch

Welcome to **NeuralBit Chess**, an engineering project combining the raw speed of a low-level chess engine (Native C / Bitboards) with the intelligence of a neural network trained via Reinforcement Learning (Deep RL / Actor-Critic architecture inspired by AlphaZero).

This project is divided into two main parts:

1. **The Core Engine (`engine.c`)**: An ultra-fast engine (up to 4 million nodes/sec on a single CPU core) that can be used as a standalone API.
2. **The Artificial Intelligence Loop**: A complete autonomous training pipeline (Self-Play in C++ with LibTorch -> Policy Gradient Optimization in Python).

---

## 🚀 Part 1: The Native C Engine (`engine.c` API)

The `engine.c` file is the beating heart of the project. It uses no external dependencies and relies entirely on bit manipulation (Bitboards) for extreme performance.

### Play Against the Classic AI

An interactive script `play.c` has been created to play against the "material" version of the AI directly in the terminal (navigating moves with arrow keys).

```bash
gcc -O3 -march=native -o play_chess play.c
./play_chess
```

### Use `engine.c` as an API

To use the engine in your own projects, simply include the file (ensure the original `main` function is renamed or commented out):

```C
#include "engine.c"
```

#### Main Structures

- **`Board`**: The state of the chessboard. Contains 12 `uint64_t` (Bitboards) for pieces, castling rights (`castling_rights`), en passant target square (`en_passant_square`), and the 50-move rule clock.
- **`MoveList`**: Contains an array of 32-bit integers representing legal moves.
- **Move Encoding**: A move is a simple `uint32_t`. Use the macros `GET_FROM(move)`, `GET_TO(move)`, `GET_PIECE(move)`, `GET_PROMOTION(move)`, and `GET_FLAGS(move)` to decode it.

#### Essential Functions

- `void init_leapers_and_masks()`: **Mandatory at startup.** Precalculates attack masks (Knights, Kings, Rays).
- `void generate_moves(Board *b, MoveList *list, int color)`: Populates the `MoveList` with all pseudo-legal moves.
- `void make_move(Board *b, uint32_t move, int color)`: Applies a move to the `Board` (updates bitboards incrementally).
- `int check_game_over(Board *b, int color)`: Returns `0` (Continue), `1` (Checkmate), or `2` (Stalemate).
- `uint32_t search_best_move(Board *b, int depth, int color, int eval_type)`: The optimized Alpha-Beta algorithm (with Null Move Pruning and Late Move Reductions) that returns the calculated best move.

---

## 🧠 Part 2: The Deep Reinforcement Learning Pipeline

This project implements an autonomous learning loop. The AI plays against itself in C++ to generate experience, then learns from its victories and defeats in Python using the **Policy Gradient** algorithm.

### Prerequisites

- Python 3.8+ (`pip install torch pandas matplotlib numpy`)
- CMake (3.14+)
- **LibTorch** (The C++ API for PyTorch, download from the official PyTorch website and extract it on your machine).

_(macOS Users: If LibTorch is blocked by Gatekeeper, run `sudo xattr -r -d com.apple.quarantine /path/to/libtorch`)_.

### The Step-by-Step Training Loop

#### Step 1: Initialize the Actor-Critic Architecture

Create the `.pt` model (TorchScript) readable by C++.

```bash
python rl_model.py
```

_Generates `actor_critic_model.pt`._

#### Step 2: Compile the C++ Executables (HPC)

The self-play factory (`selfplay.cpp`) and the evaluation arena (`arena.cpp`) must be compiled by linking LibTorch.

```bash
mkdir build
cd build

# Replace the path with your absolute path to libtorch

cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```

#### Step 3: Self-Play (Experience Generation)

The AI (The Actor) plays dozens of games against itself asynchronously and saves its decisions.

```bash
cd build
cp ../actor_critic_model.pt .
./selfplay
```

_Generates `rl_dataset.csv`._

#### Step 4: Mutation (RL Training)

The AI updates its neural networks: the probability of moves that led to a win increases, while others decrease.

```bash

# From the root directory

python train_rl.py
```

\*Updates`actor_critic_model.pt`in the`build/`folder and generates a`loss_curve.png` plot.\*

**Repeat steps 3 and 4** to continually improve the AI's skill level (Continuous Training).

#### Step 5: The Arena (Strength Test)

Pit your newly trained AI (White) against the classic material Alpha-Beta engine (Black).

```bash
cd build
./arena
```

**Note** that it is also possible to begin by training a first neural network to predict the value of a valuation function of the boards
and then use this neural network and its weights for the Actor-Critic training so it doesn't start from scratch.
For that purpose you have the two code `generate_data.c` and `train.py`.

---

_Project built from scratch in C/C++ and Python._
