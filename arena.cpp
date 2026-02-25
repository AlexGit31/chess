#include <torch/script.h>
#include <iostream>
#include <vector>

extern "C" {
    #include "engine.c" // On importe tes bitboards et ton générateur de coups
}

// ==========================================
// 1. OUTILS POUR LE RÉSEAU DE NEURONES (L'Acteur)
// ==========================================
torch::Tensor get_board_tensor(Board* b) {
    torch::Tensor tensor = torch::zeros({1, 768});
    float* data = tensor.data_ptr<float>();
    uint64_t* bitboards[12] = {
        &b->white_pawns, &b->white_knights, &b->white_bishops, &b->white_rooks, &b->white_queens, &b->white_king,
        &b->black_pawns, &b->black_knights, &b->black_bishops, &b->black_rooks, &b->black_queens, &b->black_king
    };
    for (int pt = 0; pt < 12; pt++) {
        uint64_t bb = *(bitboards[pt]);
        while (bb) {
            data[(pt * 64) + __builtin_ctzll(bb)] = 1.0f;
            bb &= bb - 1; 
        }
    }
    return tensor;
}

int get_move_index(uint32_t move) {
    return (GET_FROM(move) * 64) + GET_TO(move);
}

// L'IA joue son MEILLEUR coup (Déterministe, pas de hasard)
uint32_t get_rl_move(torch::jit::script::Module& model, Board* b, int color) {
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    
    std::vector<uint32_t> legal_moves;
    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (!is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) {
            legal_moves.push_back(list.moves[i]);
        }
        *b = backup;
    }

    if (legal_moves.empty()) return 0; // Mat ou Pat

    torch::Tensor state_tensor = get_board_tensor(b);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(state_tensor);

    auto outputs = model.forward(inputs).toTuple();
    torch::Tensor policy_tensor = outputs->elements()[0].toTensor();
    float* policy_data = policy_tensor.data_ptr<float>();

    uint32_t best_move = legal_moves[0];
    float max_logit = -1e9;

    // On cherche le coup légal avec la plus haute confiance
    for (uint32_t move : legal_moves) {
        float logit = policy_data[get_move_index(move)];
        if (logit > max_logit) {
            max_logit = logit;
            best_move = move;
        }
    }
    return best_move;
}

// ==========================================
// 2. L'ADVERSAIRE CLASSIQUE (Matériel pur)
// ==========================================
int evaluate_material_cpp(Board* b) {
    int score = 0;
    score += __builtin_popcountll(b->white_pawns) * 100;
    score += __builtin_popcountll(b->white_knights) * 300;
    score += __builtin_popcountll(b->white_bishops) * 300;
    score += __builtin_popcountll(b->white_rooks) * 500;
    score += __builtin_popcountll(b->white_queens) * 900;

    score -= __builtin_popcountll(b->black_pawns) * 100;
    score -= __builtin_popcountll(b->black_knights) * 300;
    score -= __builtin_popcountll(b->black_bishops) * 300;
    score -= __builtin_popcountll(b->black_rooks) * 500;
    score -= __builtin_popcountll(b->black_queens) * 900;
    return score;
}

int alpha_beta_classic(Board* b, int depth, int alpha, int beta, int color) {
    if (depth == 0) {
        int eval = evaluate_material_cpp(b);
        return (color == WHITE) ? eval : -eval;
    }
    
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    int best_score = -50000;
    int legal_moves = 0;

    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) {
            *b = backup; continue;
        }
        legal_moves++;
        
        int score = -alpha_beta_classic(b, depth - 1, -beta, -alpha, 1 - color);
        *b = backup;
        
        if (score > best_score) best_score = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    
    if (legal_moves == 0) return -50000 + (4 - depth);
    return best_score;
}

uint32_t get_classical_move(Board* b, int depth, int color) {
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    int best_score = -50000;
    uint32_t best_move = 0;

    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) {
            *b = backup; continue;
        }
        
        int score = -alpha_beta_classic(b, depth - 1, -50000, 50000, 1 - color);
        *b = backup;
        
        if (score > best_score) {
            best_score = score;
            best_move = list.moves[i];
        }
    }
    return best_move;
}

// Demande au Critique PyTorch d'évaluer le plateau
int evaluate_torch_value(torch::jit::script::Module& model, Board* b) {
    torch::Tensor state_tensor = get_board_tensor(b);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(state_tensor);

    // On fait passer le plateau dans le réseau
    auto outputs = model.forward(inputs).toTuple();
    
    // elements()[0] est la Policy. elements()[1] est la Value !
    torch::Tensor value_tensor = outputs->elements()[1].toTensor();
    float value = value_tensor.item<float>(); // Un nombre entre -1.0 et 1.0

    // On le convertit en "Centipions" pour l'Alpha-Beta (ex: +1.0 devient +1000)
    return (int)(value * 1000.0f);
}

int alpha_beta_rl(torch::jit::script::Module& model, Board* b, int depth, int alpha, int beta, int color) {
    if (depth == 0) {
        int eval = evaluate_torch_value(model, b);
        return (color == WHITE) ? eval : -eval;
    }
    
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    int best_score = -50000;
    int legal_moves = 0;

    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) {
            *b = backup; continue;
        }
        legal_moves++;
        
        int score = -alpha_beta_rl(model, b, depth - 1, -beta, -alpha, 1 - color);
        *b = backup;
        
        if (score > best_score) best_score = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    
    if (legal_moves == 0) return -50000 + (4 - depth);
    return best_score;
}

// La fonction pour récupérer le meilleur coup selon l'Alpha-Bêta RL
uint32_t get_rl_alphabeta_move(torch::jit::script::Module& model, Board* b, int depth, int color) {
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    int best_score = -50000;
    uint32_t best_move = 0;

    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) {
            *b = backup; continue;
        }
        
        int score = -alpha_beta_rl(model, b, depth - 1, -50000, 50000, 1 - color);
        *b = backup;
        
        if (score > best_score) {
            best_score = score;
            best_move = list.moves[i];
        }
    }
    return best_move;
}





// ==========================================
// 3. LE TOURNOI (MAIN)
// ==========================================
int main() {
    init_leapers_and_masks();
    
    std::cout << "===================================================\n";
    std::cout << "   ARENE D'EVALUATION : RL (Blancs) vs MATERIEL (Noirs)\n";
    std::cout << "===================================================\n";

    torch::jit::script::Module model;
    try {
        model = torch::jit::load("actor_critic_model.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Erreur de chargement du modele.\n";
        return -1;
    }

    Board board = {
        .white_pawns   = 0x000000000000FF00ULL, .white_rooks   = 0x0000000000000081ULL,
        .white_knights = 0x0000000000000042ULL, .white_bishops = 0x0000000000000024ULL,
        .white_queens  = 0x0000000000000008ULL, .white_king    = 0x0000000000000010ULL,
        .black_pawns   = 0x00FF000000000000ULL, .black_rooks   = 0x8100000000000000ULL,
        .black_knights = 0x4200000000000000ULL, .black_bishops = 0x2400000000000000ULL,
        .black_queens  = 0x0800000000000000ULL, .black_king    = 0x1000000000000000ULL,
        .castling_rights = 15, .en_passant_square = -1, .halfmove_clock = 0
    };

    print_board(&board);

    for (int turn = 0; turn < 150; turn++) {
        int color = turn % 2;
        
        int status = check_game_over(&board, color);
        if (status == 1) {
            std::cout << "\n🏆 ECHEC ET MAT ! " << ((color == WHITE) ? "Les Noirs (Classique)" : "Les Blancs (RL)") << " gagnent !\n";
            break;
        } else if (status == 2) {
            std::cout << "\n🤝 PAT ! Match nul.\n";
            break;
        }

        uint32_t move = 0;
        if (color == WHITE) {
            std::cout << "\n--- Tour " << (turn/2)+1 << " | IA RL (Blancs) reflechit... ---\n";
            move = get_classical_move(&board, 3, color); // Profondeur 3
        } else {
            std::cout << "\n--- Tour " << (turn/2)+1 << " | IA Classique (Noirs) reflechit... ---\n";
            move = get_rl_alphabeta_move(model, &board, 4, color); // Profondeur 3
        }

        if (move == 0) break; // Sécurité

        std::cout << "Coup joue : ";
        print_move(move);
        std::cout << "\n";
        make_move(&board, move, color);
        print_board(&board);
    }

    return 0;
}
