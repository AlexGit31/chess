#include <torch/script.h> // L'API LibTorch
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>

// On inclut directement ton code C ultra-rapide !
extern "C" {
    #include "engine.c"
}

// ==========================================
// 1. TRADUCTION DU PLATEAU POUR PYTORCH
// ==========================================
torch::Tensor get_board_tensor(Board* b) {
    // Crée un tenseur de taille [1, 768] rempli de zéros
    torch::Tensor tensor = torch::zeros({1, 768});
    float* data = tensor.data_ptr<float>(); // Accès direct à la mémoire (HPC)

    uint64_t* bitboards[12] = {
        &b->white_pawns, &b->white_knights, &b->white_bishops, &b->white_rooks, &b->white_queens, &b->white_king,
        &b->black_pawns, &b->black_knights, &b->black_bishops, &b->black_rooks, &b->black_queens, &b->black_king
    };

    for (int piece_type = 0; piece_type < 12; piece_type++) {
        uint64_t bb = *(bitboards[piece_type]);
        while (bb) {
            int square = __builtin_ctzll(bb);
            data[(piece_type * 64) + square] = 1.0f;
            bb &= bb - 1; 
        }
    }
    return tensor;
}

// Calcule l'index unique (0-4095) d'un coup
int get_move_index(uint32_t move) {
    int from = GET_FROM(move);
    int to = GET_TO(move);
    return (from * 64) + to;
}

// ==========================================
// 2. LA BOUCLE DE SELF-PLAY (Deep RL)
// ==========================================
void play_rl_game(torch::jit::script::Module& model, std::ofstream& dataset_file) {
    // Initialisation
    Board board = {
        .white_pawns   = 0x000000000000FF00ULL, .white_rooks   = 0x0000000000000081ULL,
        .white_knights = 0x0000000000000042ULL, .white_bishops = 0x0000000000000024ULL,
        .white_queens  = 0x0000000000000008ULL, .white_king    = 0x0000000000000010ULL,
        .black_pawns   = 0x00FF000000000000ULL, .black_rooks   = 0x8100000000000000ULL,
        .black_knights = 0x4200000000000000ULL, .black_bishops = 0x2400000000000000ULL,
        .black_queens  = 0x0800000000000000ULL, .black_king    = 0x1000000000000000ULL,
        .castling_rights = 15, .en_passant_square = -1, .halfmove_clock = 0
    };

    // Historique de la partie pour la rétropropagation
    std::vector<torch::Tensor> states_history;
    std::vector<int> turn_color_history;
    std::vector<int> action_history; 
    int max_moves = 300;
    int game_result = 0; // 0=Nul, 1=Blancs gagnent, -1=Noirs gagnent

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int turn = 0; turn < max_moves; turn++) {
        int color = turn % 2;

        // --- 1. Vérification de fin de partie ---
        int status = check_game_over(&board, color);
        if (status == 1) { // Mat
            game_result = (color == WHITE) ? -1 : 1; 
            break;
        } else if (status == 2) { // Pat
            game_result = 0;
            break;
        }

        // --- 2. Génération des coups légaux stricts ---
        MoveList list; list.count = 0;
        generate_moves(&board, &list, color);
        std::vector<uint32_t> legal_moves;
        for (int i = 0; i < list.count; i++) {
            Board backup = board;
            make_move(&board, list.moves[i], color);
            uint64_t my_king = (color == WHITE) ? board.white_king : board.black_king;
            if (!is_square_attacked(&board, __builtin_ctzll(my_king), 1 - color)) {
                legal_moves.push_back(list.moves[i]);
            }
            board = backup;
        }
        
        if (legal_moves.empty()) {
            game_result = (color == WHITE) ? -1 : 1; // Mat (sécurité)
            break;
        }

        // --- 3. Interrogation du Cerveau PyTorch ---
        torch::Tensor state_tensor = get_board_tensor(&board);
        states_history.push_back(state_tensor.clone()); // On sauvegarde ce qu'il a vu
        turn_color_history.push_back(color);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(state_tensor);

        // Forward Pass ! (p = Policy, v = Value)
        auto outputs = model.forward(inputs).toTuple();
        torch::Tensor policy_tensor = outputs->elements()[0].toTensor();
        float* policy_data = policy_tensor.data_ptr<float>();

        // --- 4. Le Filtre de Légalité et le Softmax ---
        std::vector<float> legal_logits;
        float max_logit = -1e9; // Pour la stabilité numérique du Softmax

        for (uint32_t move : legal_moves) {
            float logit = policy_data[get_move_index(move)];
            legal_logits.push_back(logit);
            if (logit > max_logit) max_logit = logit;
        }

        // Application du Softmax (Convertit les logits en pourcentages)
        std::vector<float> probabilities;
        float sum_exp = 0.0f;
        for (float logit : legal_logits) {
            float exp_val = std::exp(logit - max_logit);
            probabilities.push_back(exp_val);
            sum_exp += exp_val;
        }

        // --- 5. Le choix d'Acteur (Échantillonnage) ---
        // On tire au sort un coup en respectant rigoureusement les probabilités !
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
        int chosen_index = dist(gen);
        uint32_t chosen_move = legal_moves[chosen_index];
        action_history.push_back(get_move_index(chosen_move));
        // --- 6. On joue le coup ---
        make_move(&board, chosen_move, color);
    }

    // ==========================================
    // 3. SAUVEGARDE DE L'EXPÉRIENCE (Pour Python)
    // ==========================================
    // La partie est finie. On écrit chaque plateau et la vraie valeur finale (+1, -1, 0)
    for (size_t i = 0; i < states_history.size(); i++) {
        float* data = states_history[i].data_ptr<float>();
        for (int j = 0; j < 768; j++) {
            dataset_file << (int)data[j] << ",";
        }
        dataset_file << action_history[i] << ",";
        // Ajustement du point de vue : Si les Blancs ont gagné, c'est +1 pour les Blancs, mais -1 pour les Noirs.
        float final_value = (float)game_result;
        if (turn_color_history[i] == BLACK) final_value = -final_value; 

        dataset_file << final_value << "\n";
    }
}

int main() {
    init_leapers_and_masks();
    
    std::cout << "===================================================\n";
    std::cout << "   USINE A PARTIES - DEEP REINFORCEMENT LEARNING   \n";
    std::cout << "===================================================\n";

    // 1. Chargement du modèle PyTorch compilé
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("actor_critic_model.pt");
        std::cout << "Modele Actor-Critic charge avec succes !\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "Erreur : Impossible de charger actor_critic_model.pt\n";
        return -1;
    }

    std::ofstream dataset_file("rl_dataset.csv");

    // 2. Lancement des parties
    int num_games = 50; // Nombre de parties pour ce "Batch" d'expérience
    for (int i = 1; i <= num_games; i++) {
        std::cout << "Generation de la partie " << i << " / " << num_games << "...\n";
        play_rl_game(module, dataset_file);
    }

    dataset_file.close();
    std::cout << "Experience RL sauvegardee dans 'rl_dataset.csv'. Prets pour Python !\n";
    
    return 0;
}
