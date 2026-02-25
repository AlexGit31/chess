#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

// On inclut directement ton moteur C natif
#include "engine.c" 

// ==========================================
// OUTILS POUR LE TERMINAL (MAC / LINUX)
// ==========================================

// Fonction pour lire une touche sans avoir à appuyer sur Entrée
int getch(void) {
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt); // Sauvegarde les paramètres actuels
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO); // Désactive le mode canonique et l'écho
    tcsetattr(STDIN_FILENO, TCSANOW, &newt); // Applique les nouveaux paramètres
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt); // Restaure les paramètres normaux
    return ch;
}

// Efface l'écran du terminal
void clear_screen() {
    printf("\033[H\033[J");
}

// ==========================================
// INTERFACE DE SÉLECTION DU COUP (HUMAIN)
// ==========================================

uint32_t choose_human_move(Board *b, int color) {
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    
    // 1. Filtrer pour ne garder que les coups strictement légaux
    uint32_t legal_moves[256];
    int num_legal = 0;
    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (!is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) {
            legal_moves[num_legal++] = list.moves[i];
        }
        *b = backup;
    }
    
    if (num_legal == 0) return 0; // Sécurité (Géré par check_game_over normalement)

    int selected = 0;
    
    // 2. La boucle d'interface interactive
    while (1) {
        clear_screen();
        printf("===================================================\n");
        printf("   A TON TOUR ! (Joueur %s)\n", (color == WHITE) ? "Blanc" : "Noir");
        printf("===================================================\n");
        print_board(b);
        
        printf("Utilise les fleches <- et -> pour naviguer, et Entree pour valider.\n\n");
        
        // Affichage en grille façon "cd + TAB"
        for (int i = 0; i < num_legal; i++) {
            if (i == selected) {
                printf("\033[1;32m [ "); // Vert gras pour la sélection
                print_move(legal_moves[i]);
                printf(" ] \033[0m");
            } else {
                printf("   ");
                print_move(legal_moves[i]);
                printf("   ");
            }
            
            // Retour à la ligne tous les 6 coups pour la lisibilité
            if ((i + 1) % 6 == 0) printf("\n");
        }
        printf("\n");
        
        // 3. Capture du clavier
        int c1 = getch();
        
        if (c1 == '\n' || c1 == '\r') {
            // Touche Entrée
            return legal_moves[selected];
        } else if (c1 == 27) { 
            // Séquence d'échappement (Les flèches envoient 3 caractères : 27, '[', puis A/B/C/D)
            int c2 = getch();
            int c3 = getch();
            if (c2 == '[') {
                if (c3 == 'C' || c3 == 'B') { // Flèche Droite ou Bas
                    selected = (selected + 1) % num_legal;
                } else if (c3 == 'D' || c3 == 'A') { // Flèche Gauche ou Haut
                    selected = (selected - 1 + num_legal) % num_legal;
                }
            }
        }
    }
}

// ==========================================
// LE JEU PRINCIPAL
// ==========================================

int main() {
    init_leapers_and_masks();
    clear_screen();
    
    int human_color, ai_depth;
    
    printf("===================================================\n");
    printf("   AFFRONTE TON MOTEUR (Evaluation Materielle)\n");
    printf("===================================================\n\n");
    
    printf("Choisis ta couleur (0 pour Blancs, 1 pour Noirs) : ");
    if (scanf("%d", &human_color) != 1) human_color = 0;
    
    printf("Choisis la profondeur de l'IA (ex: 4, 5, 6...) : ");
    if (scanf("%d", &ai_depth) != 1) ai_depth = 5;

    // Vider le buffer de l'entrée standard
    while(getchar() != '\n'); 

    int ai_color = 1 - human_color;

    Board board = {
        .white_pawns   = 0x000000000000FF00ULL, .white_rooks   = 0x0000000000000081ULL,
        .white_knights = 0x0000000000000042ULL, .white_bishops = 0x0000000000000024ULL,
        .white_queens  = 0x0000000000000008ULL, .white_king    = 0x0000000000000010ULL,
        .black_pawns   = 0x00FF000000000000ULL, .black_rooks   = 0x8100000000000000ULL,
        .black_knights = 0x4200000000000000ULL, .black_bishops = 0x2400000000000000ULL,
        .black_queens  = 0x0800000000000000ULL, .black_king    = 0x1000000000000000ULL,
        .castling_rights = 15, .en_passant_square = -1, .halfmove_clock = 0
    };

    for (int turn = 0; turn < 500; turn++) {
        int current_color = turn % 2;
        
        int status = check_game_over(&board, current_color);
        if (status == 1) {
            clear_screen();
            print_board(&board);
            printf("\n🏆 ECHEC ET MAT ! %s gagne !\n", (current_color == WHITE) ? "Les Noirs" : "Les Blancs");
            break;
        } else if (status == 2) {
            clear_screen();
            print_board(&board);
            printf("\n🤝 PAT ! Match nul.\n");
            break;
        }

        uint32_t move = 0;
        
        if (current_color == human_color) {
            move = choose_human_move(&board, current_color);
        } else {
            clear_screen();
            printf("===================================================\n");
            printf("   L'IA REFLECHIT... (Profondeur %d)\n", ai_depth);
            printf("===================================================\n");
            print_board(&board);
            
            // On appelle ton Alpha-Beta avec eval_type = 0 (Matériel)
            move = search_best_move(&board, ai_depth, current_color, 0); 
        }

        if (move == 0) {
            printf("Erreur critique : Aucun coup trouve.\n");
            break;
        }

        make_move(&board, move, current_color);
    }

    return 0;
}
