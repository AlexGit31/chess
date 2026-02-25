#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> // Indispensable pour clock()
// ==========================================
// 1. CONSTANTES ET MACROS HPC
// ==========================================

uint64_t total_nodes = 0;

#define WHITE 0
#define BLACK 1

#define PAWN   1
#define KNIGHT 2
#define BISHOP 3
#define ROOK   4
#define QUEEN  5
#define KING   6

#define FLAG_QUIET       0
#define FLAG_DOUBLE_PUSH 1
#define FLAG_CAPTURE     2
#define INF 50000

#define FLAG_CASTLING 3

// Masques pour les droits de Roque (Castling Rights)
#define CASTLE_WK 1 // White King-side (Petit Roque Blanc)
#define CASTLE_WQ 2 // White Queen-side (Grand Roque Blanc)
#define CASTLE_BK 4 // Black King-side (Petit Roque Noir)
#define CASTLE_BQ 8 // Black Queen-side (Grand Roque Noir)


// Masques de colonnes et rangées
const uint64_t not_A_file  = 0xFEFEFEFEFEFEFEFEULL;
const uint64_t not_H_file  = 0x7F7F7F7F7F7F7F7FULL;
const uint64_t not_AB_file = 0xFCFCFCFCFCFCFCFCULL;
const uint64_t not_GH_file = 0x3F3F3F3F3F3F3F3FULL;
const uint64_t rank_4      = 0x00000000FF000000ULL;
const uint64_t rank_5      = 0x000000FF00000000ULL;

const uint64_t rank_1 = 0x00000000000000FFULL;
const uint64_t rank_8 = 0xFF00000000000000ULL;


// Encodage et décodage 32 bits
#define ENCODE_MOVE(from, to, piece, captured, promotion, flags) \
    ( (from) | ((to) << 6) | ((piece) << 12) | ((captured) << 16) | ((promotion) << 20) | ((flags) << 24) )

#define GET_FROM(move)      ( (move) & 0x3F )
#define GET_TO(move)        ( ((move) >> 6) & 0x3F )
#define GET_PIECE(move)     ( ((move) >> 12) & 0xF )
#define GET_FLAGS(move)     ( ((move) >> 24) & 0xFF )
#define GET_CAPTURED(move)  ( ((move) >> 16) & 0xF )
#define GET_PROMOTION(move) ( ((move) >> 20) & 0xF )

// Si une pièce bouge DEPUIS ou VERS l'une de ces cases, on met à jour les droits
const int castling_rights_update[64] = {
     13, 15, 15, 15, 12, 15, 15, 14, // a1 perd WQ (15 & ~2 = 13), e1 perd WK&WQ (12), h1 perd WK (14)
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
      7, 15, 15, 15,  3, 15, 15, 11  // a8 perd BQ (7), e8 perd BK&BQ (3), h8 perd BK (11)
};


// ==========================================
// 2. STRUCTURES DE DONNÉES
// ==========================================

typedef struct {
    // Les 12 bitboards des pièces [cite: 65-82]
    uint64_t white_pawns, white_rooks, white_knights;
    uint64_t white_bishops, white_queens, white_king;
    uint64_t black_pawns, black_rooks, black_knights;
    uint64_t black_bishops, black_queens, black_king;
    
    // -- NOUVEAUX CHAMPS POUR L'ÉTAT DU JEU --
    
    // Droits de Roque (4 bits: 1=Petit Roque Blanc, 2=Grand Roque Blanc, 4=Petit Noir, 8=Grand Noir)
    int castling_rights; 
    
    // Case de prise en passant (0-63). S'il n'y en a pas, on met -1.
    int en_passant_square; 
    
    // Règle des 50 coups (nulle si aucun pion bougé ou capture depuis 50 demi-coups)
    int halfmove_clock; 
} Board;

typedef struct {
    uint32_t moves[256];
    int count;
} MoveList;

// Déclaration anticipée pour le compilateur
int is_square_attacked(Board *b, int square, int attacker_color);
// ==========================================
// 3. TABLEAUX PRÉCALCULÉS & AFFICHAGE
// ==========================================

uint64_t knight_attacks[64];
uint64_t king_attacks[64];
uint64_t rook_masks[64];
uint64_t bishop_masks[64];

char get_piece_on_square(Board *b, int square) {
    uint64_t mask = 1ULL << square;
    if (b->white_pawns & mask) return 'P';   if (b->black_pawns & mask) return 'p';
    if (b->white_rooks & mask) return 'R';   if (b->black_rooks & mask) return 'r';
    if (b->white_knights & mask) return 'N'; if (b->black_knights & mask) return 'n';
    if (b->white_bishops & mask) return 'B'; if (b->black_bishops & mask) return 'b';
    if (b->white_queens & mask) return 'Q';  if (b->black_queens & mask) return 'q';
    if (b->white_king & mask) return 'K';    if (b->black_king & mask) return 'k';
    return '.';
}

void print_board(Board *b) {
    printf("\n");
    for (int rank = 7; rank >= 0; rank--) {
        printf("%d ", rank + 1);
        for (int file = 0; file < 8; file++) {
            printf("%c ", get_piece_on_square(b, rank * 8 + file));
        }
        printf("\n");
    }
    printf("  a b c d e f g h\n\n");
}

void print_move(uint32_t move) {
    int from = GET_FROM(move), to = GET_TO(move);
    printf("%c%c%c%c", 'a' + (from % 8), '1' + (from / 8), 'a' + (to % 8), '1' + (to / 8));
}

// ==========================================
// 4. INITIALISATION DU MOTEUR
// ==========================================

void init_leapers_and_masks() {
    for (int square = 0; square < 64; square++) {
        uint64_t bit = 1ULL << square;
        
        // Cavaliers
        uint64_t n = 0;
        n |= (bit << 17) & not_A_file;  n |= (bit << 15) & not_H_file;
        n |= (bit << 10) & not_AB_file; n |= (bit <<  6) & not_GH_file;
        n |= (bit >> 17) & not_H_file;  n |= (bit >> 15) & not_A_file;
        n |= (bit >> 10) & not_GH_file; n |= (bit >>  6) & not_AB_file;
        knight_attacks[square] = n;
        
        // Rois
        uint64_t k = 0;
        k |= (bit << 8) | (bit >> 8);
        k |= (bit << 1) & not_A_file; k |= (bit >> 1) & not_H_file;
        k |= (bit << 9) & not_A_file; k |= (bit << 7) & not_H_file;
        k |= (bit >> 7) & not_A_file; k |= (bit >> 9) & not_H_file;
        king_attacks[square] = k;

        // Masques Tours
        uint64_t r_mask = 0ULL;
        int tr = square / 8, tf = square % 8;
        for (int r = tr + 1; r <= 6; r++) r_mask |= (1ULL << (r * 8 + tf));
        for (int r = tr - 1; r >= 1; r--) r_mask |= (1ULL << (r * 8 + tf));
        for (int f = tf + 1; f <= 6; f++) r_mask |= (1ULL << (tr * 8 + f));
        for (int f = tf - 1; f >= 1; f--) r_mask |= (1ULL << (tr * 8 + f));
        rook_masks[square] = r_mask;

        // Masques Fous
        uint64_t b_mask = 0ULL;
        for (int r = tr + 1, f = tf + 1; r <= 6 && f <= 6; r++, f++) b_mask |= (1ULL << (r * 8 + f));
        for (int r = tr + 1, f = tf - 1; r <= 6 && f >= 1; r++, f--) b_mask |= (1ULL << (r * 8 + f));
        for (int r = tr - 1, f = tf + 1; r >= 1 && f <= 6; r--, f++) b_mask |= (1ULL << (r * 8 + f));
        for (int r = tr - 1, f = tf - 1; r >= 1 && f >= 1; r--, f--) b_mask |= (1ULL << (r * 8 + f));
        bishop_masks[square] = b_mask;
    }
}

uint64_t get_bishop_attacks_fallback(int square, uint64_t occupancy) {
    uint64_t attacks = 0ULL;
    int tr = square / 8, tf = square % 8;
    for (int r = tr + 1, f = tf + 1; r <= 7 && f <= 7; r++, f++) { attacks |= (1ULL << (r * 8 + f)); if (occupancy & (1ULL << (r * 8 + f))) break; }
    for (int r = tr + 1, f = tf - 1; r <= 7 && f >= 0; r++, f--) { attacks |= (1ULL << (r * 8 + f)); if (occupancy & (1ULL << (r * 8 + f))) break; }
    for (int r = tr - 1, f = tf + 1; r >= 0 && f <= 7; r--, f++) { attacks |= (1ULL << (r * 8 + f)); if (occupancy & (1ULL << (r * 8 + f))) break; }
    for (int r = tr - 1, f = tf - 1; r >= 0 && f >= 0; r--, f--) { attacks |= (1ULL << (r * 8 + f)); if (occupancy & (1ULL << (r * 8 + f))) break; }
    return attacks;
}

uint64_t get_rook_attacks_fallback(int square, uint64_t occupancy) {
    uint64_t attacks = 0ULL;
    int tr = square / 8, tf = square % 8;
    for (int r = tr + 1; r <= 7; r++) { attacks |= (1ULL << (r * 8 + tf)); if (occupancy & (1ULL << (r * 8 + tf))) break; }
    for (int r = tr - 1; r >= 0; r--) { attacks |= (1ULL << (r * 8 + tf)); if (occupancy & (1ULL << (r * 8 + tf))) break; }
    for (int f = tf + 1; f <= 7; f++) { attacks |= (1ULL << (tr * 8 + f)); if (occupancy & (1ULL << (tr * 8 + f))) break; }
    for (int f = tf - 1; f >= 0; f--) { attacks |= (1ULL << (tr * 8 + f)); if (occupancy & (1ULL << (tr * 8 + f))) break; }
    return attacks;
}

// ==========================================
// 5. GÉNÉRATEUR DE MOUVEMENTS (Move Generator)
// ==========================================

// Extrait les 4 options de promotion pour un pion arrivant au bout
void extract_promotions(uint64_t targets, int offset, int flag, MoveList *list) {
    while (targets) {
        int to = __builtin_ctzll(targets);
        int from = to - offset;
        int captured = (flag == FLAG_CAPTURE) ? 1 : 0;
        
        // On génère 4 coups (bits 20-23 remplis avec la pièce choisie)
        list->moves[list->count++] = ENCODE_MOVE(from, to, PAWN, captured, QUEEN, flag);
        list->moves[list->count++] = ENCODE_MOVE(from, to, PAWN, captured, ROOK, flag);
        list->moves[list->count++] = ENCODE_MOVE(from, to, PAWN, captured, BISHOP, flag);
        list->moves[list->count++] = ENCODE_MOVE(from, to, PAWN, captured, KNIGHT, flag);
        
        targets &= targets - 1;
    }
}




void extract_moves(int from, uint64_t targets, int piece, int flag, MoveList *list) {
    while (targets) {
        int to = __builtin_ctzll(targets);
        uint32_t move = ENCODE_MOVE(from, to, piece, (flag == FLAG_CAPTURE ? 1 : 0), 0, flag);
        list->moves[list->count++] = move;
        targets &= (targets - 1);
    }
}

void generate_moves(Board *b, MoveList *list, int color) {
    uint64_t white_pieces = b->white_pawns | b->white_rooks | b->white_knights | b->white_bishops | b->white_queens | b->white_king;
    uint64_t black_pieces = b->black_pawns | b->black_rooks | b->black_knights | b->black_bishops | b->black_queens | b->black_king;
    uint64_t occupied = white_pieces | black_pieces;
    uint64_t empty = ~occupied;

    uint64_t us = (color == WHITE) ? white_pieces : black_pieces;
    uint64_t them = (color == WHITE) ? black_pieces : white_pieces;

    // --- PIONS ---
    if (color == WHITE) {
        uint64_t single_push = (b->white_pawns << 8) & empty;
        uint64_t double_push = (single_push << 8) & empty & rank_4;
        uint64_t cap_left  = (b->white_pawns << 7) & not_H_file & them;
        uint64_t cap_right = (b->white_pawns << 9) & not_A_file & them;

        // Isoler les promotions
        uint64_t push_promos = single_push & rank_8;
        uint64_t cap_l_promos = cap_left & rank_8;
        uint64_t cap_r_promos = cap_right & rank_8;
        
        // Enlever les promotions des coups normaux
        single_push &= ~rank_8;
        cap_left &= ~rank_8;
        cap_right &= ~rank_8;

        // Extraire les coups normaux
        while(single_push) { int to = __builtin_ctzll(single_push); list->moves[list->count++] = ENCODE_MOVE(to-8, to, PAWN, 0, 0, FLAG_QUIET); single_push &= single_push - 1; }
        while(double_push) { int to = __builtin_ctzll(double_push); list->moves[list->count++] = ENCODE_MOVE(to-16, to, PAWN, 0, 0, FLAG_DOUBLE_PUSH); double_push &= double_push - 1; }
        while(cap_left)    { int to = __builtin_ctzll(cap_left);    list->moves[list->count++] = ENCODE_MOVE(to-7, to, PAWN, 1, 0, FLAG_CAPTURE); cap_left &= cap_left - 1; }
        while(cap_right)   { int to = __builtin_ctzll(cap_right);   list->moves[list->count++] = ENCODE_MOVE(to-9, to, PAWN, 1, 0, FLAG_CAPTURE); cap_right &= cap_right - 1; }
        
        // Extraire les promotions (offset = +8, +7, +9)
        extract_promotions(push_promos, 8, FLAG_QUIET, list);
        extract_promotions(cap_l_promos, 7, FLAG_CAPTURE, list);
        extract_promotions(cap_r_promos, 9, FLAG_CAPTURE, list);

    } else {
        uint64_t single_push = (b->black_pawns >> 8) & empty;
        uint64_t double_push = (single_push >> 8) & empty & rank_5;
        uint64_t cap_left  = (b->black_pawns >> 9) & not_H_file & them;
        uint64_t cap_right = (b->black_pawns >> 7) & not_A_file & them;

        uint64_t push_promos = single_push & rank_1;
        uint64_t cap_l_promos = cap_left & rank_1;
        uint64_t cap_r_promos = cap_right & rank_1;
        
        single_push &= ~rank_1;
        cap_left &= ~rank_1;
        cap_right &= ~rank_1;

        while(single_push) { int to = __builtin_ctzll(single_push); list->moves[list->count++] = ENCODE_MOVE(to+8, to, PAWN, 0, 0, FLAG_QUIET); single_push &= single_push - 1; }
        while(double_push) { int to = __builtin_ctzll(double_push); list->moves[list->count++] = ENCODE_MOVE(to+16, to, PAWN, 0, 0, FLAG_DOUBLE_PUSH); double_push &= double_push - 1; }
        while(cap_left)    { int to = __builtin_ctzll(cap_left);    list->moves[list->count++] = ENCODE_MOVE(to+9, to, PAWN, 1, 0, FLAG_CAPTURE); cap_left &= cap_left - 1; }
        while(cap_right)   { int to = __builtin_ctzll(cap_right);   list->moves[list->count++] = ENCODE_MOVE(to+7, to, PAWN, 1, 0, FLAG_CAPTURE); cap_right &= cap_right - 1; }
        
        // Extraire les promotions Noires (attention, l'offset mathématique est négatif ici)
        extract_promotions(push_promos, -8, FLAG_QUIET, list);
        extract_promotions(cap_l_promos, -9, FLAG_CAPTURE, list);
        extract_promotions(cap_r_promos, -7, FLAG_CAPTURE, list);
    }

    // --- CAVALIERS ET ROIS ---
    uint64_t knights = (color == WHITE) ? b->white_knights : b->black_knights;
    while (knights) {
        int from = __builtin_ctzll(knights);
        uint64_t valid = knight_attacks[from] & ~us;
        extract_moves(from, valid & empty, KNIGHT, FLAG_QUIET, list);
        extract_moves(from, valid & them, KNIGHT, FLAG_CAPTURE, list);
        knights &= knights - 1;
    }

    uint64_t king = (color == WHITE) ? b->white_king : b->black_king;
    if (king) {
        int from = __builtin_ctzll(king);
        uint64_t valid = king_attacks[from] & ~us;
        extract_moves(from, valid & empty, KING, FLAG_QUIET, list);
        extract_moves(from, valid & them, KING, FLAG_CAPTURE, list);
    }

    // --- LE ROQUE (CASTLING) ---
    if (color == WHITE) {
        // Petit Roque Blanc (K-side)
        if (b->castling_rights & CASTLE_WK) {
            // Cases f1 (index 5) et g1 (index 6) doivent être vides
            if (!(occupied & ((1ULL << 5) | (1ULL << 6)))) {
                // e1 (4), f1 (5) et g1 (6) ne doivent pas être attaquées
                if (!is_square_attacked(b, 4, BLACK) && 
                    !is_square_attacked(b, 5, BLACK) && 
                    !is_square_attacked(b, 6, BLACK)) {
                    list->moves[list->count++] = ENCODE_MOVE(4, 6, KING, 0, 0, FLAG_CASTLING);
                }
            }
        }
        // Grand Roque Blanc (Q-side)
        if (b->castling_rights & CASTLE_WQ) {
            // Cases b1 (1), c1 (2) et d1 (3) doivent être vides
            if (!(occupied & ((1ULL << 1) | (1ULL << 2) | (1ULL << 3)))) {
                // e1 (4), d1 (3) et c1 (2) ne doivent pas être attaquées
                if (!is_square_attacked(b, 4, BLACK) && 
                    !is_square_attacked(b, 3, BLACK) && 
                    !is_square_attacked(b, 2, BLACK)) {
                    list->moves[list->count++] = ENCODE_MOVE(4, 2, KING, 0, 0, FLAG_CASTLING);
                }
            }
        }
    } else {
        // Petit Roque Noir (K-side)
        if (b->castling_rights & CASTLE_BK) {
            // Cases f8 (61) et g8 (62) vides
            if (!(occupied & ((1ULL << 61) | (1ULL << 62)))) {
                if (!is_square_attacked(b, 60, WHITE) && 
                    !is_square_attacked(b, 61, WHITE) && 
                    !is_square_attacked(b, 62, WHITE)) {
                    list->moves[list->count++] = ENCODE_MOVE(60, 62, KING, 0, 0, FLAG_CASTLING);
                }
            }
        }
        // Grand Roque Noir (Q-side)
        if (b->castling_rights & CASTLE_BQ) {
            // Cases b8 (57), c8 (58) et d8 (59) vides
            if (!(occupied & ((1ULL << 57) | (1ULL << 58) | (1ULL << 59)))) {
                if (!is_square_attacked(b, 60, WHITE) && 
                    !is_square_attacked(b, 59, WHITE) && 
                    !is_square_attacked(b, 58, WHITE)) {
                    list->moves[list->count++] = ENCODE_MOVE(60, 58, KING, 0, 0, FLAG_CASTLING);
                }
            }
        }
    }




    // --- PIÈCES GLISSANTES ---
    uint64_t bishops = (color == WHITE) ? b->white_bishops : b->black_bishops;
    while (bishops) {
        int from = __builtin_ctzll(bishops);
        uint64_t valid = get_bishop_attacks_fallback(from, occupied) & ~us;
        extract_moves(from, valid & empty, BISHOP, FLAG_QUIET, list);
        extract_moves(from, valid & them, BISHOP, FLAG_CAPTURE, list);
        bishops &= bishops - 1;
    }

    uint64_t rooks = (color == WHITE) ? b->white_rooks : b->black_rooks;
    while (rooks) {
        int from = __builtin_ctzll(rooks);
        uint64_t valid = get_rook_attacks_fallback(from, occupied) & ~us;
        extract_moves(from, valid & empty, ROOK, FLAG_QUIET, list);
        extract_moves(from, valid & them, ROOK, FLAG_CAPTURE, list);
        rooks &= rooks - 1;
    }

    uint64_t queens = (color == WHITE) ? b->white_queens : b->black_queens;
    while (queens) {
        int from = __builtin_ctzll(queens);
        uint64_t valid = (get_rook_attacks_fallback(from, occupied) | get_bishop_attacks_fallback(from, occupied)) & ~us;
        extract_moves(from, valid & empty, QUEEN, FLAG_QUIET, list);
        extract_moves(from, valid & them, QUEEN, FLAG_CAPTURE, list);
        queens &= queens - 1;
    }
}

// ==========================================
// 6. MAKE MOVE (Appliquer un coup)
// ==========================================

void remove_piece(Board *b, int square, int color) {
    uint64_t mask = 1ULL << square;
    if (color == WHITE) {
        if (b->white_pawns & mask) b->white_pawns ^= mask;
        else if (b->white_knights & mask) b->white_knights ^= mask;
        else if (b->white_bishops & mask) b->white_bishops ^= mask;
        else if (b->white_rooks & mask) b->white_rooks ^= mask;
        else if (b->white_queens & mask) b->white_queens ^= mask;
    } else {
        if (b->black_pawns & mask) b->black_pawns ^= mask;
        else if (b->black_knights & mask) b->black_knights ^= mask;
        else if (b->black_bishops & mask) b->black_bishops ^= mask;
        else if (b->black_rooks & mask) b->black_rooks ^= mask;
        else if (b->black_queens & mask) b->black_queens ^= mask;
    }
}

void make_move(Board *b, uint32_t move, int color) {
    int from = GET_FROM(move);
    int to = GET_TO(move);
    int piece = GET_PIECE(move);
    int flags = GET_FLAGS(move);
    
    // Mise à jour des droits de Roque si le Roi ou une Tour bouge (ou est capturée)
    b->castling_rights &= castling_rights_update[from];
    b->castling_rights &= castling_rights_update[to];

    uint64_t move_mask = (1ULL << from) | (1ULL << to);

    if (flags == FLAG_CAPTURE) {
        remove_piece(b, to, (color == WHITE) ? BLACK : WHITE);
    }

    if (color == WHITE) {
        switch (piece) {
            case PAWN:   b->white_pawns   ^= move_mask; break;
            case KNIGHT: b->white_knights ^= move_mask; break;
            case BISHOP: b->white_bishops ^= move_mask; break;
            case ROOK:   b->white_rooks   ^= move_mask; break;
            case QUEEN:  b->white_queens  ^= move_mask; break;
            case KING:   b->white_king    ^= move_mask; break;
        }
    } else {
        switch (piece) {
            case PAWN:   b->black_pawns   ^= move_mask; break;
            case KNIGHT: b->black_knights ^= move_mask; break;
            case BISHOP: b->black_bishops ^= move_mask; break;
            case ROOK:   b->black_rooks   ^= move_mask; break;
            case QUEEN:  b->black_queens  ^= move_mask; break;
            case KING:   b->black_king    ^= move_mask; break;
        }
    }

  // --- GESTION DE LA PROMOTION ---
    int prom = GET_PROMOTION(move);
    if (prom) {
        if (color == WHITE) {
            b->white_pawns ^= (1ULL << to); // On efface le pion arrivé au bout
            switch (prom) { // On ajoute la nouvelle pièce
                case QUEEN:  b->white_queens  ^= (1ULL << to); break;
                case ROOK:   b->white_rooks   ^= (1ULL << to); break;
                case BISHOP: b->white_bishops ^= (1ULL << to); break;
                case KNIGHT: b->white_knights ^= (1ULL << to); break;
            }
        } else {
            b->black_pawns ^= (1ULL << to); // On efface le pion noir arrivé au bout
            switch (prom) {
                case QUEEN:  b->black_queens  ^= (1ULL << to); break;
                case ROOK:   b->black_rooks   ^= (1ULL << to); break;
                case BISHOP: b->black_bishops ^= (1ULL << to); break;
                case KNIGHT: b->black_knights ^= (1ULL << to); break;
            }
        }
    }

    // --- GESTION DU ROQUE (Déplacement de la Tour) ---
    if (flags == FLAG_CASTLING) {
        if (color == WHITE) {
            if (to == 6) b->white_rooks ^= (1ULL << 7) | (1ULL << 5); // h1 -> f1
            else if (to == 2) b->white_rooks ^= (1ULL << 0) | (1ULL << 3); // a1 -> d1
        } else {
            if (to == 62) b->black_rooks ^= (1ULL << 63) | (1ULL << 61); // h8 -> f8
            else if (to == 58) b->black_rooks ^= (1ULL << 56) | (1ULL << 59); // a8 -> d8
        }
    }

}

// ==========================================
// 7. ÉVALUATION ET INTELLIGENCE ARTIFICIELLE
// ==========================================

#define SCORE_PAWN   100
#define SCORE_KNIGHT 300
#define SCORE_BISHOP 300
#define SCORE_ROOK   500
#define SCORE_QUEEN  900

int evaluate_material(Board *b) {
    int score = 0;
    
    score += __builtin_popcountll(b->white_pawns)   * SCORE_PAWN;
    score += __builtin_popcountll(b->white_knights) * SCORE_KNIGHT;
    score += __builtin_popcountll(b->white_bishops) * SCORE_BISHOP;
    score += __builtin_popcountll(b->white_rooks)   * SCORE_ROOK;
    score += __builtin_popcountll(b->white_queens)  * SCORE_QUEEN;
    
    score -= __builtin_popcountll(b->black_pawns)   * SCORE_PAWN;
    score -= __builtin_popcountll(b->black_knights) * SCORE_KNIGHT;
    score -= __builtin_popcountll(b->black_bishops) * SCORE_BISHOP;
    score -= __builtin_popcountll(b->black_rooks)   * SCORE_ROOK;
    score -= __builtin_popcountll(b->black_queens)  * SCORE_QUEEN;
    
    return score;
}

// Tableaux de valeurs par case (Piece-Square Tables)
// Bonus pour encourager les pions à contrôler le centre et à avancer
const int pawn_pst[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
};

// Bonus pour centraliser les cavaliers (malus sur les bords)
const int knight_pst[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

int evaluate_positional(Board *b) {
    // 1. On prend le score de base (le matériel)
    int score = evaluate_material(b);
    
    // 2. On ajoute les bonus positionnels en extrayant les bits [cite: 13-25]
    uint64_t wp = b->white_pawns;
    while(wp) { int sq = __builtin_ctzll(wp); score += pawn_pst[sq]; wp &= wp - 1; }
    
    uint64_t bp = b->black_pawns;
    while(bp) { int sq = __builtin_ctzll(bp); score -= pawn_pst[sq ^ 56]; bp &= bp - 1; } // ^ 56 inverse le plateau
    
    uint64_t wn = b->white_knights;
    while(wn) { int sq = __builtin_ctzll(wn); score += knight_pst[sq]; wn &= wn - 1; }
    
    uint64_t bn = b->black_knights;
    while(bn) { int sq = __builtin_ctzll(bn); score -= knight_pst[sq ^ 56]; bn &= bn - 1; }
    
    // (Dans un vrai moteur, on ferait ça pour les Fous, les Rois, etc.)
    return score;
}

// Vérifie si une 'square' est attaquée par la couleur 'attacker_color'
int is_square_attacked(Board *b, int square, int attacker_color) {
    uint64_t occupied = b->white_pawns | b->white_knights | b->white_bishops | b->white_rooks | b->white_queens | b->white_king |
                        b->black_pawns | b->black_knights | b->black_bishops | b->black_rooks | b->black_queens | b->black_king;
    
    // 1. Attaqué par des Pions ? (On regarde en diagonale inverse)
    if (attacker_color == WHITE) {
        if ((1ULL << square) & ((b->white_pawns << 7) & not_H_file)) return 1;
        if ((1ULL << square) & ((b->white_pawns << 9) & not_A_file)) return 1;
    } else {
        if ((1ULL << square) & ((b->black_pawns >> 9) & not_H_file)) return 1;
        if ((1ULL << square) & ((b->black_pawns >> 7) & not_A_file)) return 1;
    }
    
    // 2. Attaqué par des Cavaliers ?
    uint64_t enemy_knights = (attacker_color == WHITE) ? b->white_knights : b->black_knights;
    if (knight_attacks[square] & enemy_knights) return 1;
    
    // 3. Attaqué par le Roi adverse ?
    uint64_t enemy_king = (attacker_color == WHITE) ? b->white_king : b->black_king;
    if (king_attacks[square] & enemy_king) return 1;
    
    // 4. Attaqué par des Fous ou des Reines ?
    uint64_t b_q = (attacker_color == WHITE) ? (b->white_bishops | b->white_queens) : (b->black_bishops | b->black_queens);
    if (get_bishop_attacks_fallback(square, occupied) & b_q) return 1;
    
    // 5. Attaqué par des Tours ou des Reines ?
    uint64_t r_q = (attacker_color == WHITE) ? (b->white_rooks | b->white_queens) : (b->black_rooks | b->black_queens);
    if (get_rook_attacks_fallback(square, occupied) & r_q) return 1;
    
    return 0; // La case est en sécurité !
}

// Renvoie 0 (Continue), 1 (Échec et Mat), ou 2 (Pat/Stalemate)
int check_game_over(Board *b, int color) {
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    
    int legal_moves = 0;
    
    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        
        // On cherche le Roi sur le NOUVEAU plateau (au cas où c'est lui qui a bougé)
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        int king_square = __builtin_ctzll(my_king);
        
        // Si le Roi n'est pas attaqué après ce coup, le coup est légal !
        if (!is_square_attacked(b, king_square, 1 - color)) {
            legal_moves++;
        }
        
        *b = backup; // Annulation
        
        // Optimisation HPC : dès qu'on trouve 1 coup légal, la partie n'est pas finie, on quitte !
        if (legal_moves > 0) return 0; 
    }
    
    // Si on arrive ici, il n'y a AUCUN coup légal.
    uint64_t current_king = (color == WHITE) ? b->white_king : b->black_king;
    int current_king_square = __builtin_ctzll(current_king);
    
    if (is_square_attacked(b, current_king_square, 1 - color)) {
        return 1; // Échec et Mat !
    } else {
        return 2; // Pat !
    }
}



// Donne un score arbitraire à un coup pour le tri
int score_move(uint32_t move) {
    int score = 0;
    if (GET_FLAGS(move) == FLAG_CAPTURE) score += 10000;
    if (GET_PROMOTION(move) == QUEEN)    score += 9000; // Très important !
    return score;
}

// Tri par insertion (le plus rapide pour des petits tableaux comme MoveList)
void sort_moves(MoveList *list) {
    for (int i = 1; i < list->count; i++) {
        uint32_t key = list->moves[i];
        int key_score = score_move(key);
        int j = i - 1;
        
        while (j >= 0 && score_move(list->moves[j]) < key_score) {
            list->moves[j + 1] = list->moves[j];
            j = j - 1;
        }
        list->moves[j + 1] = key;
    }
}

int alpha_beta(Board *b, int depth, int alpha, int beta, int color, int eval_type) {
    if (depth == 0) {
        int eval = (eval_type == 1) ? evaluate_positional(b) : evaluate_material(b);
        return (color == WHITE) ? eval : -eval; 
    }
    // ==========================================
    // NULL MOVE PRUNING (NMP)
    // ==========================================
    // Conditions : On ne doit pas être à la profondeur 0, on ne doit pas être en échec,
    // et il est déconseillé de le faire avec des valeurs alpha/beta trop proches (fenêtre nulle).
    
    int R = 2; // Facteur de réduction (généralement 2 ou 3)
    total_nodes ++;
    
    // On vérifie qu'on a le droit de tenter un Null Move
    if (depth >= 3) {
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        int king_square = __builtin_ctzll(my_king);
        
        // On ne fait PAS de Null Move si on est en échec !
        if (!is_square_attacked(b, king_square, 1 - color)) {
            
            Board null_backup = *b;
            
            // On "passe" notre tour artificiellement
            b->en_passant_square = -1; // On annule les prises en passant possibles
            // On lance une recherche avec profondeur réduite et on change de couleur
            int null_score = -alpha_beta(b, depth - 1 - R, -beta, -beta + 1, 1 - color, eval_type);
            
            *b = null_backup; // On restaure l'échiquier
            
            // Si le score est toujours supérieur à beta, on coupe !
            if (null_score >= beta) {
                return beta; 
            }
        }
    }  


    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    
    // MOVE ORDERING : On trie les coups pour l'optimisation
    sort_moves(&list);
    
    int best_score = -INF;
    int legal_moves_played = 0; // Pour détecter l'Échec et Mat
    
    for (int i = 0; i < list.count; i++) {
        Board backup = *b; 
        make_move(b, list.moves[i], color);
        
        // --- VÉRIFICATION DE LA LÉGALITÉ ---
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        int king_square = __builtin_ctzll(my_king);
        
        if (is_square_attacked(b, king_square, 1 - color)) {
            *b = backup; 
            continue;    
        }
        
        legal_moves_played++; 
        
        int score;
        int is_capture = (GET_FLAGS(list.moves[i]) == FLAG_CAPTURE);
        int is_promotion = (GET_PROMOTION(list.moves[i]) != 0);
        
        // ==========================================
        // LATE MOVE REDUCTIONS (LMR)
        // ==========================================
        // On ne réduit que si :
        // 1. On est assez profond dans l'arbre (depth >= 3)
        // 2. On a déjà cherché les 3 premiers coups "prometteurs" (legal_moves_played > 3)
        // 3. Ce n'est ni une capture, ni une promotion
        
        if (depth >= 3 && legal_moves_played > 3 && !is_capture && !is_promotion) {
            
            // On fait une recherche RÉDUITE d'un cran supplémentaire (depth - 2 au lieu de depth - 1)
            score = -alpha_beta(b, depth - 2, -beta, -alpha, 1 - color, eval_type);
            
            // Si le coup s'avère meilleur que prévu, on refait une recherche normale complète !
            if (score > alpha) {
                score = -alpha_beta(b, depth - 1, -beta, -alpha, 1 - color, eval_type);
            }
        } else {
            // Recherche normale pour les premiers coups, les captures et promotions
            score = -alpha_beta(b, depth - 1, -beta, -alpha, 1 - color, eval_type);
        }
        
        *b = backup; 
        
        if (score > best_score) best_score = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break; // Alpha-Beta classique
    }
    
    // S'il n'y a aucun coup légal, c'est soit un Échec et Mat, soit un Pat !
    if (legal_moves_played == 0) {
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        int king_square = __builtin_ctzll(my_king);
        if (is_square_attacked(b, king_square, 1 - color)) {
            return -INF + (4 - depth); // Mat ! (Plus c'est rapide, pire c'est)
        }
        return 0; // Pat (Égalité)
    }
    
    return best_score;
}


uint32_t search_best_move(Board *b, int depth, int color, int eval_type) {
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    
    // On trie les coups à la racine pour optimiser l'Alpha-Bêta
    sort_moves(&list);
    
    int best_score = -INF;
    uint32_t best_move = 0;
    int alpha = -INF, beta = INF;
    
    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        
        // --- LE CORRECTIF DE LÉGALITÉ EST ICI ---
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        int king_square = __builtin_ctzll(my_king);
        
        // Si ce coup laisse notre Roi en échec, c'est un suicide, on l'ignore !
        if (is_square_attacked(b, king_square, 1 - color)) {
            *b = backup;
            continue; 
        }
        // ----------------------------------------
        
        int score = -alpha_beta(b, depth - 1, -beta, -alpha, 1 - color, eval_type);
        *b = backup;
        
        if (score > best_score) {
            best_score = score;
            best_move = list.moves[i];
        }
        if (score > alpha) alpha = score;
    }
    
    return best_move;
}

// ==========================================
// 9. GÉNÉRATION DE DONNÉES (DEEP LEARNING)
// ==========================================

// Traduit l'échiquier 64 bits en un Tenseur (tableau plat) de 768 valeurs (0.0 ou 1.0)
void board_to_tensor(Board *b, float tensor[768]) {
    // On initialise tout à 0.0
    for (int i = 0; i < 768; i++) {
        tensor[i] = 0.0f;
    }

    // Un tableau de pointeurs vers tes 12 bitboards pour faire une boucle propre
    uint64_t* bitboards[12] = {
        &b->white_pawns, &b->white_knights, &b->white_bishops, &b->white_rooks, &b->white_queens, &b->white_king,
        &b->black_pawns, &b->black_knights, &b->black_bishops, &b->black_rooks, &b->black_queens, &b->black_king
    };

    // On parcourt les 12 bitboards
    for (int piece_type = 0; piece_type < 12; piece_type++) {
        uint64_t bb = *(bitboards[piece_type]);
        
        // On extrait les bits 1 par 1
        while (bb) {
            int square = __builtin_ctzll(bb);
            // L'index mathématique : (Type de pièce * 64) + Case
            int index = (piece_type * 64) + square;
            tensor[index] = 1.0f;
            
            bb &= bb - 1; // On efface le bit traité
        }
    }
}


// Exporte la position et son évaluation dans un fichier CSV
void export_to_csv(FILE *file, Board *b, int evaluation_score) {
    float tensor[768];
    board_to_tensor(b, tensor);

    // On écrit les 768 valeurs du plateau
    for (int i = 0; i < 768; i++) {
        // Optimisation de l'écriture : on écrit '1' ou '0' sans décimales pour gagner de la place
        fprintf(file, "%d,", (int)tensor[i]); 
    }
    
    // On ajoute le score à la toute fin de la ligne
    fprintf(file, "%d\n", evaluation_score);
}

// Fonction utilitaire pour le Machine Learning :
// Cherche le meilleur coup ET renvoie l'évaluation exacte du plateau
uint32_t search_and_score(Board *b, int depth, int color, int eval_type, int *out_score) {
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    sort_moves(&list);
    
    int best_score = -INF;
    uint32_t best_move = 0;
    int alpha = -INF, beta = INF;
    
    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        int king_square = __builtin_ctzll(my_king);
        
        if (is_square_attacked(b, king_square, 1 - color)) {
            *b = backup;
            continue; 
        }
        
        int score = -alpha_beta(b, depth - 1, -beta, -alpha, 1 - color, eval_type);
        *b = backup;
        
        if (score > best_score) {
            best_score = score;
            best_move = list.moves[i];
        }
        if (score > alpha) alpha = score;
    }
    
    // Si la liste des coups légaux est vide ou ne mène qu'à des échecs
    if (best_move == 0 && list.count > 0) {
        best_move = list.moves[0]; // Sécurité de repli
    }
    
    *out_score = best_score;
    return best_move;
}







// ==========================================
// 8. TEST PRINCIPAL
// ==========================================
int main() {
    init_leapers_and_masks();
    srand(time(NULL)); // Initialisation de l'aléatoire

    printf("===================================================\n");
    printf("   GENERATEUR DE DATASET (APPRENTISSAGE SUPERVISE)\n");
    printf("===================================================\n");

    FILE *dataset_file = fopen("dataset.csv", "w");
    if (dataset_file == NULL) {
        printf("Erreur : Impossible de creer dataset.csv\n");
        return 1;
    }

    int total_games = 100;    // Nombre de parties à générer (à augmenter plus tard !)
    int depth = 5;            // Profondeur d'évaluation (5 est un bon compromis vitesse/qualité)
    int random_plies = 4;     // Nombre de demi-coups joués au hasard au début (Exploration)
    int max_moves = 200;      // Limite de demi-coups par partie (pour éviter les boucles infinies)
    
    int total_positions_saved = 0;

    for (int game = 1; game <= total_games; game++) {
        // Plateau de départ pour chaque nouvelle partie
        Board board = {
            .white_pawns   = 0x000000000000FF00ULL, .white_rooks   = 0x0000000000000081ULL,
            .white_knights = 0x0000000000000042ULL, .white_bishops = 0x0000000000000024ULL,
            .white_queens  = 0x0000000000000008ULL, .white_king    = 0x0000000000000010ULL,
            .black_pawns   = 0x00FF000000000000ULL, .black_rooks   = 0x8100000000000000ULL,
            .black_knights = 0x4200000000000000ULL, .black_bishops = 0x2400000000000000ULL,
            .black_queens  = 0x0800000000000000ULL, .black_king    = 0x1000000000000000ULL,
            .castling_rights = 15, .en_passant_square = -1, .halfmove_clock = 0
        };

        printf("Generation de la partie %d / %d...\n", game, total_games);

        for (int turn = 0; turn < max_moves; turn++) {
            int color = turn % 2;

            if (check_game_over(&board, color) != 0) break;

            uint32_t chosen_move = 0;
            int score = 0;

            // --- PHASE 1 : EXPLORATION ALÉATOIRE ---
            if (turn < random_plies) {
                MoveList list; list.count = 0;
                generate_moves(&board, &list, color);
                
                // On filtre les coups illégaux pour ne pas faire crasher le moteur
                uint32_t legal_moves[256];
                int legal_count = 0;
                for(int i=0; i<list.count; i++) {
                    Board backup = board;
                    make_move(&board, list.moves[i], color);
                    uint64_t my_king = (color == WHITE) ? board.white_king : board.black_king;
                    if (!is_square_attacked(&board, __builtin_ctzll(my_king), 1 - color)) {
                        legal_moves[legal_count++] = list.moves[i];
                    }
                    board = backup;
                }
                
                if (legal_count > 0) {
                    chosen_move = legal_moves[rand() % legal_count];
                } else {
                    break; // Fin de partie prématurée
                }
                
                // Même sur un coup aléatoire, on évalue la position pour le Dataset
                score = alpha_beta(&board, depth, -INF, INF, color, 1);
            } 
            // --- PHASE 2 : JEU OPTIMAL (ALPHA-BETA) ---
            else {
                chosen_move = search_and_score(&board, depth, color, 1, &score);
                if (chosen_move == 0) break;
            }

            // Normalisation : On s'assure que le score est toujours vu du côté des Blancs
            // (Si c'est aux Noirs de jouer, un score positif pour eux est négatif pour les Blancs)
            int absolute_score = (color == WHITE) ? score : -score;

            // EXPORTATION DE LA DONNÉE
            export_to_csv(dataset_file, &board, absolute_score);
            total_positions_saved++;

            // Application du coup pour passer au tour suivant
            make_move(&board, chosen_move, color);
        }
    }

    fclose(dataset_file);
    printf("\nGeneration terminee ! %d positions ont ete sauvegardees dans 'dataset.csv'.\n", total_positions_saved);
    printf("===================================================\n");

    return 0;
}

