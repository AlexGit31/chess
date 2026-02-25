#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ==========================================
// 1. CONSTANTES ET MACROS HPC
// ==========================================

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
#define FLAG_CASTLING    3
#define INF 50000

#define CASTLE_WK 1 
#define CASTLE_WQ 2 
#define CASTLE_BK 4 
#define CASTLE_BQ 8 

const uint64_t not_A_file  = 0xFEFEFEFEFEFEFEFEULL;
const uint64_t not_H_file  = 0x7F7F7F7F7F7F7F7FULL;
const uint64_t not_AB_file = 0xFCFCFCFCFCFCFCFCULL;
const uint64_t not_GH_file = 0x3F3F3F3F3F3F3F3FULL;
const uint64_t rank_4      = 0x00000000FF000000ULL;
const uint64_t rank_5      = 0x000000FF00000000ULL;
const uint64_t rank_1      = 0x00000000000000FFULL;
const uint64_t rank_8      = 0xFF00000000000000ULL;

#define ENCODE_MOVE(from, to, piece, captured, promotion, flags) \
    ( (from) | ((to) << 6) | ((piece) << 12) | ((captured) << 16) | ((promotion) << 20) | ((flags) << 24) )

#define GET_FROM(move)      ( (move) & 0x3F )
#define GET_TO(move)        ( ((move) >> 6) & 0x3F )
#define GET_PIECE(move)     ( ((move) >> 12) & 0xF )
#define GET_FLAGS(move)     ( ((move) >> 24) & 0xFF )
#define GET_CAPTURED(move)  ( ((move) >> 16) & 0xF )
#define GET_PROMOTION(move) ( ((move) >> 20) & 0xF )

const int castling_rights_update[64] = {
     13, 15, 15, 15, 12, 15, 15, 14,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
      7, 15, 15, 15,  3, 15, 15, 11 
};

// ==========================================
// 2. STRUCTURES DE DONNÉES
// ==========================================

typedef struct {
    uint64_t white_pawns, white_rooks, white_knights;
    uint64_t white_bishops, white_queens, white_king;
    uint64_t black_pawns, black_rooks, black_knights;
    uint64_t black_bishops, black_queens, black_king;
    int castling_rights; 
    int en_passant_square; 
    int halfmove_clock; 
} Board;

typedef struct {
    uint32_t moves[256];
    int count;
} MoveList;

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
        uint64_t n = 0;
        n |= (bit << 17) & not_A_file;  n |= (bit << 15) & not_H_file;
        n |= (bit << 10) & not_AB_file; n |= (bit <<  6) & not_GH_file;
        n |= (bit >> 17) & not_H_file;  n |= (bit >> 15) & not_A_file;
        n |= (bit >> 10) & not_GH_file; n |= (bit >>  6) & not_AB_file;
        knight_attacks[square] = n;
        
        uint64_t k = 0;
        k |= (bit << 8) | (bit >> 8);
        k |= (bit << 1) & not_A_file; k |= (bit >> 1) & not_H_file;
        k |= (bit << 9) & not_A_file; k |= (bit << 7) & not_H_file;
        k |= (bit >> 7) & not_A_file; k |= (bit >> 9) & not_H_file;
        king_attacks[square] = k;

        uint64_t r_mask = 0ULL;
        int tr = square / 8, tf = square % 8;
        for (int r = tr + 1; r <= 6; r++) r_mask |= (1ULL << (r * 8 + tf));
        for (int r = tr - 1; r >= 1; r--) r_mask |= (1ULL << (r * 8 + tf));
        for (int f = tf + 1; f <= 6; f++) r_mask |= (1ULL << (tr * 8 + f));
        for (int f = tf - 1; f >= 1; f--) r_mask |= (1ULL << (tr * 8 + f));
        rook_masks[square] = r_mask;

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
// 5. GÉNÉRATEUR DE MOUVEMENTS
// ==========================================

void extract_promotions(uint64_t targets, int offset, int flag, MoveList *list) {
    while (targets) {
        int to = __builtin_ctzll(targets);
        int from = to - offset;
        int captured = (flag == FLAG_CAPTURE) ? 1 : 0;
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

    if (color == WHITE) {
        uint64_t single_push = (b->white_pawns << 8) & empty;
        uint64_t double_push = (single_push << 8) & empty & rank_4;
        uint64_t cap_left  = (b->white_pawns << 7) & not_H_file & them;
        uint64_t cap_right = (b->white_pawns << 9) & not_A_file & them;

        uint64_t push_promos = single_push & rank_8;
        uint64_t cap_l_promos = cap_left & rank_8;
        uint64_t cap_r_promos = cap_right & rank_8;
        
        single_push &= ~rank_8; cap_left &= ~rank_8; cap_right &= ~rank_8;

        while(single_push) { int to = __builtin_ctzll(single_push); list->moves[list->count++] = ENCODE_MOVE(to-8, to, PAWN, 0, 0, FLAG_QUIET); single_push &= single_push - 1; }
        while(double_push) { int to = __builtin_ctzll(double_push); list->moves[list->count++] = ENCODE_MOVE(to-16, to, PAWN, 0, 0, FLAG_DOUBLE_PUSH); double_push &= double_push - 1; }
        while(cap_left)    { int to = __builtin_ctzll(cap_left);    list->moves[list->count++] = ENCODE_MOVE(to-7, to, PAWN, 1, 0, FLAG_CAPTURE); cap_left &= cap_left - 1; }
        while(cap_right)   { int to = __builtin_ctzll(cap_right);   list->moves[list->count++] = ENCODE_MOVE(to-9, to, PAWN, 1, 0, FLAG_CAPTURE); cap_right &= cap_right - 1; }
        
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
        
        single_push &= ~rank_1; cap_left &= ~rank_1; cap_right &= ~rank_1;

        while(single_push) { int to = __builtin_ctzll(single_push); list->moves[list->count++] = ENCODE_MOVE(to+8, to, PAWN, 0, 0, FLAG_QUIET); single_push &= single_push - 1; }
        while(double_push) { int to = __builtin_ctzll(double_push); list->moves[list->count++] = ENCODE_MOVE(to+16, to, PAWN, 0, 0, FLAG_DOUBLE_PUSH); double_push &= double_push - 1; }
        while(cap_left)    { int to = __builtin_ctzll(cap_left);    list->moves[list->count++] = ENCODE_MOVE(to+9, to, PAWN, 1, 0, FLAG_CAPTURE); cap_left &= cap_left - 1; }
        while(cap_right)   { int to = __builtin_ctzll(cap_right);   list->moves[list->count++] = ENCODE_MOVE(to+7, to, PAWN, 1, 0, FLAG_CAPTURE); cap_right &= cap_right - 1; }
        
        extract_promotions(push_promos, -8, FLAG_QUIET, list);
        extract_promotions(cap_l_promos, -9, FLAG_CAPTURE, list);
        extract_promotions(cap_r_promos, -7, FLAG_CAPTURE, list);
    }

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

    if (color == WHITE) {
        if ((b->castling_rights & CASTLE_WK) && !(occupied & ((1ULL << 5) | (1ULL << 6)))) {
            if (!is_square_attacked(b, 4, BLACK) && !is_square_attacked(b, 5, BLACK) && !is_square_attacked(b, 6, BLACK))
                list->moves[list->count++] = ENCODE_MOVE(4, 6, KING, 0, 0, FLAG_CASTLING);
        }
        if ((b->castling_rights & CASTLE_WQ) && !(occupied & ((1ULL << 1) | (1ULL << 2) | (1ULL << 3)))) {
            if (!is_square_attacked(b, 4, BLACK) && !is_square_attacked(b, 3, BLACK) && !is_square_attacked(b, 2, BLACK))
                list->moves[list->count++] = ENCODE_MOVE(4, 2, KING, 0, 0, FLAG_CASTLING);
        }
    } else {
        if ((b->castling_rights & CASTLE_BK) && !(occupied & ((1ULL << 61) | (1ULL << 62)))) {
            if (!is_square_attacked(b, 60, WHITE) && !is_square_attacked(b, 61, WHITE) && !is_square_attacked(b, 62, WHITE))
                list->moves[list->count++] = ENCODE_MOVE(60, 62, KING, 0, 0, FLAG_CASTLING);
        }
        if ((b->castling_rights & CASTLE_BQ) && !(occupied & ((1ULL << 57) | (1ULL << 58) | (1ULL << 59)))) {
            if (!is_square_attacked(b, 60, WHITE) && !is_square_attacked(b, 59, WHITE) && !is_square_attacked(b, 58, WHITE))
                list->moves[list->count++] = ENCODE_MOVE(60, 58, KING, 0, 0, FLAG_CASTLING);
        }
    }

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
// 6. MAKE MOVE
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
    
    b->castling_rights &= castling_rights_update[from];
    b->castling_rights &= castling_rights_update[to];

    uint64_t move_mask = (1ULL << from) | (1ULL << to);

    if (flags == FLAG_CAPTURE) remove_piece(b, to, (color == WHITE) ? BLACK : WHITE);

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

    int prom = GET_PROMOTION(move);
    if (prom) {
        if (color == WHITE) {
            b->white_pawns ^= (1ULL << to);
            switch (prom) {
                case QUEEN:  b->white_queens  ^= (1ULL << to); break;
                case ROOK:   b->white_rooks   ^= (1ULL << to); break;
                case BISHOP: b->white_bishops ^= (1ULL << to); break;
                case KNIGHT: b->white_knights ^= (1ULL << to); break;
            }
        } else {
            b->black_pawns ^= (1ULL << to);
            switch (prom) {
                case QUEEN:  b->black_queens  ^= (1ULL << to); break;
                case ROOK:   b->black_rooks   ^= (1ULL << to); break;
                case BISHOP: b->black_bishops ^= (1ULL << to); break;
                case KNIGHT: b->black_knights ^= (1ULL << to); break;
            }
        }
    }

    if (flags == FLAG_CASTLING) {
        if (color == WHITE) {
            if (to == 6) b->white_rooks ^= (1ULL << 7) | (1ULL << 5); 
            else if (to == 2) b->white_rooks ^= (1ULL << 0) | (1ULL << 3); 
        } else {
            if (to == 62) b->black_rooks ^= (1ULL << 63) | (1ULL << 61); 
            else if (to == 58) b->black_rooks ^= (1ULL << 56) | (1ULL << 59); 
        }
    }
}

// ==========================================
// 7. ÉVALUATION ET INTELLIGENCE ARTIFICIELLE
// ==========================================

int is_square_attacked(Board *b, int square, int attacker_color) {
    uint64_t occupied = b->white_pawns | b->white_knights | b->white_bishops | b->white_rooks | b->white_queens | b->white_king |
                        b->black_pawns | b->black_knights | b->black_bishops | b->black_rooks | b->black_queens | b->black_king;
    
    if (attacker_color == WHITE) {
        if ((1ULL << square) & ((b->white_pawns << 7) & not_H_file)) return 1;
        if ((1ULL << square) & ((b->white_pawns << 9) & not_A_file)) return 1;
    } else {
        if ((1ULL << square) & ((b->black_pawns >> 9) & not_H_file)) return 1;
        if ((1ULL << square) & ((b->black_pawns >> 7) & not_A_file)) return 1;
    }
    
    uint64_t enemy_knights = (attacker_color == WHITE) ? b->white_knights : b->black_knights;
    if (knight_attacks[square] & enemy_knights) return 1;
    
    uint64_t enemy_king = (attacker_color == WHITE) ? b->white_king : b->black_king;
    if (king_attacks[square] & enemy_king) return 1;
    
    uint64_t b_q = (attacker_color == WHITE) ? (b->white_bishops | b->white_queens) : (b->black_bishops | b->black_queens);
    if (get_bishop_attacks_fallback(square, occupied) & b_q) return 1;
    
    uint64_t r_q = (attacker_color == WHITE) ? (b->white_rooks | b->white_queens) : (b->black_rooks | b->black_queens);
    if (get_rook_attacks_fallback(square, occupied) & r_q) return 1;
    
    return 0; 
}

int check_game_over(Board *b, int color) {
    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    
    int legal_moves = 0;
    for (int i = 0; i < list.count; i++) {
        Board backup = *b;
        make_move(b, list.moves[i], color);
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (!is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) legal_moves++;
        *b = backup;
        if (legal_moves > 0) return 0; 
    }
    
    uint64_t current_king = (color == WHITE) ? b->white_king : b->black_king;
    if (is_square_attacked(b, __builtin_ctzll(current_king), 1 - color)) return 1;
    else return 2;
}

int score_move(uint32_t move) {
    int score = 0;
    if (GET_FLAGS(move) == FLAG_CAPTURE) score += 10000;
    if (GET_PROMOTION(move) == QUEEN)    score += 9000; 
    return score;
}

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

// ==========================================
// 8. RÉSEAU DE NEURONES (NNUE C-Natif)
// ==========================================

// Tableaux globaux pour stocker les poids du réseau PyTorch
float W1[256][768]; float b1[256];
float W2[32][256];  float b2[32];
float W3[32];       float b3; 

// Charge le fichier binaire généré par Python
int load_nnue_weights(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { 
        printf("Attention: Fichier NNUE '%s' introuvable ! \n", filename); 
        return 0; 
    }
    
    // Lecture séquentielle (PyTorch écrit par couches : Couche 1, puis 2, puis 3)
    fread(W1, sizeof(float), 256 * 768, f);
    fread(b1, sizeof(float), 256, f);
    
    fread(W2, sizeof(float), 32 * 256, f);
    fread(b2, sizeof(float), 32, f);
    
    fread(W3, sizeof(float), 32, f);
    fread(&b3, sizeof(float), 1, f);
    
    fclose(f);
    printf("Reseau de neurones charge avec succes !\n");
    return 1;
}

// Extraction de l'échiquier en tenseur
void board_to_tensor(Board *b, float tensor[768]) {
    for (int i = 0; i < 768; i++) tensor[i] = 0.0f;

    uint64_t* bitboards[12] = {
        &b->white_pawns, &b->white_knights, &b->white_bishops, &b->white_rooks, &b->white_queens, &b->white_king,
        &b->black_pawns, &b->black_knights, &b->black_bishops, &b->black_rooks, &b->black_queens, &b->black_king
    };

    for (int piece_type = 0; piece_type < 12; piece_type++) {
        uint64_t bb = *(bitboards[piece_type]);
        while (bb) {
            int square = __builtin_ctzll(bb);
            tensor[(piece_type * 64) + square] = 1.0f;
            bb &= bb - 1; 
        }
    }
}

// Fonction d'activation ReLU
#define RELU(x) ((x) > 0.0f ? (x) : 0.0f)

// L'évaluation pure par le réseau de neurones !
int evaluate_nnue(Board *b) {
    float input[768];
    board_to_tensor(b, input);
    
    // Couche 1 (L'astuce HPC : on ne multiplie que si l'input est 1.0)
    float hidden1[256];
    for (int i = 0; i < 256; i++) {
        float sum = b1[i];
        for (int j = 0; j < 768; j++) {
            if (input[j] > 0.5f) { // Évite 196 000 multiplications inutiles !
                sum += W1[i][j];
            }
        }
        hidden1[i] = RELU(sum);
    }
    
    // Couche 2
    float hidden2[32];
    for (int i = 0; i < 32; i++) {
        float sum = b2[i];
        for (int j = 0; j < 256; j++) {
            sum += W2[i][j] * hidden1[j];
        }
        hidden2[i] = RELU(sum);
    }
    
    // Couche 3 (Sortie Finale)
    float output = b3;
    for (int j = 0; j < 32; j++) {
        output += W3[j] * hidden2[j];
    }
    
    // On re-multiplie par 100 pour remettre à l'échelle de l'Alpha-Bêta (Centipions)
    return (int)(output * 100.0f);
}


// ==========================================
// 9. ALPHA-BETA (NMP & LMR)
// ==========================================

int alpha_beta(Board *b, int depth, int alpha, int beta, int color, int use_nnue) {
    if (depth <= 0) {
        // Le réseau a appris la position vu des Blancs, il faut inverser pour les Noirs
        int eval = evaluate_nnue(b);
        return (color == WHITE) ? eval : -eval; 
    }
    
    // NULL MOVE PRUNING
    if (depth >= 3) {
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (!is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) {
            Board null_backup = *b;
            b->en_passant_square = -1; 
            int null_score = -alpha_beta(b, depth - 1 - 2, -beta, -beta + 1, 1 - color, use_nnue);
            *b = null_backup; 
            if (null_score >= beta) return beta; 
        }
    }

    MoveList list; list.count = 0;
    generate_moves(b, &list, color);
    sort_moves(&list);
    
    int best_score = -INF;
    int legal_moves_played = 0; 
    
    for (int i = 0; i < list.count; i++) {
        Board backup = *b; 
        make_move(b, list.moves[i], color);
        
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) {
            *b = backup; continue;    
        }
        
        legal_moves_played++; 
        int score;
        int is_cap = (GET_FLAGS(list.moves[i]) == FLAG_CAPTURE);
        int is_prom = (GET_PROMOTION(list.moves[i]) != 0);
        
        // LATE MOVE REDUCTIONS
        if (depth >= 3 && legal_moves_played > 3 && !is_cap && !is_prom) {
            score = -alpha_beta(b, depth - 2, -beta, -alpha, 1 - color, use_nnue);
            if (score > alpha) {
                score = -alpha_beta(b, depth - 1, -beta, -alpha, 1 - color, use_nnue);
            }
        } else {
            score = -alpha_beta(b, depth - 1, -beta, -alpha, 1 - color, use_nnue);
        }
        
        *b = backup; 
        
        if (score > best_score) best_score = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break; 
    }
    
    if (legal_moves_played == 0) {
        uint64_t my_king = (color == WHITE) ? b->white_king : b->black_king;
        if (is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) return -INF + (4 - depth); 
        return 0; 
    }
    
    return best_score;
}

uint32_t search_best_move(Board *b, int depth, int color, int use_nnue) {
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
        if (is_square_attacked(b, __builtin_ctzll(my_king), 1 - color)) {
            *b = backup; continue; 
        }
        
        int score = -alpha_beta(b, depth - 1, -beta, -alpha, 1 - color, use_nnue);
        *b = backup;
        
        if (score > best_score) {
            best_score = score;
            best_move = list.moves[i];
        }
        if (score > alpha) alpha = score;
    }
    
    if (best_move == 0 && list.count > 0) best_move = list.moves[0];
    return best_move;
}

// ==========================================
// 10. MAIN (LE TEST ULTIME DU NNUE)
// ==========================================

int main_old() {
    init_leapers_and_masks();

    // On charge le cerveau artificiel !
    if (!load_nnue_weights("nnue_weights.bin")) {
        return 1; // On coupe si le fichier n'est pas là
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

    printf("===================================================\n");
    printf("   MOTEUR CHESS - INTELLIGENCE NEURONALE (NNUE) \n");
    printf("===================================================\n");
    print_board(&board);

    int depth = 5; 
    int total_moves = 40; 

    for (int turn = 0; turn < total_moves; turn++) {
        int color = turn % 2;

        printf("\n--- Tour %d | Trait aux %s ---\n", (turn / 2) + 1, (color == WHITE) ? "Blancs" : "Noirs");

        int game_status = check_game_over(&board, color);
        if (game_status == 1) {
            printf("\n🏆 ECHEC ET MAT !\n"); break;
        } else if (game_status == 2) {
            printf("\n🤝 PAT !\n"); break;
        }

        // L'appel utilise dorénavant l'IA NNUE !
        uint32_t best_move = search_best_move(&board, depth, color, 1);

        printf("Coup IA : ");
        print_move(best_move);
        printf("\n");

        make_move(&board, best_move, color);
        print_board(&board);
    }

    return 0;
}

