import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pkgutil import resolve_name
import time

from PIL import Image, ImageDraw
#스테일메이트, 체크 메이트. 폰 움직임, 기물의 active를 기반으로 처리 여부 결정

files = ['a','b','c','d','e','f','g','h']  # 가로(col)
ranks = ['1','2','3','4','5','6','7','8']  # 세로(row)

board = [
    ['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'],
    ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'],
    ['a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6'],
    ['a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5'],
    ['a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4'],
    ['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3'],
    ['a2', 'b2', 'c2' ,'d2', 'e2', 'f2', 'g2', 'h2'],
    ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1']
]

def square_to_rc(square: str) -> tuple:
    """ 예: 'a2' -> (6, 1) """
    file = square[0]  # 'a' ~ 'h'
    rank = square[1]  # '1' ~ '8'
    col = files.index(file)
    row = ranks.index(rank)
    return row, col

def rc_to_square(row: int, col: int) -> str:
    """ 예: (0, 1) -> 'a2' """
    x = str(files[col] + ranks[row])
    return  x

turn = "white"
turn_count = 1

def turn_change(turn, turn_count):
    if turn == 'black':
        turn = "white"
        return turn, turn_count
    if turn == "white":
        turn = "black"
        turn_count += 1
        return turn, turn_count
    return turn, turn_count

class piece:
    def __init__(self, role, color, pos):
        self.role = role #기물의 역할
        self.color = color #기물의 색
        self.pos = pos #기물의 포지션
        self.active = True #기물 캡쳐 상태 확인
        self.enpassant = False #앙파상 가능 상태를 확인
        self.two_move_turn = None #앙파상 측정용. 움직인 턴을 기록
        self.rook_move_check = False #캐슬링에서 룩의 움직임을 판별
        self.king_move_check = False #캐슬링에서 킹의 움직임 판별
        self.absolute_pin = False
        self.check = False #기물이 체크를 걸는지 확인
        self.protected = False

def move_list(turn):
    if turn == 'white':
        possible_moves = []
        for i in range(len(black_piece_list)):
            if black_piece_list[i].check == True:
                print("black_attack_check!")
                possible_moves.append(when_checked(turn))
                return possible_moves
        for i in range(len(white_piece_list)):
            if white_piece_list[i].role == "pawn":
                possible_moves.append(white_pawn_move(white_piece_list[i]))
            elif white_piece_list[i].role == "knight":
                possible_moves.append(knight_move(white_piece_list[i]))
            elif white_piece_list[i].role == "bishop":
                possible_moves.append(bishop_move(white_piece_list[i]))
            elif white_piece_list[i].role == "rook":
                possible_moves.append(rook_move(white_piece_list[i]))
            elif white_piece_list[i].role == "queen":
                possible_moves.append(queen_move(white_piece_list[i]))
            elif white_piece_list[i].role == "king":
                possible_moves.append(king_move(white_piece_list[i]))
        return possible_moves
    if turn == 'black':
        possible_moves = []
        for i in range(len(white_piece_list)):
            if white_piece_list[i].check == True:
                print("check!")
                possible_moves.append(when_checked(turn))
                return possible_moves
        for i in range(len(black_piece_list)):
            if black_piece_list[i].role == "pawn":
                possible_moves.append(black_pawn_move(black_piece_list[i]))
            elif black_piece_list[i].role == "knight":
                possible_moves.append(knight_move(black_piece_list[i]))
            elif black_piece_list[i].role == "bishop":
                possible_moves.append(bishop_move(black_piece_list[i]))
            elif black_piece_list[i].role == "rook":
                possible_moves.append(rook_move(black_piece_list[i]))
            elif black_piece_list[i].role == "queen":
                possible_moves.append(queen_move(black_piece_list[i]))
            elif black_piece_list[i].role == "king":
                possible_moves.append(king_move(black_piece_list[i]))
        return possible_moves

def move_choice(move1, move2, possible_moves):
    if turn == "white":
        for i in range(len(black_piece_list)):
            if possible_moves[move1][move2] == black_piece_list[i].pos:
                black_piece_list[i].active = False
                return
        white_piece_list[move1].pos = possible_moves[move1][move2]
        if white_piece_list[move1] == white_king:
            white_king.king_move_check = True
        if white_piece_list[move1] == white_rook1:
            white_rook1.rook_move_check = True
        if white_piece_list[move1] == white_rook2:
            white_rook2.rook_move_check = True
        promotion()
        for i in range(len(black_piece_list)):
            black_piece_list[i].check = False
            black_piece_list[i].enpassant = False
        return when_checked(turn), white_king.king_move_check, white_rook1.rook_move_check, white_rook2.rook_move_check
    if turn == "black":
        for i in range(len(white_piece_list)):
            if possible_moves[move1][move2] == white_piece_list[i].pos:
                white_piece_list[i].active = False
                return
        black_piece_list[move1].pos = possible_moves[move1][move2]
        if black_piece_list[move1] == black_king:
            black_king.king_move_check = True
        if black_piece_list[move1] == black_rook1:
            black_rook1.rook_move_check = True
        if black_piece_list[move1] == black_rook2:
            black_rook2.rook_move_check = True
        promotion()
        for i in range(len(white_piece_list)):
            white_piece_list[i].check = False
            white_piece_list[i].enpassant = False
        return when_checked(turn), black_king.king_move_check, black_rook1.rook_move_check, black_rook2.rook_move_check


def flatten_moves(turn):
    pm = move_list(turn)  # 기존 2중 리스트
    flat = []
    for piece_idx, dests in enumerate(pm):
        for dest in range(len(dests)):
            flat.append((piece_idx, dest))
    return flat
'''

'''
def apply_flat_move(flat_move):

    piece_idx = flat_move[0]
    dest = flat_move[1]
    # pm[piece_idx] 안에 dest가 있는지 검증
    return piece_idx, dest

def reset():
    global white_piece_list, white_pawn_list, black_pawn_list, black_piece_list, pieces_list
    global white_king, black_king, white_rook1, white_rook2, black_rook1, black_rook2

    # 1) 기존 참조를 가진 리스트들을 '그 자리에서' 비움
    white_piece_list.clear()
    white_pawn_list.clear()
    black_pawn_list.clear()
    black_piece_list.clear()
    pieces_list.clear()

    white_pawn1 = piece("pawn", "white", "a2")
    white_pawn2 = piece("pawn", "white", "b2")
    white_pawn3 = piece("pawn", "white", "c2")
    white_pawn4 = piece("pawn", "white", "d2")
    white_pawn5 = piece("pawn", "white", "e2")
    white_pawn6 = piece("pawn", "white", "f2")
    white_pawn7 = piece("pawn", "white", "g2")
    white_pawn8 = piece("pawn", "white", "h2")

    white_knight1 = piece("knight", "white", "b1")
    white_knight2 = piece("knight", "white", "g1")

    white_bishop1 = piece("bishop", "white", "c1")
    white_bishop2 = piece("bishop", "white", "f1")

    white_rook1 = piece("rook", "white", "a1")
    white_rook2 = piece("rook", "white", "h1")

    white_king = piece("king", "white", "e1")
    white_queen = piece("queen", "white", "d1")

    black_pawn1 = piece("pawn", "black", "a7")
    black_pawn2 = piece("pawn", "black", "b7")
    black_pawn3 = piece("pawn", "black", "c7")
    black_pawn4 = piece("pawn", "black", "d7")
    black_pawn5 = piece("pawn", "black", "e7")
    black_pawn6 = piece("pawn", "black", "f7")
    black_pawn7 = piece("pawn", "black", "g7")
    black_pawn8 = piece("pawn", "black", "h7")

    black_knight1 = piece("knight", "black", "b8")
    black_knight2 = piece("knight", "black", "g8")

    black_bishop1 = piece("bishop", "black", "c8")
    black_bishop2 = piece("bishop", "black", "f8")

    black_rook1 = piece("rook", "black", "a8")
    black_rook2 = piece("rook", "black", "h8")

    black_king = piece("king", "black", "e8")
    black_queen = piece("queen", "black", "d8")

    white_piece_list = [
        white_pawn1, white_pawn2, white_pawn3, white_pawn4, white_pawn5, white_pawn6, white_pawn7, white_pawn8,
        white_knight1,
        white_knight2, white_bishop1, white_bishop2, white_rook1, white_rook2, white_queen, white_king
    ]

    white_pawn_list = [white_pawn1, white_pawn2, white_pawn3, white_pawn4, white_pawn5, white_pawn6, white_pawn7,
                       white_pawn8]

    black_pawn_list = [black_pawn1, black_pawn2, black_pawn3, black_pawn4, black_pawn5, black_pawn6, black_pawn7,
                       black_pawn8]

    black_piece_list = [
        black_pawn1, black_pawn2, black_pawn3, black_pawn4, black_pawn5, black_pawn6, black_pawn7, black_pawn8,
        black_knight1,
        black_knight2, black_bishop1, black_bishop2, black_rook1, black_rook2, black_queen, black_king
    ]

    pieces_list = [
        white_pawn1, white_pawn2, white_pawn3, white_pawn4, white_pawn5, white_pawn6, white_pawn7, white_pawn8,
        white_knight1,
        white_knight2, white_rook1, white_rook2, white_bishop1, white_bishop2, white_queen, white_king,
        black_pawn1, black_pawn2, black_pawn3, black_pawn4, black_pawn5, black_pawn6, black_pawn7, black_pawn8,
        black_knight1,
        black_knight2, black_bishop1, black_bishop2, black_rook1, black_rook2, black_queen, black_king
    ]
    return white_piece_list, white_pawn_list, black_pawn_list, black_piece_list, pieces_list


def white_pawn_move(piece):
    row, col = square_to_rc(piece.pos)
    possible_moves = []
    if piece.active == False or piece.absolute_pin == True:
        return possible_moves
    if row == 1:
        for i in range(1, 3):
            temp_row = row + i
            square = rc_to_square(temp_row, col)
            if not square_check(square):
                break
            if i == 2:
                possible_moves.append(square)
    temp_row = row + 1
    if in_board(temp_row, col):
        square = rc_to_square(temp_row, col)
        if square_check(square):
            possible_moves.append(square)

    for i in range(-1, 2, 2):
        temp_col = col +  i
        if not in_board(temp_row, temp_col):
            continue
        square = rc_to_square(temp_row, temp_col)
        if color_check(square, piece):
            if square == black_king.pos:
                piece.check = True
                continue
            possible_moves.append(square)
        else:
            for i in range(len(white_piece_list)):
                if white_piece_list[i].pos == square:
                    white_piece_list[i].protected = True

    if row == 5:
        for i in range(-1, 2, 2):
            temp_col = col + i
            if not in_board(temp_row, temp_col):
                continue
            square = rc_to_square(temp_row, temp_col)
            for i in range(len(black_pawn_list)):
                if square == black_pawn_list[i].pos:
                    if black_pawn_list[i].enpassant == True:
                        if in_board(temp_row, temp_col):
                            possible_moves.append(rc_to_square(temp_row, temp_col))
    return possible_moves

def white_pawn_attack(piece):
    row, col = square_to_rc(piece.pos)
    possible_moves = []
    temp_row = row + 1
    for i in range(-1, 2, 2):
        temp_col = col +  i
        if not in_board(temp_row, temp_col):
            continue
        square = rc_to_square(temp_row, temp_col)
        if color_check(square, piece):
            if color_check(square, piece):
                if square == black_king.pos:
                    piece.check = True
                    continue
            possible_moves.append(square)
        else:
            for i in range(len(white_piece_list)):
                if white_piece_list[i].pos == square:
                    white_piece_list[i].protected = True
    return possible_moves

def black_pawn_move(piece):
    row, col = square_to_rc(piece.pos)
    possible_moves = []
    if piece.active == False or piece.absolute_pin == True:
        return possible_moves
    if row == 6:
        for i in range(1, 3):
            temp_row = row - i
            square = rc_to_square(temp_row, col)
            if not square_check(square):
                break
            if i == 2:
                possible_moves.append(square)
    temp_row = row - 1
    if in_board(temp_row, col):
        square = rc_to_square(temp_row, col)
        if square_check(square):
            possible_moves.append(square)

    for i in range(-1, 2, 2):
        temp_col = col - i
        if not in_board(temp_row, temp_col):
            continue
        square = rc_to_square(temp_row, temp_col)
        if color_check(square, piece):
            if color_check(square, piece):
                if square == white_king.pos:
                    piece.check = True
                    continue
            possible_moves.append(square)
        else:
            for i in range(len(black_piece_list)):
                if black_piece_list[i].pos == square:
                    black_piece_list[i].protected = True

    if row == 4:
        for i in range(-1, 2, 2):
            temp_col = col - i
            if not in_board(temp_row, temp_col):
                continue
            square = rc_to_square(temp_row, temp_col)
            for i in range(len(white_pawn_list)):
                if square == white_pawn_list[i].pos:
                    if white_pawn_list[i].enpassant == True:
                        if in_board(temp_row, temp_col):
                            possible_moves.append(rc_to_square(temp_row, temp_col))
    return possible_moves

def black_pawn_attack(piece):
    row, col = square_to_rc(piece.pos)
    possible_moves = []
    temp_row = row - 1
    for i in range(-1, 2, 2):
        temp_col = col +  i
        if not in_board(temp_row, temp_col):
            continue
        square = rc_to_square(temp_row, temp_col)
        if color_check(square, piece):
            if color_check(square, piece):
                if square == white_king.pos:
                    piece.check = True
                    continue
            possible_moves.append(square)
        else:
            for i in range(len(white_piece_list)):
                if white_piece_list[i].pos == square:
                    white_piece_list[i].protected = True
    return possible_moves

def promotion():
    if turn == "white":
        for i in range(len(white_pawn_list)):
            row, col = square_to_rc(white_pawn_list[i].pos)
            if  row == 7:
                promotion_list = ["knight", "bishop", "rook", "queen"]
                return promotion_list
    else:
        for i in range(len(black_pawn_list)):
            row, col = square_to_rc(black_pawn_list[i].pos)
            if row == 0:
                promotion_list = ["knight", "bishop", "rook", "queen"]
                return promotion_list

def pawn_two_move(piece, square):
    if piece.role == "pawn":
        row, col = square_to_rc(piece.pos)
        target_row, target_col = square_to_rc(square)
        if abs(row - target_row) == 2:
            piece.enpassant = True

def knight_move(piece):
    row, col = square_to_rc(piece.pos)
    possible_moves = []
    if piece.active == False:
        return possible_moves
    if piece.absolute_pin == True:
        return possible_moves
    for i in range(-1, 2, 2): #up calculation
        temp_row = row + 2
        temp_col = col + i
        if not in_board(temp_row, temp_col):
            continue
        square = rc_to_square(temp_row, temp_col)
        if square_check(square):
            possible_moves.append(square)
            continue
        if color_check(square, piece):
            if define_color_in_def(piece) == True:
                if square == white_king.pos:
                    piece.check = True
                    continue
            else:
                if square == black_king.pos:
                    piece.check = True
                    continue
            possible_moves.append(square)
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if square == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if square == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        continue

    for i in range(-1, 2, 2):
        temp_row = row - 2
        temp_col = col + i
        if not in_board(temp_row, temp_col):
            continue
        square = rc_to_square(temp_row, temp_col)
        if square_check(square):
            possible_moves.append(square)
            continue
        if color_check(square, piece):
            if define_color_in_def(piece) == True:
                if square == white_king.pos:
                    piece.check = True
                    continue
            else:
                if square == black_king.pos:
                    piece.check = True
                    continue
            possible_moves.append(square)
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if square == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if square == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        continue
    for i in range(-1, 2, 2):
        temp_col = col + 2
        temp_row = row + i
        if not in_board(temp_row, temp_col):
            continue
        square = rc_to_square(temp_row, temp_col)
        if square_check(square):
            possible_moves.append(square)
            continue
        if color_check(square, piece):
            if define_color_in_def(piece) == True:
                if square == white_king.pos:
                    piece.check = True
                    continue
            else:
                if square == black_king.pos:
                    piece.check = True
                    continue
            possible_moves.append(square)
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if square == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if square == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        continue
    for i in range(-1, 2, 2):
        temp_col = col - 2
        temp_row = row + i
        if not in_board(temp_row, temp_col):
            continue
        square = rc_to_square(temp_row, temp_col)
        if square_check(square):
            possible_moves.append(square)
            continue
        if color_check(square, piece):
            if define_color_in_def(piece) == True:
                if square == white_king.pos:
                    piece.check = True
                    continue
            else:
                if square == black_king.pos:
                    piece.check = True
                    continue
            possible_moves.append(square)
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if square == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if square == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        continue
    return possible_moves

def rook_move(piece):
    row, col = square_to_rc(piece.pos)
    possible_moves = []

    if not piece.active or piece.absolute_pin:
        return possible_moves

    # → (row, col+i)
    for i in range(1, 8):
        r, c = row, col + i
        if not in_board(r, c):
            break
        sq = rc_to_square(r, c)

        if square_check(sq):
            possible_moves.append(sq)
            continue

        if color_check(sq, piece):
            if define_color_in_def(piece) == True:
                if sq == white_king.pos:
                    piece.check = True
                    break
            else:
                if sq == black_king.pos:
                    piece.check = True
                    break
            possible_moves.append(sq)
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if sq == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if sq == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        break

    # ← (row, col-i)
    for i in range(1, 8):
        r, c = row, col - i
        if not in_board(r, c):
            break
        sq = rc_to_square(r, c)

        if square_check(sq):
            possible_moves.append(sq)
            continue

        if color_check(sq, piece):
            if define_color_in_def(piece) == True:
                if sq == white_king.pos:
                    piece.check = True
                    break
            else:
                if sq == black_king.pos:
                    piece.check = True
                    break
            possible_moves.append(sq)
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if sq == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if sq == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        break

    # ↑ (row+i, col)
    for i in range(1, 8):
        r, c = row + i, col
        if not in_board(r, c):
            break
        sq = rc_to_square(r, c)

        if square_check(sq):
            possible_moves.append(sq)
            continue

        if color_check(sq, piece):
            if define_color_in_def(piece) == True:
                if sq == white_king.pos:
                    piece.check = True
                    break
            else:
                if sq == black_king.pos:
                    piece.check = True
                    break
            possible_moves.append(sq)
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if sq == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if sq == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        break

    # ↓ (row-i, col)
    for i in range(1, 8):
        r, c = row - i, col
        if not in_board(r, c):
            break
        sq = rc_to_square(r, c)

        if square_check(sq):
            possible_moves.append(sq)
            continue

        if color_check(sq, piece):
            if define_color_in_def(piece) == True:
                if sq == white_king.pos:
                    piece.check = True
                    break
            else:
                if sq == black_king.pos:
                    piece.check = True
                    break
            possible_moves.append(sq)
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if sq == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if sq == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        break

    return possible_moves

def bishop_move(piece):
    row, col = square_to_rc(piece.pos)
    positive_column = 8 - col
    positive_row = 8 - row
    possible_moves = []
    if not piece.active or piece.absolute_pin:
        return possible_moves
    for i in range(1, positive_column):  # 이동은 이등변 삼각형으로 이루어지기 때문에 기준은 하나로 충분
        if not in_board(row + i, col + i):
            break
        square = rc_to_square(row + i, col + i)
        if not in_board(row + i, col + i):
            break
        if square_check(rc_to_square(row + i, col + i)):  # 우 상향 방향 이동 계산. 막히지 않고 갈 수 있는 거리
            possible_moves.append(rc_to_square(row + i, col + i))
            continue

        if color_check(rc_to_square(row + i, col + i), piece):
            if define_color_in_def(piece) == True:
                if rc_to_square(row + i, col + i) == white_king.pos:
                    piece.check = True
                    break
            else:
                if rc_to_square(row + i, col + i) == black_king.pos:
                    piece.check = True
                    break
            possible_moves.append(rc_to_square(row + i, col + i))
            break
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if square == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if square == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        break

    for i in range(1, col):
        if not in_board(row - i, col - i):
            break
        if square_check(rc_to_square(row - i, col - i)):  # 좌 하향 방향 이동 계산. 막히지 않고 갈 수 있는 거리
            possible_moves.append(rc_to_square(row - i, col - i))
            continue
        square = rc_to_square(row - i, col - i)
        if color_check(rc_to_square(row - i, col - i), piece):
            if define_color_in_def(piece) == True:
                if rc_to_square(row - i, col - i) == white_king.pos:
                    piece.check = True
                    break
            else:
                if rc_to_square(row - i, col - i) == black_king.pos:
                    piece.check = True
                    break
            possible_moves.append(rc_to_square(row - i, col - i))
            break
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if square == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if square == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        break

    for i in range(1, positive_row):
        if not in_board(row + i, col - i):
            break
        if square_check(rc_to_square(row + i, col - i)):  # 좌 상향 방향 이동 계산. 막히지 않고 갈 수 있는 거리
            possible_moves.append(rc_to_square(row + i, col - i))
            continue
        square = rc_to_square(row + i, col - i)
        if color_check(rc_to_square(row + i, col - i), piece):
            if define_color_in_def(piece) == True:
                if rc_to_square(row + i, col - i) == white_king.pos:
                    piece.check = True
                    break
            else:
                if rc_to_square(row + i, col - i) == black_king.pos:
                    piece.check = True
                    break
            possible_moves.append(rc_to_square(row + i, col - i))
            break
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if square == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if square == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        break

    for i in range(1, row):
        if not in_board(row - i, col + i):
            break
        if square_check(rc_to_square(row - i, col + i)):  # 우 하향 방향 이동 계산. 막히지 않고 갈 수 있는 거리
            possible_moves.append(rc_to_square(row - i, col + i))
            continue
        square = rc_to_square(row - i, col + i)
        if color_check(rc_to_square(row - i, col + i), piece):
            if define_color_in_def(piece) == True:
                if rc_to_square(row - i, col + i) == white_king.pos:
                    piece.check = True
                    break
            else:
                if rc_to_square(row - i, col + i) == black_king.pos:
                    piece.check = True
                    break
            possible_moves.append(rc_to_square(row - i, col + i))
            break
        else:
            if define_color_in_def(piece) == True:
                for i in range(len(white_piece_list)):
                    if square == white_piece_list[i].pos:
                        white_piece_list[i].protected = True
            else:
                for i in range(len(black_piece_list)):
                    if square == black_piece_list[i].pos:
                        black_piece_list[i].protected = True
        break

    return possible_moves

def queen_move(piece):
    row, col = square_to_rc(piece.pos)
    possible_moves = []

    if not piece.active or piece.absolute_pin:
        return possible_moves
    vertical_move = rook_move(piece)
    diagonal_move = bishop_move(piece)
    possible_moves.extend(vertical_move)
    possible_moves.extend(diagonal_move)
    return possible_moves


#아군 기물이 있는지 체크 안 함. 잡을 수 있는지 체크 안 함.
def king_move(piece):
    row, col = square_to_rc(piece.pos)
    possible_moves = []
    if piece.color == "white":
        if white_short_castle(piece, white_rook1):
            possible_moves.append(rc_to_square(row, col + 2))
        if white_long_castle(piece, white_rook2):
            possible_moves.append(rc_to_square(row, col - 2))
    if piece.color == "black":
        if black_short_castle(piece, black_rook1):
            possible_moves.append(rc_to_square(row, col + 2))
        if black_long_castle(piece, black_rook2):
            possible_moves.append(rc_to_square(row, col - 2))
    for i in range(-1, 2, 2):
        temp_row = row + 1
        if not in_board(temp_row, col + i):
            continue
        target_square = rc_to_square(temp_row, col + i)
        if square_check(target_square):
            possible_moves.append(target_square)
            continue
        color = piece.color
        if color_check(target_square, piece):
            if color == "white":
                for i in range(len(black_piece_list)):
                    if target_square == black_piece_list[i].pos:
                        if black_piece_list[i].protected == True:
                            continue
                        else:
                            possible_moves.append(target_square)
            else:
                for i in range(len(white_piece_list)):
                    if target_square == white_piece_list[i].pos:
                        if white_piece_list[i].protected == True:
                            continue
                        else:
                            possible_moves.append(target_square)
    for i in range(-1, 2, 2):
        temp_row = row - 1
        if not in_board(temp_row, col + i):
            continue
        target_square = rc_to_square(temp_row, col + i)
        if square_check(target_square):
            possible_moves.append(target_square)
            continue
        color = piece.color
        if color_check(target_square, piece):
            if color == "white":
                for i in range(len(black_piece_list)):
                    if target_square == black_piece_list[i].pos:
                        if black_piece_list[i].protected == True:
                            continue
                        else:
                            possible_moves.append(target_square)
            else:
                for i in range(len(white_piece_list)):
                    if target_square == white_piece_list[i].pos:
                        if white_piece_list[i].protected == True:
                            continue
                        else:
                            possible_moves.append(target_square)
    for i in range(-1, 2, 2):
        if not in_board(row, col + i):
            continue
        target_square = rc_to_square(row, col + i)
        if square_check(target_square):
            possible_moves.append(target_square)
            continue
        color = piece.color
        if color_check(target_square, piece):
            if color == "white":
                for i in range(len(black_piece_list)):
                    if target_square == black_piece_list[i].pos:
                        if black_piece_list[i].protected == True:
                            continue
                        else:
                            possible_moves.append(target_square)
            else:
                for i in range(len(white_piece_list)):
                    if target_square == white_piece_list[i].pos:
                        if white_piece_list[i].protected == True:
                            continue
                        else:
                            possible_moves.append(target_square)
    for i in range(-1, 2, 2):
        if not in_board(row + i, col):
            continue
        target_square = rc_to_square(row + i, col)
        if square_check(target_square):
            possible_moves.append(target_square)
            continue
        color = piece.color
        if color_check(target_square, piece):
            if color == "white":
                for i in range(len(black_piece_list)):
                    if target_square == black_piece_list[i].pos:
                        if black_piece_list[i].protected == True:
                            continue
                        else:
                            possible_moves.append(target_square)
            else:
                for i in range(len(white_piece_list)):
                    if target_square == white_piece_list[i].pos:
                        if white_piece_list[i].protected == True:
                            continue
                        else:
                            possible_moves.append(target_square)
    attack_square = []
    if define_color_in_def(piece):
        for i in range(len(black_piece_list)):
            if black_piece_list[i].role == "pawn":
                attack_square.extend(black_pawn_attack(black_piece_list[i]))
            if black_piece_list[i].role == "knight":
                attack_square.extend(knight_move(black_piece_list[i]))
            if black_piece_list[i].role == "bishop":
                attack_square.extend(bishop_move(black_piece_list[i]))
            if black_piece_list[i].role == "rook":
                attack_square.extend(rook_move(black_piece_list[i]))
            if black_piece_list[i].role == "queen":
                attack_square.extend(queen_move(black_piece_list[i]))
            if black_piece_list[i].role == "king":
                attack_square.extend(king_attack_square(black_piece_list[i]))
        for m in possible_moves[:]:  # 복사본으로 순회
            if m in attack_square:
                possible_moves.remove(m)

    else:
        for i in range(len(white_piece_list)):
            if white_piece_list[i].role == "pawn":
                attack_square.extend(white_pawn_attack(white_piece_list[i]))
            if white_piece_list[i].role == "knight":
                attack_square.extend(knight_move(white_piece_list[i]))
            if white_piece_list[i].role == "bishop":
                attack_square.extend(bishop_move(white_piece_list[i]))
            if white_piece_list[i].role == "rook":
                attack_square.extend(rook_move(white_piece_list[i]))
            if white_piece_list[i].role == "queen":
                attack_square.extend(queen_move(white_piece_list[i]))
            if white_piece_list[i].role == "king":
                attack_square.extend(king_attack_square(white_piece_list[i]))
        for m in possible_moves[:]:  # 복사본으로 순회
            if m in attack_square:
                possible_moves.remove(m)
    return possible_moves

def king_attack_square(piece):
    row, col = square_to_rc(piece.pos)
    attack_list = []
    for i in range(-1, 2, 2):
        temp_row = row + 1
        if not in_board(temp_row, col + i):
            continue
        target_square = rc_to_square(temp_row, col + i)
        attack_list.append(target_square)
    for i in range(-1, 2, 2):
        temp_row = row - 1
        if not in_board(temp_row, col + i):
            continue
        target_square = rc_to_square(temp_row, col + i)
        attack_list.append(target_square)
    for i in range(-1, 2, 2):
        if not in_board(row, col + i):
            continue
        target_square = rc_to_square(row, col + i)
        attack_list.append(target_square)
    for i in range(-1, 2, 2):
        if not in_board(row + i, col):
            continue
        target_square = rc_to_square(row + i, col)
        attack_list.append(target_square)
    return attack_list

def white_short_castle(piece, piece2):
    if piece.role == "king" and piece2.role == "rook":
        if piece.king_move_check == False and piece2.rook_move_check == False\
                and square_check('f1') and square_check('g1'): #기본 조건인 킹 움직임과 룩 움직임 그리고 중간 칸이 비었는지 여부 확인
            # 공격 하는 기물이 있는지 파악. 칸이 공격당하는 중이면 거짓을 반환
            for i in pieces_list:
                if i.color == "white":
                    if i.role == "pawn":
                        white_pawn_attack_list = white_pawn_attack(i)
                        for p in white_pawn_attack_list:
                            if p == 'f1' or p == 'g1':
                                return False
                    if i.role == "knight":
                        knight_attack_list = knight_move(i)
                        for p in knight_attack_list:
                            if p == 'f1' or p == 'g1':
                                return False
                    if i.role == "bishop":
                        bishop_attack_list = bishop_move(i)
                        for p in bishop_attack_list:
                            if p == 'f1' or p == 'g1':
                                return False
                    if i.role == "rook":
                        rook_attack_list = rook_move(i)
                        for p in rook_attack_list:
                            if p == 'f1' or p == 'g1':
                                return False
                    if i.role == "queen":
                        queen_attack_list = queen_move(i)
                        for p in queen_attack_list:
                            if p == 'f1' or p == 'g1':
                                return False
                return True
            return True
        return False
    return False

def white_long_castle(piece, piece2):
    if piece.role == "king" and piece2.role == "rook":
        if piece.king_move_check == False and piece2.rook_move_check == False\
                and square_check('c1') and square_check('d1'):
            for i in pieces_list:
                if i.color == "white":
                    if i.role == "pawn":
                        white_pawn_attack_list = white_pawn_attack(i)
                        for p in white_pawn_attack_list:
                            if p == 'c1' or p == 'd1':
                                return False
                    if i.role == "knight":
                        knight_attack_list = knight_move(i)
                        for p in knight_attack_list:
                            if p == 'c1' or p == 'd1':
                                return False
                    if i.role == "bishop":
                        bishop_attack_list = bishop_move(i)
                        for p in bishop_attack_list:
                            if p == 'c1' or p == 'd1':
                                return False
                    if i.role == "rook":
                        rook_attack_list = rook_move(i)
                        for p in rook_attack_list:
                            if p == 'c1' or p == 'd1':
                                return False
                    if i.role == "queen":
                        queen_attack_list = queen_move(i)
                        for p in queen_attack_list:
                            if p == 'c1' or p == 'd1':
                                return False
                return True
            return True
        return False
    return False

def black_short_castle(piece, piece2):
    if piece.role == "king" and piece2.role == "rook":
        if piece.king_move_check == False and piece2.rook_move_check == False\
                and square_check('f8') and square_check('g8'):
            for i in pieces_list:
                if i.color == "white":
                    if i.role == "pawn":
                        white_pawn_attack_list = white_pawn_attack(i)
                        for p in white_pawn_attack_list:
                            if p == 'f8' or p == 'g8':
                                return False
                    if i.role == "knight":
                        knight_attack_list = knight_move(i)
                        for p in knight_attack_list:
                            if p == 'f8' or p == 'g8':
                                return False
                    if i.role == "bishop":
                        bishop_attack_list = bishop_move(i)
                        for p in bishop_attack_list:
                            if p == 'f8' or p == 'g8':
                                return False
                    if i.role == "rook":
                        rook_attack_list = rook_move(i)
                        for p in rook_attack_list:
                            if p == 'f8' or p == 'g8':
                                return False
                    if i.role == "queen":
                        queen_attack_list = queen_move(i)
                        for p in queen_attack_list:
                            if p == 'f8' or p == 'g8':
                                return False
                return True
            return True
        return False
    return False

def black_long_castle(piece, piece2):
    if piece.role == "king" and piece2.role == "rook":
        if piece.king_move_check == False and piece2.rook_move_check == False\
                and square_check('c8') and square_check('d8'): #기본 조건인 킹 움직임과 룩 움직임 그리고 중간 칸이 비었는지 여부 확인
            #공격 하는 기물이 있는지 파악. 칸이 공격당하는 중이면 거짓을 반환
            for i in pieces_list:
                if i.color == "white":
                    if i.role == "pawn":
                        white_pawn_attack_list = white_pawn_attack(i)
                        for p in white_pawn_attack_list:
                            if p == 'c8' or p == 'd8':
                                return False
                    if i.role == "knight":
                        knight_attack_list = knight_move(i)
                        for p in knight_attack_list:
                            if p == 'c8' or p == 'd8':
                                return False
                    if i.role == "bishop":
                        bishop_attack_list = bishop_move(i)
                        for p in bishop_attack_list:
                            if p == 'c8' or p == 'd8':
                                return False
                    if i.role == "rook":
                        rook_attack_list = rook_move(i)
                        for p in rook_attack_list:
                            if p == 'c8' or p == 'd8':
                                return False
                    if i.role == "queen":
                        queen_attack_list = queen_move(i)
                        for p in queen_attack_list:
                            if p == 'c8' or p == 'd8':
                                return False
                return True
            return True
        return False
    return False

def diagonal_pin(piece):
    row, col = square_to_rc(piece.pos)
    positive_row = 8 - row
    positive_col = 8 - col
    min_val = min(positive_row, positive_col) #계산해야하는 사각형의 넓이 지정
    temp_piece_list = []
    if min_val >= 2: #두칸 이상 남았을 경우만 계산
        for i in range(min_val):
            temp_row = row + i
            temp_col = col + i
            temp_square = rc_to_square(temp_row, temp_col)
            if square_check(temp_square) == False: #기물의 존재 여부 확인
                for p in pieces_list:
                    if p.pos == temp_square:
                        temp_piece_list.append(p) #기물 불러오기
            if len(temp_piece_list) == 2: #2개 수집했을 경우 중지
                break
        if len(temp_piece_list) == 2:
            piece_1 = temp_piece_list[0]
            piece_2 = temp_piece_list[1]
            if piece_1.color == piece.color and piece_2.color != piece.color: # 두 번째가 공격 기물, 첫 번째가 수비 기물
                if piece_2.role == "Queen" or piece_2.role == "bishop": #역할 확인
                    piece_1.absolute_pin = True #절대핀 활성화
    min_val = min(positive_row, col)  # 계산해야하는 사각형의 넓이 지정
    temp_piece_list = []
    if min_val >= 2:
        for i in range(min_val):
            temp_row = row + i
            temp_col = col - i
            temp_square = rc_to_square(temp_row, temp_col)
            if square_check(temp_square) == False:
                for p in pieces_list:
                    if p.pos == temp_square:
                        temp_piece_list.append(p)
            if len(temp_piece_list) == 2:
                break
        if len(temp_piece_list) == 2:
            piece_1 = temp_piece_list[0]
            piece_2 = temp_piece_list[1]
            if piece_1.color == piece.color and piece_2.color != piece.color:
                if piece_2.role == "Queen" or piece_2.role == "bishop":
                    piece_1.absolute_pin = True
    min_val = min(row, col)  # 계산해야하는 사각형의 넓이 지정
    temp_piece_list = []
    if min_val >= 2:
        for i in range(min_val):
            temp_row = row - i
            temp_col = col - i
            temp_square = rc_to_square(temp_row, temp_col)
            if square_check(temp_square) == False:
                for p in pieces_list:
                    if p.pos == temp_square:
                        temp_piece_list.append(p)
            if len(temp_piece_list) == 2:
                break
        if len(temp_piece_list) == 2:
            piece_1 = temp_piece_list[0]
            piece_2 = temp_piece_list[1]
            if piece_1.color == piece.color and piece_2.color != piece.color:
                if piece_2.role == "Queen" or piece_2.role == "bishop":
                    piece_1.absolute_pin = True
    min_val = min(row, positive_col)  # 계산해야하는 사각형의 넓이 지정
    temp_piece_list = []
    if min_val >= 2:
        for i in range(min_val):
            temp_row = row - i
            temp_col = col + i
            temp_square = rc_to_square(temp_row, temp_col)
            if square_check(temp_square) == False:
                for p in pieces_list:
                    if p.pos == temp_square:
                        temp_piece_list.append(p)
            if len(temp_piece_list) == 2:
                break
        if len(temp_piece_list) == 2:
            piece_1 = temp_piece_list[0]
            piece_2 = temp_piece_list[1]
            if piece_1.color == piece.color and piece_2.color != piece.color:
                if piece_2.role == "Queen" or piece_2.role == "bishop":
                    piece_1.absolute_pin = True

def vertical_pin(piece):
    row, col = square_to_rc(piece.pos)
    positive_row = 8 - row
    positive_col = 8 - col
    temp_piece_list = []
    if positive_row >= 2:
        for i in range(positive_row):
            temp_row = row + i
            temp_square = rc_to_square(temp_row, col)
            if square_check(temp_square) == False:
                for p in pieces_list:
                    if p.pos == temp_square:
                        temp_piece_list.append(p)
            if len(temp_piece_list) == 2:
                break
        if len(temp_piece_list) == 2:
            piece_1 = temp_piece_list[0]
            piece_2 = temp_piece_list[1]
            if piece_1.color == piece.color and piece_2.color != piece.color:
                if piece_2.role == "Queen" or piece_2.role == "rook":
                    piece_1.absolute_pin = True
    temp_piece_list = []
    if col >= 2:
        for i in range(col):
            temp_col = col - i
            temp_square = rc_to_square(row, temp_col)
            if square_check(temp_square) == False:
                for p in pieces_list:
                    if p.pos == temp_square:
                        temp_piece_list.append(p)
            if len(temp_piece_list) == 2:
                break
        if len(temp_piece_list) == 2:
            piece_1 = temp_piece_list[0]
            piece_2 = temp_piece_list[1]
            if piece_1.color == piece.color and piece_2.color != piece.color:
                if piece_2.role == "Queen" or piece_2.role == "rook":
                    piece_1.absolute_pin = True
    temp_piece_list = []
    if row >= 2:
        for i in range(row):
            temp_row = row - i
            temp_square = rc_to_square(temp_row, col)
            if square_check(temp_square) == False:
                for p in pieces_list:
                    if p.pos == temp_square:
                        temp_piece_list.append(p)
                if len(temp_piece_list) == 2:
                    break
        if len(temp_piece_list) == 2:
            piece_1 = temp_piece_list[0]
            piece_2 = temp_piece_list[1]
            if piece_1.color == piece.color and piece_2.color != piece.color:
                if piece_2.role == "Queen" or piece_2.role == "rook":
                    piece_1.absolute_pin = True
    temp_piece_list = []
    if positive_col >= 2:
        for i in range(positive_col):
            temp_col = col + i
            temp_square = rc_to_square(row, temp_col)
            if square_check(temp_square) == False:
                for p in pieces_list:
                    if p.pos == temp_square:
                        temp_piece_list.append(p)
            if len(temp_piece_list) == 2:
                break
        if len(temp_piece_list) == 2:
            piece_1 = temp_piece_list[0]
            piece_2 = temp_piece_list[1]
            if piece_1.color == piece.color and piece_2.color != piece.color:
                if piece_2.role == "Queen" or piece_2.role == "rook":
                    piece_1.absolute_pin = True

def square_check(square: str) -> bool:
    for p in pieces_list:
        if p.pos == square and p.active == True:
            return False
    return True

def color_check(square: str, piece) -> bool:
    for p in pieces_list:
        if p.pos == square and p.color != piece.color and p.active == True:
            return True
    return  False

def define_color_in_def(piece):
    if piece.color == "white":
        return True
    else:
        return False

def in_board(row: int, col: int) -> bool:
    return 0 <= row < 8 and 0 <= col < 8

def CheckMate(possible_moves):
    total = sum(len(lst) for lst in possible_moves)
    if total == 0:
        if turn == "white":
            for i in range(len(black_piece_list)):
                if black_piece_list[i].check == True:
                    reward = +1
                    return "black", "checkmate", reward
                else:
                    reward = 0
                    return "black", "stalemate", reward
        if turn == "black":
            for i in range(len(white_piece_list)):
                if white_piece_list[i].check == True:
                    reward = +1
                    return "white", "checkmate", reward
                else:
                    reward = 0
                    return "white", "stalemate", reward
    else:
        if turn == "white":
            return "white", False, 0
        else:
            return "black", False, 0

def when_checked(turn):
    check_pieces = []
    possible_moves = []
    attack_list = []
    if turn == "white":
        for i in range(len(black_piece_list)):
            if black_piece_list[i].check == True:
                check_pieces.append(black_piece_list[i])
                if len(check_pieces) > 1:
                    possible_moves.append(king_move(white_king))
                    return possible_moves
                if black_piece_list[i].role == "pawn":
                    attack_list.append(black_piece_list[i].pos)
                    possible_moves.append(check_move(attack_list, "white"))
                    return possible_moves
                if black_piece_list[i].role == "knight":
                    attack_list.append(black_piece_list[i].pos)
                    possible_moves.append(check_move(attack_list, "white"))
                    return possible_moves
                if black_piece_list[i].role == "rook":
                    row, col = square_to_rc(black_piece_list[i].pos)
                    target_row, target_col = square_to_rc(white_king.pos)
                    cal_row = row - target_row
                    cal_col = col - target_col
                    if cal_row != 0:
                        if cal_row < 0:
                            for i in range(abs(cal_row) + 1):
                                row += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                        elif cal_row > 0:
                            for i in range(abs(cal_row) + 1):
                                row -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                    elif cal_col != 0:
                        if cal_col < 0:
                            for i in range(abs(cal_col) + 1):
                                col += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                        elif cal_col > 0:
                            for i in range(abs(cal_col) + 1):
                                col -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                        return possible_moves
                if black_piece_list[i].role == "bishop":
                    row, col = square_to_rc(black_piece_list[i].pos)
                    target_row, target_col = square_to_rc(white_king.pos)
                    cal_row = row - target_row
                    cal_col = col - target_col
                    if cal_row < 0 and cal_col < 0:
                        for i in range(abs(cal_row) + 1):
                            row += i
                            col += i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "white"))
                    elif cal_row > 0 and cal_col < 0:
                        for i in range(abs(cal_row) + 1):
                            row -= i
                            col += i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "white"))
                    elif cal_row > 0 and cal_col > 0:
                        for i in range(abs(cal_row) + 1):
                            row -= i
                            col -= i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "white"))
                    elif cal_row < 0 and cal_col > 0:
                        for i in range(abs(cal_row) + 1):
                            row += i
                            col -= i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "white"))
                    return possible_moves
                if black_piece_list[i].role == "queen":
                    row, col = square_to_rc(black_piece_list[i].pos)
                    target_row, target_col = square_to_rc(white_king.pos)
                    cal_row = row - target_row
                    cal_col = col - target_col
                    if cal_row < 0 and cal_col < 0:
                        for i in range(abs(cal_row) + 1):
                            row += i
                            col += i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "white"))
                    elif cal_row > 0 and cal_col < 0:
                        for i in range(abs(cal_row) + 1):
                            row -= i
                            col += i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "white"))
                    elif cal_row > 0 and cal_col > 0:
                        for i in range(abs(cal_row) + 1):
                            row -= i
                            col -= i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "white"))
                    elif cal_row < 0 and cal_col > 0:
                        for i in range(abs(cal_row) + 1):
                            row += i
                            col -= i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "white"))
                    if cal_row != 0:
                        if cal_row < 0:
                            for i in range(abs(cal_row) + 1):
                                row += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                        if cal_row > 0:
                            for i in range(abs(cal_row) + 1):
                                row -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                    elif cal_col != 0:
                        if cal_col < 0:
                            for i in range(abs(cal_col) + 1):
                                col += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                        if cal_col > 0:
                            for i in range(abs(cal_col) + 1):
                                col -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                    elif cal_row != 0:
                        if cal_row < 0:
                            for i in range(abs(cal_row) + 1):
                                row += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                        elif cal_row > 0:
                            for i in range(abs(cal_row) + 1):
                                row -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "white"))
                    return possible_moves
                return possible_moves
            return possible_moves
        return possible_moves
    if turn == "black":
        for i in range(len(white_piece_list)):
            if white_piece_list[i].check == True:
                check_pieces.append(white_piece_list[i])
                if len(check_pieces) > 1:
                    possible_moves.append(king_move(black_king))
                    return possible_moves
                if white_piece_list[i].role == "pawn":
                    attack_list.append(white_piece_list[i].pos)
                    possible_moves.append(check_move(attack_list, "black"))
                    return possible_moves
                if white_piece_list[i].role == "knight":
                    attack_list.append(white_piece_list[i].pos)
                    possible_moves.append(check_move(attack_list, "black"))
                    return possible_moves
                if white_piece_list[i].role == "rook":
                    row, col = square_to_rc(white_piece_list[i].pos)
                    target_row, target_col = square_to_rc(black_king.pos)
                    cal_row = row - target_row
                    cal_col = col - target_col
                    if cal_row != 0:
                        if cal_row < 0:
                            for i in range(abs(cal_row) + 1):
                                row += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                        elif cal_row > 0:
                            for i in range(abs(cal_row) + 1):
                                row -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                    elif cal_col != 0:
                        if cal_col < 0:
                            for i in range(abs(cal_col) + 1):
                                col += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                        elif cal_col > 0:
                            for i in range(abs(cal_col) + 1):
                                col -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                    return possible_moves
                if white_piece_list[i].role == "bishop":
                    row, col = square_to_rc(white_piece_list[i].pos)
                    target_row, target_col = square_to_rc(black_king.pos)
                    cal_row = row - target_row
                    cal_col = col - target_col
                    if cal_row < 0 and cal_col < 0:
                        for i in range(abs(cal_row) + 1):
                            row += i
                            col += i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "black"))
                    elif cal_row > 0 and cal_col < 0:
                        for i in range(abs(cal_row) + 1):
                            row -= i
                            col += i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "black"))
                    elif cal_row > 0 and cal_col > 0:
                        for i in range(abs(cal_row) + 1):
                            row -= i
                            col -= i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "black"))
                    elif cal_row < 0 and cal_col > 0:
                        for i in range(abs(cal_row) + 1):
                            row += i
                            col -= i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "black"))
                    return possible_moves
                if white_piece_list[i].role == "queen":
                    row, col = square_to_rc(white_piece_list[i])
                    target_row, target_col = square_to_rc(black_king.pos)
                    cal_row = row - target_row
                    cal_col = col - target_col
                    if cal_row < 0 and cal_col < 0:
                        for i in range(abs(cal_row) + 1):
                            row += i
                            col += i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "black"))
                    elif cal_row > 0 and cal_col < 0:
                        for i in range(abs(cal_row) + 1):
                            row -= i
                            col += i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "black"))
                    elif cal_row > 0 and cal_col > 0:
                        for i in range(abs(cal_row) + 1):
                            row -= i
                            col -= i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "black"))
                    elif cal_row < 0 and cal_col > 0:
                        for i in range(abs(cal_row) + 1):
                            row += i
                            col -= i
                            if not in_board(row, col):
                                break
                            square = rc_to_square(row, col)
                            attack_list.append(square)
                            possible_moves.append(check_move(attack_list, "black"))
                    if cal_row != 0:
                        if cal_row < 0:
                            for i in range(abs(cal_row) + 1):
                                row += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                        if cal_row > 0:
                            for i in range(abs(cal_row) + 1):
                                row -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                    elif cal_col != 0:
                        if cal_col < 0:
                            for i in range(abs(cal_col) + 1):
                                col += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                        if cal_col > 0:
                            for i in range(abs(cal_col) + 1):
                                col -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                    elif cal_row != 0:
                        if cal_row < 0:
                            for i in range(abs(cal_row) + 1):
                                row += i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                        elif cal_row > 0:
                            for i in range(abs(cal_row) + 1):
                                row -= i
                                if not in_board(row, col):
                                    break
                                square = rc_to_square(row, col)
                                attack_list.append(square)
                                possible_moves.append(check_move(attack_list, "black"))
                    return possible_moves
                return possible_moves
            return possible_moves
        return possible_moves

def check_move(attack_list, color):
    possible_moves = []
    if color == "white":
        for i in range(len(white_piece_list)):
            if white_piece_list[i].role == "pawn":
                piece_move_list = white_pawn_move(white_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if white_piece_list[i].role == "knight":
                piece_move_list = knight_move(white_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if white_piece_list[i].role == "bishop":
                piece_move_list = bishop_move(white_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if white_piece_list[i].role == "rook":
                piece_move_list = rook_move(white_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if white_piece_list[i].role == "queen":
                piece_move_list = queen_move(white_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if white_piece_list[i].role == "king":
                possible_moves.append(king_move(white_piece_list[i]))
        return possible_moves
    if color == "black":
        for i in range(len(black_piece_list)):
            if black_piece_list[i].role == "pawn":
                piece_move_list = black_pawn_move(black_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if black_piece_list[i].role == "knight":
                piece_move_list = knight_move(black_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if black_piece_list[i].role == "bishop":
                piece_move_list = bishop_move(black_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if black_piece_list[i].role == "rook":
                piece_move_list = rook_move(black_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if black_piece_list[i].role == "queen":
                piece_move_list = queen_move(black_piece_list[i])
                sorted_list = [x for x in piece_move_list if x in attack_list]
                possible_moves.append(sorted_list)
            if black_piece_list[i].role == "king":
                possible_moves.append(king_move(black_piece_list[i]))
        return possible_moves



white_pawn1 = piece("pawn", "white", "a2")
white_pawn2 = piece("pawn", "white", "b2")
white_pawn3 = piece("pawn", "white", "c2")
white_pawn4 = piece("pawn", "white", "d2")
white_pawn5 = piece("pawn", "white", "e2")
white_pawn6 = piece("pawn", "white", "f2")
white_pawn7 = piece("pawn", "white", "g2")
white_pawn8 = piece("pawn", "white", "h2")

white_knight1 = piece("knight", "white", "b1")
white_knight2 = piece("knight", "white", "g1")

white_bishop1 = piece("bishop", "white", "c1")
white_bishop2 = piece("bishop", "white", "f1")

white_rook1 = piece("rook", "white", "a1")
white_rook2 = piece("rook", "white", "h1")

white_king = piece("king", "white", "e1")
white_queen = piece("queen", "white", "d1")

black_pawn1 = piece("pawn", "black", "a7")
black_pawn2 = piece("pawn", "black", "b7")
black_pawn3 = piece("pawn", "black", "c7")
black_pawn4 = piece("pawn", "black", "d7")
black_pawn5 = piece("pawn", "black", "e7")
black_pawn6 = piece("pawn", "black", "f7")
black_pawn7 = piece("pawn", "black", "g7")
black_pawn8 = piece("pawn", "black", "h7")

black_knight1 = piece("knight", "black", "b8")
black_knight2 = piece("knight", "black", "g8")

black_bishop1 = piece("bishop", "black", "c8")
black_bishop2 = piece("bishop", "black", "f8")

black_rook1 = piece("rook", "black", "a8")
black_rook2 = piece("rook", "black", "h8")

black_king = piece("king", "black", "e8")
black_queen = piece("queen", "black", "d8")

white_piece_list = [
    white_pawn1, white_pawn2, white_pawn3, white_pawn4,white_pawn5, white_pawn6, white_pawn7, white_pawn8, white_knight1,
    white_knight2, white_bishop1, white_bishop2, white_rook1, white_rook2,white_queen, white_king
]

white_pawn_list = [white_pawn1, white_pawn2, white_pawn3, white_pawn4,white_pawn5, white_pawn6, white_pawn7, white_pawn8]

black_pawn_list = [black_pawn1, black_pawn2, black_pawn3, black_pawn4, black_pawn5, black_pawn6, black_pawn7, black_pawn8]

black_piece_list = [
    black_pawn1, black_pawn2, black_pawn3, black_pawn4, black_pawn5, black_pawn6, black_pawn7, black_pawn8, black_knight1,
    black_knight2, black_bishop1, black_bishop2, black_rook1, black_rook2, black_queen, black_king
]

pieces_list = [
    white_pawn1, white_pawn2, white_pawn3, white_pawn4,white_pawn5, white_pawn6, white_pawn7, white_pawn8, white_knight1,
    white_knight2, white_rook1, white_rook2, white_bishop1, white_bishop2, white_queen, white_king,
    black_pawn1, black_pawn2, black_pawn3, black_pawn4, black_pawn5, black_pawn6, black_pawn7, black_pawn8, black_knight1,
    black_knight2, black_bishop1, black_bishop2, black_rook1, black_rook2, black_queen, black_king
]

class DQN(nn.Module):
    def __init__(self, in_ch=832, n_actions=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, done = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(ns),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)

device = "cuda" if torch.cuda.is_available() else "cpu"

game_count = 0

def state_to_tensor(state_2d, device):
    # state_2d: list[list[float]] (H x W)
    # DQN 입력을 (1, H, W) 또는 (1, H*W)로 맞출 수 있음
    x = torch.tensor(state_2d, dtype=torch.float32, device=device)
    return x  # shape: (H, W)

def state_to_tensor_flat(state_2d, device):
    x = torch.tensor(state_2d, dtype=torch.float32, device=device)  # (H, W)
    return x.flatten()  # (H*W,)

def flatten_action(i, j, W):
    return i * W + j

def unflatten_action(a, W):
    return a // W, a % W

def pack_action(n, m):
    return n * MAX_MOVES + m

def unpack_action(a):
    return a // MAX_MOVES, a % MAX_MOVES

MAX_PIECES = 16
MAX_MOVES  = 256
N_ACTIONS  = MAX_PIECES * MAX_MOVES  # 4096

def pack_action(n, m):
    return n * MAX_MOVES + m

def select_action(model, state, legal_pairs, eps, device):
    """
    legal_pairs: [(n,m), ...]  (n=기물 인덱스 0..15, m=이동 인덱스 0..255)
    """

    # 1) legal (n,m) 정리 + 범위체크 + 1D action_id로 변환
    legal_1d = []
    for nm in legal_pairs:
        # 혹시 텐서가 섞였을 때 방어
        if isinstance(nm, torch.Tensor):
            nm = nm.tolist()

        n, m = nm  # 튜플/리스트라고 가정
        n = int(n); m = int(m)

        if 0 <= n < MAX_PIECES and 0 <= m < MAX_MOVES:
            legal_1d.append(pack_action(n, m))

    if not legal_1d:
        raise RuntimeError(f"legal_pairs empty or out of range. original size={len(legal_pairs)}")

    # 2) epsilon-greedy
    if random.random() < eps:
        a = random.choice(legal_1d)
        return a  # 필요하면 unpack해서 (n,m)로 반환해도 됨

    # 3) state 텐서화 (경고 제거 버전)
    if isinstance(state, torch.Tensor):
        x = state.to(device).float()
    else:
        x = torch.as_tensor(state, dtype=torch.float32, device=device)

    with torch.no_grad():
        q = model(x.unsqueeze(0))[0]  # (4096,)

        # 4) mask는 무조건 (4096,)
        mask = torch.full((N_ACTIONS,), float("-inf"), device=device)
        idx = torch.tensor(legal_1d, dtype=torch.long, device=device)
        mask[idx] = 0.0

        a = torch.argmax(q + mask).item()
        return a
PIECE_TO_CH = {
    # white
    ("white","pawn"): 0,
    ("white","knight"): 1,
    ("white","bishop"): 2,
    ("white","rook"): 3,
    ("white","queen"): 4,
    ("white","king"): 5,
    # black
    ("black","pawn"): 6,
    ("black","knight"): 7,
    ("black","bishop"): 8,
    ("black","rook"): 9,
    ("black","queen"): 10,
    ("black","king"): 11,
}

def square_to_index(sq: str) -> int:
    # sq: 'a1'..'h8'
    file = files.index(sq[0])
    rank = ranks.index(sq[1])
    # 너 코드에서 row=ranks.index(rank) (a2 -> row=1) 였지?
    # 여기서는 (row, col) = (rank_index, file_index)
    row = rank
    col = file
    return row * 8 + col  # 0..63

def index_to_square(idx: int) -> str:
    row = idx // 8
    col = idx % 8
    return files[col] + ranks[row]


def board_to_tensor(pieces_list, turn: str) -> torch.Tensor:
    x = np.zeros((13, 8, 8), dtype=np.float32)

    for p in pieces_list:
        if not p.active:
            continue
        key = (p.color.lower(), p.role.lower())
        if key not in PIECE_TO_CH:
            continue
        ch = PIECE_TO_CH[key]
        idx = square_to_index(p.pos)
        r, c = idx // 8, idx % 8
        x[ch, r, c] = 1.0

    # turn plane
    x[12, :, :] = 1.0 if turn == "white" else 0.0
    return torch.from_numpy(x)

black_dqn = DQN().to(device)
white_dqn = DQN().to(device)
opt_black = optim.Adam(black_dqn.parameters())
opt_white = optim.Adam(white_dqn.parameters())
buf_white = ReplayBuffer(capacity=50000)
buf_black = ReplayBuffer(capacity=50000)
black_target = DQN().to(device)
white_target = DQN().to(device)
black_target.eval()
white_target.eval()

state = board_to_tensor(pieces_list, turn).to(device).flatten()

batch_size = 64
step_count = 0
eps = 1

import cv2
import numpy as np
from PIL import Image, ImageDraw

def render_board_image(pieces_list):
    image = Image.new("RGB", (640, 640), color="white")
    draw = ImageDraw.Draw(image)

    # 체스판 칠하기 (네 코드 방식 그대로 가능)
    for r in range(8):
        for c in range(8):
            x0, y0 = c*80, r*80
            x1, y1 = x0+80, y0+80
            if (r+c) % 2 == 0:
                draw.rectangle((x0,y0,x1,y1), fill="white", outline="black")
            else:
                draw.rectangle((x0,y0,x1,y1), fill="green", outline="black")

    # 기물 그리기
    for p in pieces_list:
        if not p.active:
            continue
        y, x = square_to_rc(p.pos)
        color = p.color.lower()
        role = p.role.lower()
        path = f"C:/Users/ginok/Downloads/chess_pieces_80x80_transparent/{color}_{role}.png"
        piece_img = Image.open(path).convert("RGBA")

        # 너 코드에서 y축 뒤집었지? 그대로 적용
        image.paste(piece_img, (80 * x, 560 - 80 * y), piece_img)

    return image

def show_live(pieces_list, win="chess"):
    img = render_board_image(pieces_list)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow(win, frame)
    # 1ms 기다리며 창 갱신 + 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    return key

while game_count <= 3000:

    diagonal_pin(white_bishop1)
    diagonal_pin(white_bishop2)
    diagonal_pin(white_queen)
    vertical_pin(white_rook1)
    vertical_pin(white_rook2)

    diagonal_pin(black_bishop1)
    diagonal_pin(black_bishop2)
    diagonal_pin(black_queen)
    vertical_pin(black_rook1)
    vertical_pin(black_rook2)

    move = move_list(turn)

    print(move)

    legal_pairs = [(n, m) for n, moves in enumerate(move) for m in range(len(moves))]
    '''
    if len(legal_pairs) == 0:
        color, status, reward = CheckMate(move)  # move가 빈 리스트들로 구성돼도 len(possible_moves)==0 이어야 함
        print("TERMINAL:", turn, status, "reward:", reward)
        reset()
        game_count += 1
        continue
    '''
    mn = min(move)
    mx = max(move)
    print("legal size:", len(move), "min:", mn, "max:", mx)
    if turn == "white":
        a1d = select_action(white_dqn, state, legal_pairs, eps, device)
        n, m = unpack_action(a1d)
        print(move[n][m])
        move_choice(n, m, move)
        move = move_list(turn)
        color, status, reward = CheckMate(move)
        next_state = board_to_tensor(pieces_list, turn).to(device).flatten()
        buf_white.push(state.cpu(), a1d, reward, next_state.cpu(), status)
        state = board_to_tensor(pieces_list, turn).to(device).flatten()
        step_count += 1

        # 학습
        if len(buf_white) >= batch_size:
            S, A, R, NS, D = buf_white.sample(batch_size)
            S = S.to(device)
            A = A.to(device)
            R = R.to(device)
            NS = NS.to(device)
            D = D.to(device)

            q = white_dqn(S).gather(1, A.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # 타겟은 max_a' Q_target(s', a')
                q_next = white_target(NS).max(dim=1).values
                y = R + 0.99 * (1.0 - D) * q_next

            loss = nn.functional.smooth_l1_loss(q, y)
            opt_white.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(white_dqn.parameters(), 1.0)
            opt_white.step()

        # 타겟 업데이트
        if step_count % 200 == 0:
            white_target.load_state_dict(white_dqn.state_dict())

        if status:
            break

    if turn == "black":
        a1d = select_action(black_dqn, state, legal_pairs, eps, device)
        n, m = unpack_action(a1d)
        print(move[n][m])
        move_choice(n, m, move)
        move = move_list(turn)
        color, status, reward = CheckMate(move)
        next_state = board_to_tensor(pieces_list, turn).to(device).flatten()
        buf_black.push(state.cpu(), a1d, reward, next_state.cpu(), status)
        state = board_to_tensor(pieces_list, turn).to(device).flatten()
        step_count += 1
        # 학습
        if len(buf_black) >= batch_size:
            S, A, R, NS, D = buf_black.sample(batch_size)
            S = S.to(device)
            A = A.to(device)
            R = R.to(device)
            NS = NS.to(device)
            D = D.to(device)

            q = black_dqn(S).gather(1, A.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # 타겟은 max_a' Q_target(s', a')
                q_next = black_target(NS).max(dim=1).values
                y = R + 0.99 * (1.0 - D) * q_next

            loss = nn.functional.smooth_l1_loss(q, y)
            opt_black.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(black_dqn.parameters(), 1.0)
            opt_black.step()

        # 타겟 업데이트
        if step_count % 200 == 0:
            black_target.load_state_dict(black_dqn.state_dict())

        if status:
            break

    eps = max(0.05, eps * 0.999)

    if (game_count + 1) % 50 == 0:
        print(f"[EP {game_count + 1}] eps={eps:.3f} last_status={status: ('status')} buffer={len(buf_black)}")

    for i in range(len(pieces_list)):
        pieces_list[i].absolute_pin = False
        pieces_list[i].protected = False

    turn, turn_count = turn_change(turn, turn_count)
    time.sleep(1)

    key = show_live(pieces_list)
    if key == ord('q'):
        break

    '''
    # 빈 이미지 생성 (RGB, 크기: 800x800, 색상: 흰색)
    image = Image.new("RGB", (640, 640), color="white")

    # 이미지를 조작할 도구 생성
    draw = ImageDraw.Draw(image)

    # 직사각형 패턴 그리기
    for a in range(0, 8, 2):
        for i in range(4):  # 반복문을 통해 세 줄 생성
            # 첫 번째 직사각형 (흰색)
            draw.rectangle((0 + a * 80, 160 * i, 80 + a * 80, 160 * i + 80), fill="white", outline="black")

            # 두 번째 직사각형 (녹색)
            draw.rectangle((0 + a * 80, 160 * i + 80, 80 + a * 80, 160 * i + 160), fill="green", outline="black")

    for a in range(1, 9, 2):
        for i in range(4):  # 반복문을 통해 세 줄 생성
            # 첫 번째 직사각형 (흰색)
            draw.rectangle((0 + a * 80, 160 * i, 80 + a * 80, 160 * i + 80), fill="green", outline="black")

            # 두 번째 직사각형 (녹색)
            draw.rectangle((0 + a * 80, 160 * i + 80, 80 + a * 80, 160 * i + 160), fill="white", outline="black")

    for p in pieces_list:
        if p.active == True:
            y, x = square_to_rc(p.pos)

            color = p.color.lower()  # "white"/"black"
            role = p.role.lower()  # "pawn"...

            path = f"C:/Users/ginok/Downloads/chess_pieces_80x80_transparent/{color}_{role}.png"
            piece_img = Image.open(path).convert("RGBA")

            image.paste(piece_img, (80 * x, 560 - 80 * y), piece_img)
        else:
            continue

    image.save("generated_image.png")
    image.show()
    '''

