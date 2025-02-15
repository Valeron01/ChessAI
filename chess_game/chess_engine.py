import copy
import enum

import numpy as np
from chess_game.common_things import Piece, PieceType, PieceColor, invert_color


class ChessField:
    @staticmethod
    def init_game() -> "ChessField":
        field = np.full([8, 8], None, dtype=Piece)
        for i in range(8):
            for j in range(8):
                field[i, j] = Piece(PieceType.EMPTY, None)
        for i in range(8):
            field[1, i] = Piece(PieceType.PAWN, PieceColor.BLACK)
            field[-2, i] = Piece(PieceType.PAWN, PieceColor.WHITE)

        piece_types = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP]
        for i, piece_type in enumerate(piece_types):
            field[0, i] = Piece(piece_type, PieceColor.BLACK)
            field[0, -i-1] = Piece(piece_type, PieceColor.BLACK)
            field[-1, i] = Piece(piece_type, PieceColor.WHITE)
            field[-1, -i-1] = Piece(piece_type, PieceColor.WHITE)

        field[0, 3] = Piece(PieceType.QUEEN, PieceColor.BLACK)
        field[0, 4] = Piece(PieceType.KING, PieceColor.BLACK)

        field[-1, 3] = Piece(PieceType.QUEEN, PieceColor.WHITE)
        field[-1, 4] = Piece(PieceType.KING, PieceColor.WHITE)

        return ChessField(field)

    @staticmethod
    def init_random():
        field = ChessField.init_game()
        arr = field.__field_array
        arr = arr.flatten()
        np.random.shuffle(arr)
        arr = arr.reshape([8, 8])
        field.__field_array = arr
        return field

    def __init__(self, field_array: np.ndarray[Piece]):
        assert field_array.shape == (8, 8)
        self.__field_array = field_array

    @staticmethod
    def is_cell_black(row, column):
        return (row + column) % 2 != 0

    def __getitem__(self, item) -> Piece:
        row, column = item
        if row < 0 or column < 0 or row > 7 or column > 7:
            raise IndexError(f"Row index {row} or column {column} is outside of bounds")

        return self.__field_array[row, column]

    def __setitem__(self, key, value):
        row, column = key
        if row < 0 or column < 0 or row > 7 or column > 7:
            raise IndexError(f"Row index {row} or column {column} is outside of bounds")

        self.__field_array[row, column] = value

    def is_empty(self, row, column):
        piece = self.__field_array[row, column]
        return piece.piece_type == PieceType.EMPTY

    def flipped_sides(self):
        self_copy = copy.deepcopy(self)
        self_copy.__field_array = np.flip(self_copy.__field_array, axis=[0])
        for i in range(8):
            for j in range(8):
                piece = self_copy[i, j]
                piece.piece_color = invert_color(piece.piece_color)
        return self_copy

    @staticmethod
    def flip_move(source_row, source_column, target_row, target_column):
        return 7 - source_row, source_column, 7 - target_row, target_column


class MoveChecker:
    @staticmethod
    def __check_move_pawn(field: ChessField, source_row, source_column, target_row, target_column):
        source_pawn = field[source_row, source_column]
        target_piece = field[target_row, target_column]

        delta_row = target_row - source_row
        delta_column = target_column - source_column

        if source_pawn.piece_color == PieceColor.BLACK:
            if delta_row == 1:
                if delta_column == 0 and target_piece.piece_type == PieceType.EMPTY:
                    return True
                if (delta_column == -1 or delta_column == 1) and target_piece.piece_color == PieceColor.WHITE:
                    return True
            if delta_row == 2 and delta_column == 0 and field.is_empty(source_row + 1, source_column):
                return True
        if source_pawn.piece_color == PieceColor.WHITE:
            if delta_row == -1:
                if delta_column == 0 and target_piece.piece_type == PieceType.EMPTY:
                    return True
                if (delta_column == -1 or delta_column == 1) and target_piece.piece_color == PieceColor.BLACK:
                    return True
            if delta_row == -2 and delta_column == 0 and field.is_empty(source_row - 1, source_column):
                return True
        return False

    @staticmethod
    def __check_move_knight(field: ChessField, source_row, source_column, target_row, target_column):
        source_piece = field[source_row, source_column]
        target_piece = field[target_row, target_column]

        delta_row = target_row - source_row
        delta_column = target_column - source_column

        if (abs(delta_row) == 2 and abs(delta_column) == 1) or (abs(delta_row) == 1 and abs(delta_column) == 2):
            if source_piece.piece_color != target_piece.piece_color:
                return True

        return False

    @staticmethod
    def __check_move_bishop(field: ChessField, source_row, source_column, target_row, target_column):
        delta_row = target_row - source_row
        delta_column = target_column - source_column

        if abs(delta_column) != abs(delta_row):
            return False

        delta_i = np.sign(delta_row)
        delta_j = np.sign(delta_column)

        i = source_row + delta_i
        j = source_column + delta_j
        has_path = True
        index = 0
        max_index = max(abs(delta_row), abs(delta_column)) - 1
        while index < max_index and has_path:
            has_path = field.is_empty(i, j)
            i += delta_i
            j += delta_j
            index += 1

        return has_path

    @staticmethod
    def __check_move_rook(field: ChessField, source_row, source_column, target_row, target_column):
        delta_row = target_row - source_row
        delta_column = target_column - source_column
        delta_i = np.sign(delta_row)
        delta_j = np.sign(delta_column)

        if not (delta_i == 0 or delta_j == 0):
            return False

        i = source_row + delta_i
        j = source_column + delta_j
        has_path = True
        index = 0
        max_index = max(abs(delta_row), abs(delta_column)) - 1
        while index < max_index and has_path:
            has_path = field.is_empty(i, j)
            i += delta_i
            j += delta_j
            index += 1

        return has_path

    @staticmethod
    def __check_move_queen(field: ChessField, source_row, source_column, target_row, target_column):
        return MoveChecker.__check_move_rook(
            field, source_row, source_column, target_row, target_column) or MoveChecker.__check_move_bishop(
            field, source_row, source_column, target_row, target_column
        )

    @staticmethod
    def __check_move_king(field: ChessField, source_row, source_column, target_row, target_column):
        delta_row = target_row - source_row
        delta_col = target_column - source_column

        if abs(delta_row) > 1 or abs(delta_col) > 1:
            return False

        king_color = field[source_row, source_column].piece_color
        opponent_color = invert_color(king_color)
        for i in range(8):
            for j in range(8):
                if field[i, j].piece_color == opponent_color:
                    if field[i, j].piece_type == PieceType.KING:
                        if abs(i - target_row) <= 1 or abs(j - target_column) <= 1:
                            return False
                    elif MoveChecker.check_move(field, i, j, target_row, target_column):
                        return False

        return True


    @staticmethod
    def check_move(field: ChessField, source_row, source_column, target_row, target_column):
        try:
            target_piece = field[target_row, target_column]
            source_piece = field[source_row, source_column]
        except IndexError as e:
            print(e)
            return False

        if source_row == target_row and source_column == target_column:
            return False
        if source_piece.piece_type == PieceType.EMPTY:
            return False

        if target_piece.piece_color == source_piece.piece_color:
            return False

        if source_piece.piece_type == PieceType.PAWN:
            return MoveChecker.__check_move_pawn(field, source_row, source_column, target_row, target_column)

        if source_piece.piece_type == PieceType.KNIGHT:
            return MoveChecker.__check_move_knight(field, source_row, source_column, target_row, target_column)

        if source_piece.piece_type == PieceType.BISHOP:
            return MoveChecker.__check_move_bishop(field, source_row, source_column, target_row, target_column)

        if source_piece.piece_type == PieceType.ROOK:
            return MoveChecker.__check_move_rook(field, source_row, source_column, target_row, target_column)

        if source_piece.piece_type == PieceType.QUEEN:
            return MoveChecker.__check_move_queen(field, source_row, source_column, target_row, target_column)

        if source_piece.piece_type == PieceType.KING:
            return MoveChecker.__check_move_king(field, source_row, source_column, target_row, target_column)

        raise NotImplementedError()


class StepResult(enum.Enum):
    INVALID_MOVE = 0
    PERFORMED = 1
    PERFORMED_KILL = 2


class ChessEngine:

    @staticmethod
    def init_game():
        field = ChessField.init_game()
        current_player_color = PieceColor.WHITE
        dead_whites = []
        dead_blacks = []

        return ChessEngine(field, current_player_color, dead_whites, dead_blacks)

    def __init__(self, field: ChessField, current_player_color: PieceColor, dead_whites, dead_blacks):
        self.field = field
        self.current_player_color = current_player_color
        self.dead_whites = dead_whites
        self.dead_blacks = dead_blacks

    def make_step(self, source_row, source_column, target_row, target_column):
        is_step_possible = self.field[source_row, source_column].piece_color == self.current_player_color
        is_step_possible = is_step_possible and MoveChecker.check_move(
            self.field, source_row, source_column, target_row, target_column
        )
        step_result = StepResult.INVALID_MOVE
        killed_piece = None
        moved_piece = None
        if is_step_possible:
            target_piece = self.field[target_row, target_column]
            if target_piece.piece_type != PieceType.EMPTY:
                if target_piece.piece_color == PieceColor.BLACK:
                    self.dead_blacks.append(target_piece.piece_type)
                if target_piece.piece_color == PieceColor.WHITE:
                    self.dead_whites.append(target_piece.piece_type)
                killed_piece = target_piece.piece_type
                step_result = StepResult.PERFORMED_KILL
            else:
                step_result = StepResult.PERFORMED
                moved_piece = self.field[source_row, source_column].piece_type

            self.current_player_color = invert_color(self.current_player_color)

            self.field[target_row, target_column] = self.field[source_row, source_column]
            self.field[source_row, source_column] = Piece(PieceType.EMPTY, None)

        return step_result, moved_piece, killed_piece

    def flipped_sides(self):
        board_inverted = self.field.flipped_sides()
        player_color = invert_color(self.current_player_color)
        return ChessEngine(
            board_inverted, player_color, copy.deepcopy(self.dead_blacks), copy.deepcopy(self.dead_whites)
        )
