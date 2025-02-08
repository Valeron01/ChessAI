import numpy as np
from common_things import Piece, PieceType, PieceColor


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

    def is_empty(self, row, column):
        piece = self.__field_array[row, column]
        return piece.piece_type == PieceType.EMPTY


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
    def check_move(field: ChessField, source_row, source_column, target_row, target_column):
        source_piece = field[source_row, source_column]
        if source_row == target_row and source_column == target_column:
            return False
        if source_piece.piece_type == PieceType.EMPTY:
            return False
        target_piece = field[target_row, target_column]
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

        raise NotImplementedError()






if __name__ == '__main__':
    f = ChessField.init_game()
    print(f.__field_array)