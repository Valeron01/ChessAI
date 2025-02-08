import numpy as np
from common_things import Piece, PieceType, PieceColor


class ChessField:
    @staticmethod
    def init_game() -> "ChessField":
        field = np.full([8, 8], Piece(PieceType.EMPTY, None), dtype=Piece)
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
        self.field_array = field_array

    @staticmethod
    def is_cell_black(row, column):
        return (row + column) % 2 != 0


if __name__ == '__main__':
    f = ChessField.init_game()
    print(f.field_array)