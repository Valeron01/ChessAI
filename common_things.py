import enum
import typing


class PieceType(enum.Enum):
    PAWN = 0
    ROOK = 1
    KNIGHT = 2
    BISHOP = 3
    QUEEN = 4
    KING = 5

    EMPTY = 6


class PieceColor(enum.Enum):
    BLACK = 8
    WHITE = 16


class Piece:
    def __init__(self, piece_type: PieceType, piece_color: typing.Union[PieceColor, None]):
        self.piece_type = piece_type
        self.piece_color = piece_color


def invert_color(piece_color):

    if piece_color is None:
        return None
    if piece_color == PieceColor.WHITE:
        return PieceColor.BLACK
    elif piece_color == PieceColor.BLACK:
        return PieceColor.WHITE
    else:
        raise NotImplementedError()
