import enum


class PieceType(enum.Enum):
    PAWN = 0
    ROOK = 1
    KNIGHT = 2
    BISHOP = 3
    QUEEN = 4
    KING = 5


class PieceColor(enum.Enum):
    BLACK = 8
    WHITE = 16

