#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from chess_engine import ChessField
from common_things import PieceType, PieceColor, Piece


class PieceRenderer:
    def __init__(self, cell_size, field_size: int = 8, font_color=0x000000):
        # pieces = u"♔♕♖♗♘♙♚♛♜♝♞♟"
        self.field_size = field_size
        self.cell_size = cell_size
        self.font_size = cell_size * 80 // 64
        self.font = ImageFont.truetype(r"./meta/FreeSerif.ttf", self.font_size)
        self.font_color = font_color
        self.pieces = {
            PieceColor.BLACK: {
                PieceType.PAWN: "♟",
                PieceType.ROOK: "♜",
                PieceType.KNIGHT: "♞",
                PieceType.BISHOP: "♝",
                PieceType.QUEEN: "♛",
                PieceType.KING: "♚",
            },
            PieceColor.WHITE: {
                PieceType.PAWN: "♙",
                PieceType.ROOK: "♖",
                PieceType.KNIGHT: "♘",
                PieceType.BISHOP: "♗",
                PieceType.QUEEN: "♕",
                PieceType.KING: "♔",
            }
        }

        self.empty_chessboard_image = build_checkerboard_image(field_size, cell_size)

    def draw_piece(self, image: Image.Image, x, y, piece_type: PieceType, piece_color: PieceColor):
        drawer = ImageDraw.Draw(image)
        text = self.pieces[piece_color][piece_type]
        bbox = drawer.textbbox((0, 0), text, self.font)
        center_dx = (bbox[2] + bbox[0]) // 2
        center_dy = (bbox[3] + bbox[1]) // 2

        position_x = x - center_dx
        position_y = y - center_dy

        drawer.text((position_x, position_y), text, font=self.font, fill=self.font_color)

        return image

    def render_field(self, chessboard_field_array: ChessField) -> np.ndarray:
        image = self.empty_chessboard_image.copy()
        for i in range(self.field_size):
            for j in range(self.field_size):
                x = self.cell_size / 2 + j * self.cell_size
                y = self.cell_size / 2 + i * self.cell_size
                piece: Piece = chessboard_field_array[i, j]
                if piece.piece_type != PieceType.EMPTY:
                    self.draw_piece(image, x, y, piece_type=piece.piece_type, piece_color=piece.piece_color)

        return np.uint8(image)[..., ::-1]


def build_checkerboard_image(field_size, cell_size):
    resulted_image = np.zeros([field_size * cell_size, field_size * cell_size, 3], dtype=np.uint8)
    colors_dict = {
        True: [105, 58, 20],
        False: [235, 146, 63]
    }
    for i in range(field_size):
        for j in range(field_size):
            color = ChessField.is_cell_black(i, j)
            resulted_image[i * cell_size:i*cell_size + cell_size, j * cell_size:j * cell_size + cell_size] = colors_dict[color]
    return Image.fromarray(resulted_image, "RGB")


