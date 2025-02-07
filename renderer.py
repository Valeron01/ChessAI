#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from common_things import PieceType, PieceColor


class PieceRenderer:
    def __init__(self, cell_size, font_color=0x000000):
        # pieces = u"♔♕♖♗♘♙♚♛♜♝♞♟"
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


def build_checkerboard_image(field_size, cell_size):
    resulted_image = np.zeros([field_size * cell_size, field_size * cell_size, 3], dtype=np.uint8)
    colors_dict = {
        0: [105, 58, 20],
        1: [235, 146, 63]
    }
    for i in range(field_size):
        for j in range(field_size):
            color = (i + j) % 2
            resulted_image[i * cell_size:i*cell_size + cell_size, j * cell_size:j * cell_size + cell_size] = colors_dict[color]
    return Image.fromarray(resulted_image, "RGB")


def main():
    # image = Image.new("RGB", (500, 500), 0xffffff)
    checkerboard = build_checkerboard_image(8, 64)
    renderer = PieceRenderer(64)
    renderer.draw_piece(
        checkerboard, 32, 32, PieceType.QUEEN, PieceColor.BLACK
    )

    checkerboard.save("./test_image.jpg")



if __name__ == '__main__':
    main()
