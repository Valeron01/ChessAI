#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from common_things import PieceType, PieceColor


class PieceRenderer:
    def __init__(self, font_size, font_color=0x000000):
        # pieces = u"♔♕♖♗♘♙♚♛♜♝♞♟"
        self.font = ImageFont.truetype(r"./FreeSerif.ttf", font_size)
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


def main():
    image = Image.new("RGB", (500, 500), 0xffffff)

    renderer = PieceRenderer(150)
    renderer.draw_piece(
        image, 100, 100, PieceType.QUEEN, PieceColor.WHITE
    )

    image.save("./test_image.jpg")



if __name__ == '__main__':
    main()
