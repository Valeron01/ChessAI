import cv2
import numpy as np

from chess_engine import ChessField, MoveChecker
from common_things import PieceColor, PieceType
from renderer import PieceRenderer


def main():
    field = ChessField.init_game()
    renderer = PieceRenderer(64)


    image = renderer.render_field(field)
    cv2.imshow("qwe", image)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()
