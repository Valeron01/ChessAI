import random

import cv2
import numpy as np
from tqdm import trange

from chess_game.chess_engine import ChessField, MoveChecker, ChessEngine, StepResult, invert_move
from chess_game.renderer import PieceRenderer


def main():
    chess = ChessEngine.init_game()
    chess_flipped = chess.flipped_sides()
    renderer = PieceRenderer(64)
    for i in trange(3000):
        source_row, source_column, target_row, target_column = np.unravel_index(random.randrange(0, 4096), [8, 8, 8, 8])
        step_result, action_result_dict = chess.make_step(
            source_row, source_column, target_row, target_column
        )
        _ = chess_flipped.make_step(
            *invert_move(source_row, source_column, target_row, target_column)
        )

        if step_result == StepResult.PERFORMED or step_result == StepResult.PERFORMED_KILL:
            image = renderer.render_field(chess.field)
            image_inverted = renderer.render_field(chess_flipped.field)
            cv2.imshow("qwe", np.concatenate([image, image_inverted], 1))
            cv2.waitKey(1)

    cv2.waitKey(0)

    flipped_game = chess.flipped_sides()
    image_flipped = renderer.render_field(flipped_game.field)
    cv2.imshow("qwe_flipped", image_flipped)
    cv2.waitKey(0)




if __name__ == '__main__':
    main()
