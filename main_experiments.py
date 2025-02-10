import cv2
import numpy as np
from tqdm import trange

from chess_engine import ChessField, MoveChecker, ChessEngine, StepResult
from common_things import PieceColor, PieceType
from renderer import PieceRenderer


def main():
    chess = ChessEngine.init_game()
    renderer = PieceRenderer(64)
    for i in trange(1000):
        source_row, source_column, target_row, target_column = np.random.randint(0, 8, [4], dtype=int)
        step_result, action_result_dict = chess.make_step(
            source_row, source_column, target_row, target_column
        )
        # print(step_result)
        if step_result == StepResult.PERFORMED or step_result == StepResult.PERFORMED_KILL:
            image = renderer.render_field(chess.field)
            cv2.imshow("qwe", image)
            cv2.waitKey(1)

    cv2.waitKey(0)

    flipped_game = chess.flipped_sides()
    image_flipped = renderer.render_field(flipped_game.field)
    cv2.imshow("qwe_flipped", image_flipped)
    cv2.waitKey(0)




if __name__ == '__main__':
    main()
