import random

import numpy as np
import torch

from chess_game.chess_engine import ChessField, ChessEngine, PieceColor, PieceType, StepResult, invert_move


class ChessEnv:
    def __init__(
            self,
            performed_reward: float,
            blocked_reward: float,
            terminate_iters: int,
            fifty_rule_steps: int,
            fifty_rule_penalty: float,
            rand_field_prob: float
    ):
        self.fifty_rule_penalty = fifty_rule_penalty
        self.fifty_rule_steps = fifty_rule_steps
        self.terminate_iters = terminate_iters
        self.blocked_reward = blocked_reward
        self.performed_reward = performed_reward
        self.chess_game = ChessEngine.init_game()

        if rand_field_prob > random.random():
            self.chess_game.field = ChessField.init_random()

        self.reward_kill = {
            PieceType.PAWN: 1,
            PieceType.BISHOP: 3,
            PieceType.KNIGHT: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 20
        }

        self.steps_made = 0

    def get_state(self) -> torch.Tensor:
        field = self.chess_game.field
        if self.chess_game.current_player_color == PieceColor.BLACK:
            field = self.chess_game.field.flipped_sides()

        result = np.zeros([8, 8], dtype=np.int32)
        for i in range(8):
            for j in range(8):
                piece = field[i, j]
                piece_color = piece.piece_color
                piece_type = piece.piece_type
                if piece_type == PieceType.EMPTY:
                    continue
                shift_index = 0 if piece_color == PieceColor.WHITE else 6
                piece_index = int(piece_type.value) + shift_index

                result[i, j] = piece_index + 1
        return torch.from_numpy(result)




