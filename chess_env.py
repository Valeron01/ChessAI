import random

import numpy as np
import torch

from chess_game.chess_engine import ChessField, ChessEngine, PieceColor, PieceType, StepResult, invert_move


class ChessEnv:
    def __init__(
            self, performed_reward, blocked_reward, terminate_iters, fifty_rule_steps, fifty_rule_penalty, rand_field_prob
    ):
        self.fifty_rule_penalty = fifty_rule_penalty
        self.fifty_rule_steps = fifty_rule_steps
        self.terminate_iters = terminate_iters
        self.blocked_reward = blocked_reward
        self.performed_reward = performed_reward
        self.chess_game_whites = ChessEngine.init_game()

        if rand_field_prob > random.random():
            self.chess_game_whites.field = ChessField.init_random()

        self.reward_kill = {
            PieceType.PAWN: 1,
            PieceType.BISHOP: 3,
            PieceType.KNIGHT: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 20
        }

        self.steps_made = 0

        self.reversible_steps_white = 0
        self.reversible_steps_black = 0

    def get_state_whites(self):
        result = np.zeros([8, 8], dtype=np.int32)

        for i in range(8):
            for j in range(8):
                piece_color = self.chess_game_whites.field[i, j].piece_color
                piece_type = self.chess_game_whites.field[i, j].piece_type
                if piece_type == PieceType.EMPTY:
                    continue

                shift_index = 0 if piece_color == PieceColor.WHITE else 6

                piece_index = int(piece_type.value) + shift_index

                result[i, j] = piece_index + 1

        return torch.from_numpy(result)

    def get_state_blacks(self):
        field = self.chess_game_whites.field.flipped_sides()
        result = np.zeros([8, 8], dtype=np.int32)
        for i in range(8):
            for j in range(8):
                piece_color = field[i, j].piece_color
                piece_type = field[i, j].piece_type
                if piece_type == PieceType.EMPTY:
                    continue

                shift_index = 0 if piece_color == PieceColor.WHITE else 6

                piece_index = int(piece_type.value) + shift_index

                result[i, j] = piece_index + 1

        return torch.from_numpy(result)

    def step_whites(self, prediction_index):
        assert self.chess_game_whites.current_player_color == PieceColor.WHITE
        self.steps_made += 1
        source_row, source_column, target_row, target_column = np.unravel_index(prediction_index, [8, 8, 8, 8])
        step_result, step_dict = self.chess_game_whites.make_step(source_row, source_column, target_row, target_column)
        reward_whites = 0
        reward_blacks = 0
        if step_result == StepResult.PERFORMED:
            reward_whites = self.performed_reward
            if step_dict["moved_piece"] != PieceType.PAWN:
                self.reversible_steps_white += 1
            else:
                self.reversible_steps_white = 0

        elif step_result == StepResult.INVALID_MOVE:
            reward_whites = self.blocked_reward
        elif step_result == StepResult.PERFORMED_KILL:
            reward_whites = self.reward_kill[step_dict["killed_piece"]]
            reward_blacks = -reward_whites
            self.reversible_steps_white = 0
        done = step_dict.get("killed_piece", None) == PieceType.KING
        done = done or self.steps_made >= self.terminate_iters
        if self.reversible_steps_white >= self.fifty_rule_steps:
            reward_whites = self.fifty_rule_penalty

        return reward_whites, reward_blacks, done, step_result

    def step_blacks(self, prediction_index):
        assert self.chess_game_whites.current_player_color == PieceColor.BLACK
        self.steps_made += 1
        source_row, source_column, target_row, target_column = np.unravel_index(prediction_index, [8, 8, 8, 8])
        source_row, source_column, target_row, target_column = invert_move(source_row, source_column, target_row, target_column)

        step_result, step_dict = self.chess_game_whites.make_step(source_row, source_column, target_row, target_column)

        reward_blacks = 0
        reward_whites = 0
        if step_result == StepResult.PERFORMED:
            reward_blacks = self.performed_reward
            if step_dict["moved_piece"] != PieceType.PAWN:
                self.reversible_steps_black += 1
            else:
                self.reversible_steps_black = 0
        elif step_result == StepResult.INVALID_MOVE:
            reward_blacks = self.blocked_reward
        elif step_result == StepResult.PERFORMED_KILL:
            reward_blacks = self.reward_kill[step_dict["killed_piece"]]
            reward_whites = -reward_blacks
            self.reversible_steps_black = 0
        done = step_dict.get("killed_piece", None) == PieceType.KNIGHT
        done = done or self.steps_made >= self.terminate_iters

        if self.reversible_steps_black >= self.fifty_rule_steps:
            reward_blacks = self.fifty_rule_penalty

        return reward_blacks, reward_whites, done, step_result

