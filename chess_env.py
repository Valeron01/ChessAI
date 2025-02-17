import random

import numpy as np
import torch

from chess_game.chess_engine import ChessField, ChessEngine, PieceColor, PieceType, StepResult


class ChessEnv:
    def __init__(
            self,
            performed_reward: float,
            blocked_reward: float,
            terminate_iters: int,
            fifty_rule_steps: int,
            fifty_rule_penalty: float,
            rand_field_prob: float,
            n_bad_steps_to_terminate: int
    ):
        self.n_bad_steps_to_terminate = n_bad_steps_to_terminate
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
        self.invertable_steps_made = 0
        self.good_steps = 0
        self.bad_steps = 0

        self.history_size = 8

        self._field_states = [torch.zeros(12, 8, 8)] * self.history_size
        self.side_invertion_indices = ChessEnv.build_invert_side_indices(self.history_size)

    @staticmethod
    def build_invert_side_indices(history_size):
        single_indices = torch.cat([torch.arange(6, 12), torch.arange(0, 6)])
        resulted_indices = []
        for i in range(history_size):
            resulted_indices.append(single_indices + i * 12)

        return torch.cat(resulted_indices)

    def _current_field_state(self) -> torch.Tensor:
        field = self.chess_game.field

        resulted_state = np.zeros([12, 8, 8], dtype=np.float32)
        for i in range(8):
            for j in range(8):
                piece = field[i, j]
                piece_color = piece.piece_color
                piece_type = piece.piece_type
                if piece_type == PieceType.EMPTY:
                    continue
                shift_index = 0 if piece_color == PieceColor.WHITE else 6
                piece_index = int(piece_type.value) + shift_index

                resulted_state[piece_index, i, j] = 1
        return torch.from_numpy(resulted_state)

    def state(self):
        current_field_state = self._current_field_state()

        self._field_states.append(current_field_state)
        self._field_states = self._field_states[1:]
        assert len(self._field_states) == self.history_size

        resulted_state = torch.cat(self._field_states, 0)

        if self.chess_game.current_player_color == PieceColor.BLACK:
            resulted_state = resulted_state.flip([1])[self.side_invertion_indices]
        assert resulted_state.dtype == torch.float32
        return resulted_state

    def step(self, action: int):
        source_row, source_column, target_row, target_column = np.unravel_index(action, [8, 8, 8, 8])
        self.steps_made += 1

        if self.chess_game.current_player_color == PieceColor.BLACK:
            source_row = 7 - source_row
            target_row = 7 - target_row

        step_result, moved_piece, killed_piece = self.chess_game.make_step(source_row, source_column, target_row, target_column)
        reward = 0
        opponent_reward = 0
        done = False
        terminated = False
        return_mask = False
        if step_result == StepResult.INVALID_MOVE:
            reward = self.blocked_reward
            self.bad_steps += 1
            return_mask = True
        else:
            self.good_steps += 1
            if killed_piece is not None:
                reward = self.reward_kill[killed_piece]
                opponent_reward = -reward
            else:
                reward = self.performed_reward

            if killed_piece == PieceType.KING:
                done = True

            if moved_piece != PieceType.PAWN and killed_piece is None:
                self.invertable_steps_made += 1
            else:
                self.invertable_steps_made = 0

        if self.steps_made >= self.terminate_iters:
            terminated = True
            done = True
        if self.bad_steps >= self.n_bad_steps_to_terminate:
            done = True

        if self.invertable_steps_made >= self.fifty_rule_steps:
            terminated = True
            done = True
            print()
            print("Terminated, maybe loop")
            print()
            reward = self.fifty_rule_penalty

        reward = reward / 10
        return reward, opponent_reward, terminated, done, return_mask
