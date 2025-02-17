import cv2
import numpy as np
import torch

from chess_env import ChessEnv
from chess_game.chess_engine import ChessField
from chess_game.common_things import PieceColor
from chess_game.renderer import PieceRenderer
from nn_modules.basic_transformer_model import BasicTransformerModel
env_params = {
    "performed_reward": -0.00005,
    "blocked_reward": -0.02,
    "terminate_iters": 512,
    "fifty_rule_steps": 15,
    "fifty_rule_penalty": -0.2,
    "rand_field_prob": 0,
    "n_bad_steps_to_terminate": 10000
    }
env = ChessEnv(
    **env_params
)

model = torch.load(
    "/home/valera/PycharmProjects/ChessAI/logs_ppo/run_227/Checkpoints/Checkpoint.pt"
).eval().requires_grad_(False)
renderer = PieceRenderer(64)
device = "cuda"
for i_step in range(0, 10000):
    env.chess_game.current_player_color = PieceColor.WHITE
    state = env.state().to(device)[None]
    print(state.dtype)
    with torch.inference_mode():
        actions_per_env, values_per_env = model(state)

    step_index = actions_per_env.sample().item()
    print(np.unravel_index(step_index, [8, 8, 8, 8]))

    env.step(step_index)

    field = env.chess_game.field

    image = renderer.render_field(field)
    cv2.imshow("qwe", image)
    cv2.waitKey(1)
cv2.waitKey(0)
cv2.waitKey(0)
cv2.waitKey(0)
