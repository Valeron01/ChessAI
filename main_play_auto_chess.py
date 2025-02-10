import cv2
import torch

from chess_env import ChessEnv
from chess_game.common_things import PieceColor
from chess_game.renderer import PieceRenderer
from nn_modules.basic_transformer_model import BasicTransformerModel

env = ChessEnv(
    0, 0, 1000, 50, 0
)

model = torch.load(
    "/home/valera/PycharmProjects/ChessAI/logs_ppo/run_68/Checkpoints/Checkpoint.pt"
).eval().requires_grad_(False)
renderer = PieceRenderer(64)
device = "cuda"
for i_step in range(0, 1000):
    current_side = env.chess_game_whites.current_player_color
    if current_side == PieceColor.WHITE:
        state = env.get_state_whites()
    else:
        state = env.get_state_blacks()
    state = state.to(device)[None]

    with torch.inference_mode():
        actions_per_env, values_per_env = model(state)

    step_index = actions_per_env.sample().item()

    if current_side == PieceColor.WHITE:
        reward_whites, reward_blacks, done, step_result = env.step_whites(step_index)
    else:
        reward_whites, reward_blacks, done, step_result = env.step_blacks(step_index)
    print(reward_whites)

    print(done)

    field = env.chess_game_whites.field

    image = renderer.render_field(field)
    cv2.imshow("qwe", image)
    cv2.waitKey(0)
