import os.path
import random

import numpy as np
from torch import nn
import torch
from tqdm import trange

import tb_utils
from chess_env import ChessEnv
from chess_game.chess_engine import StepResult
from chess_game.common_things import PieceColor
from nn_modules.basic_transformer_model import BasicTransformerModel


def compute_returns_per_env(rewards, gamma, dones, side_indices, last_values_whites, last_values_black):
    assert rewards.ndim == 2
    assert dones.ndim == 2
    returns_result = torch.zeros_like(rewards)
    returns_cumsum = torch.cat([last_values_whites[..., None] * 0, last_values_black[..., None] * 0], dim=-1)
    env_indices = torch.arange(returns_cumsum.shape[0])
    for frame_index in reversed(range(rewards.shape[0])):
        indices = side_indices[frame_index]
        returns_cumsum[env_indices, indices] *= 1 - dones[frame_index]

        returns_cumsum[env_indices, indices] = rewards[frame_index] + returns_cumsum[env_indices, indices] * gamma

        returns_result[frame_index] = returns_cumsum[env_indices, indices]

    return returns_result


def main_test_returns_computation():
    rewards = torch.FloatTensor([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
        [3, 0, 1],
        [-5, 1, 0]
    ])
    dones = torch.FloatTensor([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])
    side_indices = torch.IntTensor([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    returns = compute_returns_per_env(rewards, 0.9, dones, side_indices, torch.zeros([3]), torch.zeros([3]) + 1)
    print(returns)


def main():
    seed = 0
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    writer = tb_utils.build_logger(
        "./logs_ppo"
    )
    model_hparams = {
        "dim_model": 128,
        "n_heads": 2,
        "dim_feedforward": 256,
        "n_layers": 6,
        "n_layers_head": 2
    }
    device = "cuda:0"
    n_iterations = 10000000
    batch_size = 128
    lr = 6e-5
    n_epochs = 8 # Try a Different epoch count
    gamma = 0.8
    num_actions_to_collect = 4096
    epsilon = 0.2
    entropy_coefficient = 0.005
    return_coefficient = 2
    n_envs = 1
    model = BasicTransformerModel(**model_hparams).to(device)

    env_params = {
        "performed_reward": 2,
        "blocked_reward": -3,
        "terminate_iters": 256,
        "fifty_rule_steps": 30,
        "fifty_rule_penalty": -2,
        "rand_field_prob": 0.0
    }
    hparam_dict = {
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "lr": lr,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "num_actions_to_collect": num_actions_to_collect,
        "epsilon": epsilon,
        "entropy_coefficient": entropy_coefficient,
        "return_coefficient": return_coefficient,
        "model_class": str(model.__class__),
    }
    hparam_dict.update(env_params)
    hparam_dict.update(model_hparams)
    writer.add_hparams(hparam_dict, metric_dict={"default_hp": -1})

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    black_penalties_per_env = [0 for _ in range(n_envs)]
    white_penalty_per_env = [0 for _ in range(n_envs)]
    envs = [ChessEnv(**env_params) for _ in range(n_envs)]
    for epoch in range(0, n_iterations):
        states = []
        rewards = []
        actions = []
        values = []
        dones = []
        old_log_probs = []
        side_indices = []
        blocked = []

        model = model.eval()
        n_good_steps = 0
        n_taken_pieces_white = 0
        n_taken_pieces_black = 0

        for i_step in range(0, num_actions_to_collect, n_envs):
            states_per_env = []
            side_indices_per_inv = []
            for env_index, env in enumerate(envs):
                current_side = env.chess_game_whites.current_player_color
                if current_side == PieceColor.WHITE:
                    state = env.get_state_whites()
                    side_indices_per_inv.append(0)
                else:
                    state = env.get_state_blacks()
                    side_indices_per_inv.append(1)
                state = state[None]
                states_per_env.append(state)

            side_indices.append(side_indices_per_inv)
            states_per_env = torch.cat(states_per_env, 0)
            states.append(states_per_env)
            with torch.inference_mode():
                actions_per_env, values_per_env = model(states_per_env)

            actions_sampled = actions_per_env.sample()
            old_log_probs.append(actions_per_env.log_prob(actions_sampled))

            rewards_per_env = []
            dones_per_env = []

            for env_index, env in enumerate(envs):
                step_index = actions_sampled[env_index].item()
                current_side = env.chess_game_whites.current_player_color
                reward = 0
                if current_side == PieceColor.WHITE:
                    reward_whites, reward_blacks, done, step_result = env.step_whites(step_index)
                    reward = reward_whites + white_penalty_per_env[env_index]
                    black_penalty = reward_blacks
                    white_penalty_per_env[env_index] = 0
                else:
                    reward_blacks, reward_whites, done, step_result = env.step_blacks(step_index)
                    reward = reward_blacks + black_penalties_per_env[env_index]
                    white_penalty = reward_whites
                    black_penalties_per_env[env_index] = 0

                if step_result != StepResult.INVALID_MOVE:
                    n_good_steps += 1
                if step_result == StepResult.PERFORMED_KILL:
                    if current_side == PieceColor.WHITE:
                        n_taken_pieces_white += 1
                    else:
                        n_taken_pieces_black += 1
                blocked.append(step_result == StepResult.INVALID_MOVE)

                rewards_per_env.append(reward)
                dones_per_env.append(done)
                if done:
                    print("Env is done")
                    envs[env_index] = ChessEnv(**env_params)
                    white_penalty_per_env[env_index] = 0
                    black_penalties_per_env[env_index] = 0

            rewards.append(rewards_per_env)
            actions.append(actions_sampled)
            dones.append(dones_per_env)
            values.append(values_per_env)

        dones_per_env = dones[-1]
        last_values_white_per_env = []
        last_values_black_per_env = []
        for env_index, env in enumerate(envs):
            with torch.inference_mode():
                _, last_value_white = model(env.get_state_whites()[None].to(device))
                _, last_value_black = model(env.get_state_blacks()[None].to(device))
                last_value_black = last_value_black * (1 - dones_per_env[env_index])
                last_value_white = last_value_white * (1 - dones_per_env[env_index])

                last_values_white_per_env.append(last_value_white)
                last_values_black_per_env.append(last_value_black)
        last_values_white_per_env = torch.cat(last_values_white_per_env, 0).cpu()
        last_values_black_per_env = torch.cat(last_values_black_per_env, 0).cpu()
        states = torch.cat(states, 0).to(device)
        rewards = torch.from_numpy(np.float32(rewards))
        dones = torch.from_numpy(np.float32(dones))
        actions = torch.cat(actions, 0).to(device)
        old_log_probs = torch.cat(old_log_probs, 0).to(device)
        side_indices = torch.from_numpy(np.int32(side_indices))

        returns = compute_returns_per_env(
            rewards, gamma, dones, side_indices, last_values_white_per_env, last_values_black_per_env
        ).to(device).flatten(0, 1)

        model = model.train()
        for i in trange(num_actions_to_collect * n_epochs // batch_size):
            samples_indices = torch.randint(0, states.shape[0], [batch_size])

            states_batch = states[samples_indices]
            actions_batch = actions[samples_indices]
            returns_batch = returns[samples_indices]
            old_log_probs_batch = old_log_probs[samples_indices]

            predicted_actions, predicted_returns = model(states_batch)
            new_log_probs = predicted_actions.log_prob(actions_batch)
            ratios = (new_log_probs - old_log_probs_batch).exp()

            loss_returns = nn.functional.l1_loss(predicted_returns, returns_batch)
            clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
            advantages = returns_batch - predicted_returns.detach()

            advantages_log = advantages.detach().mean()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach()
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
            entropy_loss = -predicted_actions.entropy().mean()
            total_loss = policy_loss + loss_returns * return_coefficient + entropy_loss * entropy_coefficient

            writer.add_scalar("total_loss", total_loss, epoch)
            writer.add_scalar("policy_loss", policy_loss, epoch)
            writer.add_scalar("entropy_loss", entropy_loss, epoch)
            writer.add_scalar("advantages", advantages_log, epoch)
            writer.add_scalar("returns_loss", loss_returns, epoch)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

        if epoch % 10 == 0:
            target_path = os.path.join(writer.log_dir, "Checkpoints/Checkpoint.pt")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            torch.save(model, target_path)

        writer.add_scalar("mean_rewards", rewards.mean(), epoch)
        writer.add_scalar("mean_rewards_abs", rewards.abs().mean(), epoch)
        writer.add_scalar("good_steps_percentage", n_good_steps / num_actions_to_collect, epoch)
        writer.add_scalar("n_taken_pieces_white", n_taken_pieces_white, epoch)
        writer.add_scalar("n_taken_pieces_black", n_taken_pieces_black, epoch)


if __name__ == '__main__':
    # main_test_returns_computation()
    main()
