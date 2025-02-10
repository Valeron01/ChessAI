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


def compute_returns(rewards, gamma, dones, last_values):
    result = []
    cumulative_sum = last_values
    for i in reversed(range(len(rewards))):
        cumulative_sum = cumulative_sum * (1 - dones[i])
        cumulative_sum = cumulative_sum * gamma + rewards[i]
        result.append(cumulative_sum)
    result = torch.cat(result)
    result = torch.flip(result, [0])
    return result


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
        "dim_feedforward": 384,
        "n_layers": 5,
        "n_layers_head": 2
    }
    device = "cuda:0"
    n_iterations = 10000000
    batch_size = 128
    lr = 7e-5
    n_epochs = 6 # Try a Different epoch count
    gamma = 0.95
    num_actions_to_collect = 2048
    epsilon = 0.2
    entropy_coefficient = 0.001
    return_coefficient = 0.5
    n_envs = 32
    model = BasicTransformerModel(**model_hparams).to(device)

    env_params = {
        "performed_reward": 0.05,
        "blocked_reward": -1,
        "terminate_iters": 512,
        "fifty_rule_steps": 25,
        "fifty_rule_penalty": -6
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

    envs = [ChessEnv(**env_params) for _ in range(n_envs)]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    blacks_penalties = [0 for _ in range(n_envs)]
    white_penalties = [0 for _ in range(n_envs)]
    for epoch in range(0, n_iterations):
        print(f"Epoch {epoch}")
        states = []
        rewards = []
        actions = []
        values = []
        dones = []
        old_log_probs = []
        side_indices = []

        model = model.eval()
        n_good_steps = 0
        n_taken_pieces = 0
        for i_step in range(0, num_actions_to_collect, n_envs):
            states_per_env = []
            side_indices_per_env = []
            for env_index, env in enumerate(envs):
                env: ChessEnv
                current_side = env.chess_game_whites.current_player_color
                if current_side == PieceColor.WHITE:
                    state = env.get_state_whites()
                    side_indices_per_env.append(0)
                else:
                    state = env.get_state_blacks()
                    side_indices_per_env.append(1)
                states_per_env.append(state[None])
            side_indices.append(side_indices_per_env)

            states_per_env = torch.cat(states_per_env, dim=0).to(device)
            with torch.inference_mode():
                actions_per_env, values_per_env = model(states_per_env)
            actions_per_env_sampled = actions_per_env.sample()
            old_log_probs.append(actions_per_env.log_prob(actions_per_env_sampled)[None])
            actions_per_env_sampled = actions_per_env_sampled.cpu()

            rewards_per_env = []
            dones_per_env = []
            for env_index, env in enumerate(envs):
                env: ChessEnv
                current_side = env.chess_game_whites.current_player_color
                step_index = actions_per_env_sampled[env_index].item()
                if current_side == PieceColor.WHITE:
                    reward_whites, reward_blacks, done, step_result = env.step_whites(step_index)
                    blacks_penalties[env_index] = reward_blacks
                    total_reward = reward_whites + white_penalties[env_index]
                    white_penalties[env_index] = 0
                else:
                    reward_blacks, reward_whites, done, step_result = env.step_blacks(step_index)
                    white_penalties[env_index] = reward_whites
                    total_reward = reward_blacks + blacks_penalties[env_index]
                    blacks_penalties[env_index] = 0
                rewards_per_env.append(total_reward)
                dones_per_env.append(done)

                if step_result != StepResult.INVALID_MOVE:
                    n_good_steps += 1
                if step_result == StepResult.PERFORMED_KILL:
                    n_taken_pieces += 1
                if done:
                    print("Env is done")
                    envs[env_index] = ChessEnv(**env_params)
                    blacks_penalties[env_index] = 0
                    white_penalties[env_index] = 0

            rewards_per_env = torch.FloatTensor(rewards_per_env)
            dones_per_env = torch.FloatTensor(dones_per_env)

            rewards.append(rewards_per_env)
            states.append(states_per_env[None])
            actions.append(actions_per_env_sampled[None])
            dones.append(dones_per_env[None])
            values.append(values_per_env[None])

        # Last step evaluation
        states_per_env = []
        for env_index, env in enumerate(envs):
            env: ChessEnv
            current_side = env.chess_game_whites.current_player_color
            if current_side == PieceColor.WHITE:
                state = env.get_state_whites()
            else:
                state = env.get_state_blacks()
            states_per_env.append(state[None])

        states_per_env = torch.cat(states_per_env, dim=0).to(device)
        with torch.inference_mode():
            _, last_values_per_env = model(states_per_env)

        states = torch.cat(states, 0).to(device)
        rewards = torch.from_numpy(np.float32(rewards))
        dones = torch.from_numpy(np.float32(dones))
        actions = torch.cat(actions, 0).to(device)
        old_log_probs = torch.cat(old_log_probs, 0).to(device)

        last_values_per_env = last_values_per_env.cpu() * (1 - dones[-1])
        side_indices = torch.from_numpy(np.int32(side_indices))
        rewards_whites = rewards[side_indices == 0]
        rewards_blacks = rewards[side_indices == 1]
        print(side_indices.shape)
        print(rewards.shape)
        print(rewards_whites.shape)
        rewards_whites[-1] += torch.FloatTensor(white_penalties)
        rewards_blacks[-1] += torch.FloatTensor(blacks_penalties)
        assert False
        returns = compute_returns(
            rewards, gamma, dones, last_values_per_env
        ).to(device)

        model = model.train()
        states = states.flatten(0, 1)
        returns = returns.flatten(0, 1)
        actions = actions.flatten(0, 1)
        old_log_probs = old_log_probs.flatten(0, 1)

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
        writer.add_scalar("n_taken_pieces", n_taken_pieces, epoch)


if __name__ == '__main__':
    main()