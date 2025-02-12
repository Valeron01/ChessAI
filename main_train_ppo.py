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
        "n_heads": 4,
        "dim_feedforward": 256,
        "n_layers": 5,
        "n_layers_head": 1
    }
    device = "cuda:0"
    n_iterations = 10000000
    batch_size = 128
    lr = 8e-5
    n_epochs = 8 # Try a Different epoch count
    gamma = 0.8
    num_actions_to_collect = 4096
    epsilon = 0.2
    entropy_coefficient = 0.01
    return_coefficient = 0.5
    model = BasicTransformerModel(**model_hparams).to(device)

    env_params = {
        "performed_reward": 0.02,
        "blocked_reward": -5,
        "terminate_iters": 256,
        "fifty_rule_steps": 10,
        "fifty_rule_penalty": -0.2,
        "rand_field_prob": 0.2
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

    env = ChessEnv(**env_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    black_penalty = 0
    white_penalty = 0
    for epoch in range(0, n_iterations):
        print(f"Epoch {epoch}")
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
        for i_step in range(0, num_actions_to_collect):
            current_side = env.chess_game_whites.current_player_color
            if current_side == PieceColor.WHITE:
                state = env.get_state_whites()
                side_indices.append(0)
            else:
                state = env.get_state_blacks()
                side_indices.append(1)
            state = state[None]
            states.append(state)
            with torch.inference_mode():
                action, value = model(state)

            action_sampled = action.sample()
            old_log_probs.append(action.log_prob(action_sampled))

            step_index = action_sampled.item()
            reward = 0
            if current_side == PieceColor.WHITE:
                reward_whites, reward_blacks, done, step_result = env.step_whites(step_index)
                reward = reward_whites + white_penalty
                black_penalty = reward_blacks
                white_penalty = 0
            else:
                reward_blacks, reward_whites, done, step_result = env.step_blacks(step_index)
                reward = reward_blacks + black_penalty
                white_penalty = reward_whites
                black_penalty = 0

            if step_result != StepResult.INVALID_MOVE:
                n_good_steps += 1
            if step_result == StepResult.PERFORMED_KILL:
                if current_side == PieceColor.WHITE:
                    n_taken_pieces_white += 1
                else:
                    n_taken_pieces_black += 1
            blocked.append(step_result == StepResult.INVALID_MOVE)
            if done:
                print("Env is done")
                env = ChessEnv(**env_params)
                white_penalty = 0
                black_penalty = 0

            rewards.append(reward)
            actions.append(action_sampled)
            dones.append(done)
            values.append(value)

        with torch.inference_mode():
            _, last_value_white = model(env.get_state_whites()[None].to(device))
            _, last_value_black = model(env.get_state_blacks()[None].to(device))
            last_value_black = last_value_black * (1 - dones[-1])
            last_value_white = last_value_white * (1 - dones[-1])

        states = torch.cat(states, 0).to(device)
        rewards = torch.from_numpy(np.float32(rewards))
        dones = torch.from_numpy(np.float32(dones))
        actions = torch.cat(actions, 0).to(device)
        old_log_probs = torch.cat(old_log_probs, 0).to(device)
        blocked = torch.FloatTensor(blocked)

        side_indices = torch.from_numpy(np.int32(side_indices))
        rewards_whites = rewards[side_indices == 0]
        rewards_blacks = rewards[side_indices == 1]
        returns = torch.zeros_like(rewards).to(device)
        if rewards_whites.shape[0] != 0:
            rewards_whites[-1] += white_penalty

            returns_whites = compute_returns(
                rewards_whites.cpu(), gamma, dones, last_value_white.cpu()
            ).to(device)
            returns[side_indices == 0] = returns_whites
        if rewards_blacks.shape[0] != 0:
            rewards_blacks[-1] += black_penalty
            returns_blacks = compute_returns(
                rewards_blacks.cpu(), gamma, dones, last_value_black.cpu()
            ).to(device)
            returns[side_indices == 1] = returns_blacks

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
        writer.add_scalar("mean_reward_white", rewards_whites.mean(), epoch)
        writer.add_scalar("mean_reward_black", rewards_blacks.mean(), epoch)


if __name__ == '__main__':
    main()