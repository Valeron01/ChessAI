import os.path
import random

import numpy as np
import yaml
from torch import nn
import torch
from tqdm import trange

import tb_utils
from chess_env import ChessEnv
from nn_modules.basic_transformer_model import BasicTransformerModel


def compute_returns_per_env(rewards, gamma, dones, return_masks):
    assert rewards.ndim == 2
    assert dones.ndim == 2
    assert return_masks.ndim == 2
    returns_result = torch.zeros_like(rewards)
    returns_cumsum = 0
    for frame_index in reversed(range(rewards.shape[0])):
        returns_cumsum *= 1 - dones[frame_index]
        returns_cumsum = rewards[frame_index] * (1 - return_masks[frame_index]) + returns_cumsum * gamma
        returns_result[frame_index] = rewards[frame_index] * return_masks[frame_index] + returns_cumsum * (1 - return_masks[frame_index])

    return returns_result


def main():
    seed = 2
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    writer = tb_utils.build_logger(
        "./logs_ppo"
    )
    model_hparams = {
        "dim_model": 128,
        "n_heads": 8,
        "dim_feedforward": 256,
        "n_layers": 4,
        "n_layers_head": 1,
        "input_dim": 96
    }
    device = "cuda:0"
    n_iterations = 10000000
    batch_size = 128
    lr = 6e-5
    n_epochs = 2  # Try a Different epoch count
    gamma = 0.9
    num_actions_to_collect = 2048
    epsilon = 0.13
    entropy_coefficient = 0.06
    return_coefficient = 0.5
    n_env_pairs = 8
    model = BasicTransformerModel(**model_hparams).to(device)

    env_params = {
        "performed_reward": -0.01,
        "blocked_reward": -1,
        "terminate_iters": 128,
        "fifty_rule_steps": 25,
        "fifty_rule_penalty": -2,
        "rand_field_prob": 0.1,
        "n_bad_steps_to_terminate": 5
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
    with open(os.path.join(writer.log_dir, "hyper_parameters.yaml"), "w") as f:
        yaml.dump(hparam_dict, f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    envs = []
    for _ in range(n_env_pairs):
        env = ChessEnv(**env_params)
        envs.append(env)
    opponent_rewards = [0 for _ in range(n_env_pairs)]
    for epoch in range(0, n_iterations):
        env_logs = []
        states = []
        rewards = []
        actions = []
        values = []
        dones = []
        terminates = []
        old_log_probs = []
        return_masks = []

        model = model.eval()
        for i_step in range(0, num_actions_to_collect // n_env_pairs):
            states_per_env_first_team = torch.cat([env.state()[None] for env in envs], 0).to(device)
            with torch.inference_mode():
                distributions_per_env_first_team, values_per_env_first_team = model(states_per_env_first_team)
            actions_sampled_per_env_first_team = distributions_per_env_first_team.sample()
            old_log_probs_per_env_first_team = distributions_per_env_first_team.log_prob(actions_sampled_per_env_first_team)

            rewards_per_env_first_team = []
            terminates_per_env_first_team = []
            dones_per_env_first_team = []
            actions_sampled_per_env_first_team = actions_sampled_per_env_first_team.cpu()
            return_masks_per_env = []
            for env_index in range(n_env_pairs):
                env = envs[env_index]
                reward, opponent_reward, terminated, done, return_mask = env.step(actions_sampled_per_env_first_team[env_index].item())
                reward = reward + opponent_rewards[env_index]
                opponent_rewards[env_index] = opponent_reward
                rewards_per_env_first_team.append(reward)
                terminates_per_env_first_team.append(terminated)
                dones_per_env_first_team.append(done)
                return_masks_per_env.append(return_mask)

            # ####################### Second Team
            states_per_env_second_team = torch.cat([env.state()[None] for env in envs], 0).to(device)
            with torch.inference_mode():
                distributions_per_env_second_team, values_per_env_second_team = model(states_per_env_second_team)
            actions_sampled_per_env_second_team = distributions_per_env_second_team.sample()
            old_log_probs_per_env_second_team = distributions_per_env_second_team.log_prob(actions_sampled_per_env_second_team)

            rewards_per_env_second_team = []
            terminates_per_env_second_team = []
            dones_per_env_second_team = []
            actions_sampled_per_env_second_team = actions_sampled_per_env_second_team.cpu()
            for env_index in range(n_env_pairs):
                env = envs[env_index]
                reward, opponent_reward, terminated, done, return_mask = env.step(
                    actions_sampled_per_env_second_team[env_index].item())
                reward = reward + opponent_rewards[env_index]
                opponent_rewards[env_index] = opponent_reward
                rewards_per_env_second_team.append(reward)
                terminates_per_env_second_team.append(terminated)
                dones_per_env_second_team.append(done)
                return_masks_per_env.append(return_mask)

            rewards_per_env = rewards_per_env_first_team + rewards_per_env_second_team
            terminates_per_env = terminates_per_env_first_team + terminates_per_env_second_team
            dones_per_env = dones_per_env_first_team + dones_per_env_second_team

            rewards.append(torch.FloatTensor(rewards_per_env)[None])
            terminates.append(torch.FloatTensor(terminates_per_env)[None])
            dones.append(torch.FloatTensor(dones_per_env)[None])
            values.append(torch.cat([values_per_env_first_team, values_per_env_second_team], 0)[None])
            states.append(torch.cat([states_per_env_first_team, states_per_env_second_team], 0)[None])
            actions.append(torch.cat([actions_sampled_per_env_first_team, actions_sampled_per_env_second_team], 0)[None])
            old_log_probs.append(torch.cat([old_log_probs_per_env_first_team, old_log_probs_per_env_second_team], 0)[None])
            return_masks.append(torch.FloatTensor(return_masks_per_env)[None])

            for env_index in range(n_env_pairs):
                if dones_per_env_first_team[env_index] and dones_per_env_second_team[env_index]:
                    env_logs.append(envs[env_index])
                    envs[env_index] = ChessEnv(**env_params)
                    opponent_rewards[env_index] = 0

        total_steps = 0
        n_good_steps = 0
        n_taken_pieces_white = 0
        n_taken_pieces_black = 0
        for i in envs + env_logs:
            n_good_steps += i.good_steps
            n_taken_pieces_white += len(i.chess_game.dead_blacks)
            n_taken_pieces_black += len(i.chess_game.dead_whites)
            total_steps += i.steps_made
        mean_lifetime = total_steps / (len(envs) + len(env_logs))

        rewards = torch.cat(rewards, 0)
        terminates = torch.cat(terminates, 0)
        values = torch.cat(values, 0)
        dones = torch.cat(dones, 0)
        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        old_log_probs = torch.cat(old_log_probs, 0)
        return_masks = torch.cat(return_masks, 0)

        returns = compute_returns_per_env(
            rewards, gamma, dones, return_masks
        ).to(device)

        # print(rewards.shape)
        # print(terminates.shape)
        # print(dones.shape)
        # print(states.shape)
        # print(actions.shape)
        # print(old_log_probs.shape)
        # print(returns.shape)
        # assert False

        states = states.flatten(0, 1).to(device)
        actions = actions.flatten(0, 1).to(device)
        returns = returns.flatten(0, 1).to(device)
        old_log_probs = old_log_probs.flatten(0, 1).to(device)
        values = values.flatten(0, 1).to(device)
        return_masks = return_masks.flatten(0, 1).to(device)

        model = model.train()
        for _ in trange(num_actions_to_collect * n_epochs // batch_size):
            samples_indices = torch.randint(0, states.shape[0], [batch_size])

            states_batch = states[samples_indices]
            actions_batch = actions[samples_indices]
            returns_batch = returns[samples_indices]
            old_log_probs_batch = old_log_probs[samples_indices]
            values_batch = values[samples_indices]
            return_masks_batch = return_masks[samples_indices]
            return_masks_batch_inv = 1 - return_masks_batch

            predicted_actions, predicted_returns = model(states_batch)
            new_log_probs = predicted_actions.log_prob(actions_batch)
            ratios = (new_log_probs - old_log_probs_batch).exp()

            loss_returns = nn.functional.mse_loss(predicted_returns * return_masks_batch_inv, returns_batch * return_masks_batch_inv)
            clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
            advantages = returns_batch - values_batch.detach()
            advantages = advantages.detach()

            advantages_log = advantages.mean()
            advantages_std = (advantages.std() + 1e-8)
            advantages = (advantages - advantages.mean()) / advantages_std
            advantages_max = advantages.abs().max()

            advantages = advantages * return_masks_batch_inv + (returns_batch * return_masks_batch) * advantages_max

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
            torch.nn.utils.clip_grad_norm_(model.transformer.parameters(), max_norm=50)
            optimizer.step()

        if epoch % 10 == 0 and epoch != 0:
            target_path = os.path.join(writer.log_dir, "Checkpoints/Checkpoint.pt")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            torch.save(model, target_path)

        writer.add_scalar("mean_rewards", rewards.mean(), epoch)
        writer.add_scalar("mean_rewards_abs", rewards.abs().mean(), epoch)
        writer.add_scalar("good_steps_percentage", n_good_steps / total_steps, epoch)
        writer.add_scalar("n_taken_pieces_white", n_taken_pieces_white, epoch)
        writer.add_scalar("n_taken_pieces_black", n_taken_pieces_black, epoch)
        writer.add_scalar("mean_returns", returns.mean(), epoch)
        writer.add_scalar("max_returns", returns.max(), epoch)
        writer.add_scalar("min_returns", returns.min(), epoch)
        writer.add_scalar("mean_lifetime", mean_lifetime, epoch)



if __name__ == '__main__':
    # main_test_returns_computation()
    main()
