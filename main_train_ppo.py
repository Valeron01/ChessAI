import os.path
import random

import numpy as np
from torch import nn
import torch

import tb_utils
from nn_modules.basic_transformer_model import BasicTransformerModel


def compute_returns(rewards, gamma, dones):
    result = np.zeros(len(rewards))
    cumulative_sum = 0
    for i in reversed(range(len(rewards))):
        if dones[i]:
            cumulative_sum = 0
        cumulative_sum = cumulative_sum * gamma + rewards[i]
        result[i] = cumulative_sum
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
        "dim_model": 384,
        "n_heads": 6,
        "dim_feedforward": 1024,
        "n_layers": 5,
        "n_layers_head": 2
    }
    model = BasicTransformerModel(**model_hparams)

    n_iterations = 10000000
    batch_size = 128
    lr = 3e-5
    n_epochs = 8 # Try a Different epoch count
    gamma = 0.95
    num_actions_to_collect = 4096
    epsilon = 0.2
    entropy_coefficient = 0.0001
    return_coefficient = 0.5

    env_params = {

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)
    n_episodes = 0
    for epoch in range(3350, n_iterations):
        if epoch == 3370:
            print("Changing LR")
            for g in optimizer.param_groups:
                g['lr'] = lr
        env = Game2048PPOWrapper()

        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        old_log_probs = []
        game_scores = []
        num_steps = []
        game_max_scores = []
        model = model.eval()
        game_iter = 0
        while True:
            state_tensor = torch.from_numpy(env.game.field).cuda()[None]
            with torch.no_grad():
                action, value = model(state_tensor)
            action_sample = action.sample()

            old_log_probs.append(action.log_prob(action_sample))
            states.append(state_tensor)
            actions.append(action_sample)

            reward, done = env.make_step(action_sample.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            if done:
                print(epoch, n_episodes, env.score, env.n_steps, env.max_value)
                num_steps.append(env.n_steps)
                game_scores.append(env.score)
                n_episodes += 1
                game_max_scores.append(env.max_value)
                env = Game2048PPOWrapper()

            game_iter += 1
            if game_iter >= num_actions_to_collect and done:
                break

        returns = compute_returns(rewards, gamma, dones)

        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        returns = torch.from_numpy(returns).cuda().float()

        old_log_probs = torch.cat(old_log_probs, 0)
        model = model.train()

        for i in range(num_actions_to_collect * n_epochs // batch_size):
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

        writer.add_scalar("mean_max_score", np.mean(game_max_scores), epoch)
        writer.add_scalar("max_max_score", np.max(game_max_scores), epoch)
        writer.add_scalar("min_max_score", np.min(game_max_scores), epoch)
        writer.add_scalar("mean_score", np.mean(game_scores), epoch)
        writer.add_scalar("max_score", np.max(game_scores), epoch)
        writer.add_scalar("games_played", len(game_scores), epoch)
        writer.add_scalar("n_steps_mean", np.mean(num_steps), epoch)


if __name__ == '__main__':
    main()