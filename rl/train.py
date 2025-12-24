"""
Mini Isaac - RL Training Script

Train an agent to play Mini Isaac using PPO from stable-baselines3.
"""

import os
import sys
import argparse
from datetime import datetime

# Add current directory to path for env import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import MiniIsaacEnv

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("stable-baselines3 not installed. Install with:")
    print("  pip install stable-baselines3[extra]")


def train_ppo(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    save_path: str = "models",
    log_path: str = "logs",
    eval_freq: int = 10000,
):
    """Train a PPO agent on Mini Isaac."""
    if not SB3_AVAILABLE:
        print("Cannot train without stable-baselines3")
        return None

    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_mini_isaac_{timestamp}"

    # Create vectorized environment
    print(f"Creating {n_envs} parallel environments...")
    env = make_vec_env(MiniIsaacEnv, n_envs=n_envs)

    # Create eval environment
    eval_env = Monitor(MiniIsaacEnv())

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/{run_name}_best",
        log_path=f"{log_path}/{run_name}",
        eval_freq=eval_freq // n_envs,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=f"{save_path}/{run_name}_checkpoints",
        name_prefix="ppo_mini_isaac",
    )

    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"{log_path}/tensorboard/{run_name}",
    )

    # Train
    print(f"Training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{save_path}/{run_name}_final"
    model.save(final_path)
    print(f"Model saved to {final_path}")

    return model


def train_dqn(
    total_timesteps: int = 500_000,
    save_path: str = "models",
    log_path: str = "logs",
    eval_freq: int = 10000,
):
    """Train a DQN agent on Mini Isaac."""
    if not SB3_AVAILABLE:
        print("Cannot train without stable-baselines3")
        return None

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dqn_mini_isaac_{timestamp}"

    # Create environment
    env = Monitor(MiniIsaacEnv())
    eval_env = Monitor(MiniIsaacEnv())

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/{run_name}_best",
        log_path=f"{log_path}/{run_name}",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    # Create DQN agent
    print("Creating DQN agent...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=f"{log_path}/tensorboard/{run_name}",
    )

    # Train
    print(f"Training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{save_path}/{run_name}_final"
    model.save(final_path)
    print(f"Model saved to {final_path}")

    return model


def evaluate(model_path: str, n_episodes: int = 10, render: bool = False):
    """Evaluate a trained model."""
    if not SB3_AVAILABLE:
        print("Cannot evaluate without stable-baselines3")
        return

    # Load model
    if "dqn" in model_path.lower():
        model = DQN.load(model_path)
    else:
        model = PPO.load(model_path)

    # Create environment
    render_mode = "human" if render else None
    env = MiniIsaacEnv(render_mode=render_mode)

    rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if render:
                env.render()

        rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}, info={info}")

    env.close()

    print(f"\n--- Evaluation Results ({n_episodes} episodes) ---")
    print(f"Mean reward: {sum(rewards) / len(rewards):.2f}")
    print(f"Mean episode length: {sum(episode_lengths) / len(episode_lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate Mini Isaac RL agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo", help="Algorithm")
    train_parser.add_argument("--timesteps", type=int, default=500_000, help="Training timesteps")
    train_parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments (PPO only)")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("model", help="Path to model file")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    eval_parser.add_argument("--render", action="store_true", help="Render gameplay")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test environment with random actions")
    test_parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    test_parser.add_argument("--render", action="store_true", help="Render gameplay")

    args = parser.parse_args()

    if args.command == "train":
        if args.algo == "ppo":
            train_ppo(total_timesteps=args.timesteps, n_envs=args.n_envs)
        else:
            train_dqn(total_timesteps=args.timesteps)

    elif args.command == "eval":
        evaluate(args.model, n_episodes=args.episodes, render=args.render)

    elif args.command == "test":
        # Random agent test
        render_mode = "human" if args.render else None
        env = MiniIsaacEnv(render_mode=render_mode)

        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

                if args.render:
                    env.render()

            print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

        env.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
