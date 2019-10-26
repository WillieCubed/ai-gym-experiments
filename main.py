"""The entrypoint for all experiments in this repository.

TODO: Implement a CLI to select experiments and modify params
"""
from openai_gym.experiments import play_cartpole_basic, play_cartpole_q_learning, play_breakout


def main():
    """Run all experiments sequentially."""
    play_cartpole_basic()
    play_cartpole_q_learning()
    play_breakout()


if __name__ == '__main__':
    main()
