"""A collection of reinforcement learning experiments using OpenAI Gym."""
from collections import defaultdict
import gym


def play_cartpole_basic():
    """Play the "hello world" of machine learning: CartPole.

    This is directly from the OpenAI Gym Docs.
    """
    env = gym.make('CartPole-v0')
    env.reset()
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


def play_cartpole_q_learning():
    """Play a game of CartPole using deep-Q reinforcement learning.
    """
    Q = defaultdict(float)
    gamma = 0.99  # Discounting factor
    alpha = 0.5  # Soft update param

    env = gym.make('CartPole-v0')
    actions = env.action_space

    def update_Q(s, r, a, s_next, done):
        """Updates the current q value.

        This learns the action value (Q-value) and estimates the next action
        using the Bellman equation, estimating the next action by adopting the
        best Q value instead of following the current policy.

        TODO: Document parameters.
        """
        max_q_next = max([Q[s_next, action] for action in actions])
        # Do not include the next state's value if currently at the terminal state.
        Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])


def play_breakout():
    """TODO: Reimplement this."""
