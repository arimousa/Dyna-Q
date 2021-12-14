import itertools
import sys
import numpy as np
from collections import defaultdict, namedtuple

from gridworld import GridworldEnv

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
      Q: A dictionary that maps from state -> action-values.
        Each value is a numpy array of length nA (see below)
      epsilon: The probability to select a random action . float between 0 and 1.
      nA: Number of actions in the environment.

    Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.random.choice(np.flatnonzero(Q[observation] == Q[observation].max()))
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def finding_q(Q, state, next_state, action, alpha, reward, done, discount_factor):
    next_action = np.argmax(Q[next_state])
    Q[state][action] += alpha * (
            reward + (1 - done) * discount_factor * Q[next_state][next_action] - Q[state][action])
    return Q


def dyna_q_learning(env, num_episodes, discount_factor=0.95, alpha=0.1, epsilon=0.1, n=50):
    """
    Dyna-Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
      env: environment.
      num_episodes: Number of episodes to run for.
      discount_factor: Lambda time discount factor.
      alpha: TD learning rate.
      epsilon: Chance the sample a random action. Float betwen 0 and 1.
      n: number of planning steps

    Returns:
      A tuple (Q, episode_lengths).
      Q is the optimal action-value function, a dictionary mapping state -> action values.
      stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))

    # The model.
    # A nested dictionary that maps state -> (action -> (next state, reward, terminal flag)).
    M = defaultdict(lambda: np.zeros((env.nA, 3)))

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)
    previous_sa = []

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        for i in itertools.count():
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = i

            Q = finding_q(Q, state, next_state, action, alpha, reward, done, discount_factor)
            M[state][action] = [next_state, reward, done]
            if (state, action) not in previous_sa:
                previous_sa.append((state, action))

            for j in range(n):
                img_state, img_action = previous_sa[np.random.choice(len(previous_sa))]
                img_next_state, img_reward, img_done = M[img_state][img_action]
                Q = finding_q(Q, img_state, img_next_state, img_action, alpha, img_reward,
                              img_done, discount_factor)

            if done:
                break
            state = next_state
    return Q, stats


if __name__ == "__main__":
    np.random.seed(0)
    env = GridworldEnv()
    Q, stats = dyna_q_learning(env, 1000)

    print("")
    for k, v in Q.items():
        print("%s: %s" % (k, v.tolist()))
