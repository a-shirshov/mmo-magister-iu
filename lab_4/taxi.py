import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange

torch.manual_seed(4)

env = gym.make('Taxi-v3')

def policy_evaluation(env: gym.Env, policy: torch.Tensor, gamma: float, threshold: float, max_iterations: int):
    V = torch.zeros(env.observation_space.n)
    delta = float("inf")
    iterations = 0

    while delta >= threshold and iterations < max_iterations:
        delta = 0  # Reset delta for this iteration
        V_tmp = torch.zeros_like(V)
        for state in range(env.observation_space.n):
            v = 0  # Accumulate the value for this state
            for action, act_prob in enumerate(policy[state]):
                for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                    v += act_prob * trans_prob * (reward + gamma * V[new_state])
            V_tmp[state] = v
            delta = max(delta, torch.abs(V_tmp[state] - V[state]).item())
        V = V_tmp.clone()
        iterations += 1

    if iterations >= max_iterations:
        print("Policy evaluation reached the maximum number of iterations and may not have converged.")

    return V

def policy_improvement(env: gym.Env, V: torch.Tensor, gamma: float):
    policy = torch.zeros((env.observation_space.n, env.action_space.n))
    
    for state in range(env.observation_space.n):
        v_action = torch.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_action[action] += trans_prob * (reward + gamma * V[new_state])
        best_action = torch.argmax(v_action).item()
        policy[state][best_action] = 1.0
    return policy

def policy_iteration(env: gym.Env, gamma: float, threshold: float, max_iterations: int):
    policy = torch.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    iterations = 0

    while True:
        improved_value = policy_evaluation(env, policy, gamma, threshold, max_iterations)
        improved_policy = policy_improvement(env, improved_value, gamma)

        if torch.equal(policy, improved_policy) or iterations >= max_iterations:
            if iterations >= max_iterations:
                print("Policy iteration reached the maximum number of iterations and may not have converged.")
            break

        policy = improved_policy.clone()
        iterations += 1

    return improved_policy, improved_value

# Increase the maximum number of iterations and adjust the threshold
max_iterations = 10000  # Increased maximum number of iterations
threshold = 1e-3  # Adjusted threshold for convergence

optimal_policy, optimal_value = policy_iteration(env, 0.99, threshold, max_iterations)
print('Optimal values:\n{}'.format(optimal_value))
print('Optimal policy:\n{}'.format(optimal_policy))
