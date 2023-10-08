# Reinforcement Learning Explained

## Introduction
Reinforcement Learning (RL) is a powerful machine learning paradigm that focuses on training agents to make sequential decisions in dynamic environments. It has applications in robotics, game playing, autonomous systems, and more. In this comprehensive guide, we will delve into RL, covering its principles, algorithms, and real-world applications.

## Table of Contents
1. [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
2. [Key Concepts in RL](#key-concepts-in-rl)
   - [Agent, Environment, and State](#agent-environment-and-state)
   - [Actions and Rewards](#actions-and-rewards)
   - [Policy](#policy)
3. [Markov Decision Processes (MDPs)](#markov-decision-processes-mdps)
   - [MDP Components](#mdp-components)
   - [Value Functions](#value-functions)
   - [Policy Evaluation and Improvement](#policy-evaluation-and-improvement)
4. [RL Algorithms](#rl-algorithms)
   - [Q-Learning](#q-learning)
   - [Deep Q-Networks (DQNs)](#deep-q-networks-dqns)
   - [Policy Gradient Methods](#policy-gradient-methods)
5. [Implementing RL](#implementing-rl)
   - [Using Python and Libraries](#using-python-and-libraries)
   - [Building RL Environments](#building-rl-environments)
   - [Training RL Agents](#training-rl-agents)
6. [Resources](#resources)
    - [Books](#books)
    - [Online Courses](#online-courses)
    - [Additional Reading](#additional-reading)

## Introduction to Reinforcement Learning
Machine learning can be divided into 3 main paradigms. These are supervised learning, unsupervised learning, and reinforcement learning. Most of you probably know a lot about supervised and unsupervised learning, but the third branch is just as important. Recently reinforcement learning has gathered lots of attention, and understanding its fundamentals properly is important. Reinforcement learning can be a little bit daunting to get into as even the fundamentals are a little bit complex.
Agent, Environment, State, Reward
<img width="479" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/a9965cb2-8921-43e0-8ba8-b056ba0b9a47">
Reinforcement learning models learn from an environment. The environment has a set of rules and is usually assumed to be deterministic. A reinforcement learning model interacts with the environment through an agent. The agent has a state in the environment, and the agent can perform actions that change the state of the agent in the environment.
Let’s look at a chess bot example: The environment is the chessboard, the agent is the chess bot, and the state of the environment is the position of all the pieces. Given the state of the chessboard, there are only a finite number of legal moves (actions) that can be made, these are determined by the environment. For example, the king can’t move the same way as the queen.
When the agent takes an action, the environment will receive this as an input and will output the resulting state and a reward (see diagram above). In the chess example, after the agent moves a piece the environment would return what the chessboard looks like after this move and the opponent’s move, this way it's the agent’s turn again. The environment will also return a reward, say if you capture a piece for example.
## Key Concepts in RL
Classical machine learning algorithms would really struggle to learn from the problem setting described above. So how can we teach a model to learn from an environment by interacting with it?
In other machine learning problems we usually start by defining a loss function, then we look to optimize. In reinforcement learning, we cannot immediately do that. To help us formulate a loss, we can start by looking at the rewards given back by the environment.
Going back to the chess example, clearly, we want the bot to win the chess game. However, it would be impractical to only reward the model when it wins the game (Markov Decision Process). The model would struggle to learn move by move, making the training process slow and possibly non-converging. We want short-term rewards too, this may be something like capturing a piece in the chess example (Dynamic Programming).
### Agent, Environment, and State
Let’s generalize what we have seen so far. An agent interacts with an environment through actions, these actions change the state of the environment. The goal of the model is to determine what actions will lead to the maximum reward.
To determine the best action, reinforcement learning works by estimating the value of actions. The value of an action indicates how good an action is, e.g. how good a chess move is. All of reinforcement learning revolves around this idea of estimating the optimal value function.
Value: The value of an action is defined as the sum of the immediate reward received by taking an action plus the expected value of the resulting state multiplied by a scaling term. In other words, the value of an action is how good the next state will be after taking that action, plus the expected future reward from that new state.
Reinforcement learning models update their value function by interacting with the environment, choosing an action, looking at the new state, looking at the reward then updating.
Aside from the value function, the model needs to learn a policy.
Policy: The policy of the algorithm is how it chooses what action to take based on the value of the current state.
Reinforcement learning algorithms want to evaluate states as best as possible (value function) to help them make decisions (policy) that lead to the maximum reward.
### Actions and Rewards
In reinforcement learning, rewards are associated with actions because they are used to guide the learning process of an agent. The agent learns to take actions that lead to high rewards and avoid actions that lead to low rewards. This is done through a process called trial and error, where the agent repeatedly takes actions and receives feedback in the form of rewards.The rewards are a way of providing feedback to the agent about the quality of its actions. They help the agent learn which actions are more likely to lead to successful outcomes, and which actions are less likely to do so.It is important to note that the rewards are associated with actions regardless of what state the agent ends up in because the ultimate goal of the agent is not to reach a specific state, but rather to maximize the total reward it receives over time. The agent is not necessarily trying to reach a specific state, but rather trying to find the best way to get the most rewards over time.
### Policy
So how can we choose what action to take based on the value of actions? One could define a greedy policy to always to always choose the action with the highest immediate reward. As I discussed earlier on, simply looking at immediate reward won't necessarily result in a long-term reward (in chess, always taking a piece will result in a higher immediate reward but might not be the best move). We need to take into account the expected reward of future states using the value function.
So maximizing immediate reward does not work, but what about a second policy that always takes the action with the highest value. Remember we are attempting to learn the value function. Taking the action that has the highest value will cause the model to get stuck in local minima. If our current value function is not optimal and we always choose the action with the highest value, then the model might never see actions that would result in much greater reward.
To improve the model’s estimation of the value function the model must explore. Optimizing the value function requires a fine balance between exploration and exploitation. Exploitation refers to taking the best action according to the value function (taking what we think is the best move). Exploration refers to taking a random action instead of the one recommended by the value function, this allows us to explore other states, adding randomness and improving the performance of the model.
Too much exploitation and the model will get stuck in local minima, too much exploration and the model won't converge at all.
## Markov Decision Processes (MDPs)
Markov Decision Processes (MDPs) are mathematical models used in reinforcement learning and decision-making under uncertainty. They provide a formal framework for modeling sequential decision-making problems where an agent interacts with an environment.

## MDP Components

MDPs consist of the following components:

### 1. States (S)

- States represent the possible situations or configurations in the environment. In an MDP, the set of all possible states is denoted as S.

### 2. Actions (A)

- Actions are the choices or decisions that the agent can make. The set of all possible actions is denoted as A.

### 3. Transition Probabilities (P)

- Transition probabilities describe the likelihood of transitioning from one state to another when an action is taken. Mathematically, P(s' | s, a) represents the probability of transitioning to state s' when action a is taken in state s.

### 4. Rewards (R)

- Rewards represent the immediate numerical feedback the agent receives after taking an action in a particular state. The reward function R(s, a, s') assigns a real number to each transition (s, a, s').

### 5. Discount Factor (γ)

- The discount factor (γ) is a value between 0 and 1 that determines the agent's preference for immediate rewards versus future rewards. It influences the agent's decision-making to balance short-term and long-term gains.

## Value Functions

Value functions are essential in MDPs for evaluating and comparing different policies and making decisions. There are two primary types of value functions:

### 1. State Value Function (V(s))

- The state value function, V(s), represents the expected cumulative reward an agent can obtain starting from a specific state s and following a policy π. It measures the long-term value of being in a particular state under a given policy.

### 2. Action Value Function (Q(s, a))

- The action value function, Q(s, a), represents the expected cumulative reward an agent can obtain starting from a specific state s, taking action a, and then following a policy π. It measures the long-term value of taking a particular action in a given state under a given policy.

## Policy Evaluation and Improvement

MDPs often involve finding an optimal policy that maximizes the expected cumulative reward over time. Two fundamental processes in solving MDPs are policy evaluation and policy improvement:

### Policy Evaluation

- Policy evaluation is the process of determining the value functions (V(s) or Q(s, a)) for a given policy π. It involves iteratively updating the value functions based on the Bellman equations until they converge.

### Policy Improvement

- Policy improvement is the process of finding a better policy by selecting actions that maximize the value function. It typically involves greedily selecting actions that lead to higher expected cumulative rewards according to the current value function.

In reinforcement learning, algorithms such as the Bellman equation, dynamic programming, and various Monte Carlo and temporal difference methods are used to solve MDPs and find optimal policies.

MDPs provide a foundational framework for modeling and solving decision-making problems in various domains, including robotics, game playing, finance, and healthcare.

## RL Algorithms
Explore common RL algorithms used for training agents.

### Q-Learning

Q-Learning is a fundamental reinforcement learning algorithm used to learn action-value functions (Q-functions) in Markov Decision Processes (MDPs). Key characteristics of Q-Learning include:

- **Value Iteration**: Q-Learning employs value iteration, where it estimates the optimal action-value function by iteratively updating Q-values based on observed rewards and transitions.

- **Off-Policy**: It is an off-policy algorithm, meaning it learns the optimal policy independently of the agent's behavior policy, making it suitable for exploration and exploitation.

- **Exploration Strategy**: Q-Learning typically uses ε-greedy exploration, where the agent chooses the action with the highest estimated Q-value with probability 1 - ε and explores randomly with probability ε.

- **Convergence**: Q-Learning converges to the optimal Q-values when certain conditions are met, such as sufficient exploration.

### Deep Q-Networks (DQNs)

Deep Q-Networks (DQNs) extend Q-Learning by using deep neural networks to approximate the action-value function. Key aspects of DQNs include:

- **Function Approximation**: DQNs employ deep neural networks to approximate the Q-function. This allows them to handle high-dimensional state spaces encountered in complex tasks.

- **Experience Replay**: DQNs use experience replay to store and randomly sample past experiences. This helps stabilize training by breaking temporal correlations in the data and mitigating catastrophic forgetting.

- **Target Network**: DQNs use a separate target network to stabilize the Q-value target during training. The target network's parameters are periodically updated to match the main network's parameters.

- **Loss Function**: DQNs use a loss function, such as the mean squared error loss, to minimize the difference between predicted Q-values and target Q-values.

### Policy Gradient Methods

Policy gradient methods are a family of reinforcement learning algorithms that directly optimize the agent's policy. Key characteristics of policy gradient methods include:

- **Policy Parameterization**: These methods parameterize the policy as a probability distribution over actions, typically using a neural network. The policy is then updated to maximize expected rewards.

- **Gradient Ascent**: Policy gradient methods use gradient ascent to update the policy parameters. The gradient of the expected cumulative reward is computed with respect to the policy parameters and used to guide policy updates.

- **Stochastic Policies**: These methods often employ stochastic policies, allowing the agent to explore the action space more effectively and handle uncertainty.

- **Policy Objective**: The policy is updated to maximize the expected return or a surrogate objective that encourages better policies. Common objectives include REINFORCE and PPO (Proximal Policy Optimization).

Policy gradient methods are particularly effective in environments with continuous action spaces and can handle both single-agent and multi-agent scenarios.

## Implementing RL
```python
import numpy as np

# Define the environment
class GridWorld:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.state_space = num_rows * num_cols
        self.action_space = 4  # 4 possible actions: Up, Down, Left, Right
        self.grid = np.zeros((num_rows, num_cols))  # Grid world

    def reset(self):
        # Reset the agent's position to a random state
        row = np.random.randint(0, self.num_rows)
        col = np.random.randint(0, self.num_cols)
        self.state = (row, col)
        return self.state

    def step(self, action):
        # Perform the selected action and return the next state and reward
        row, col = self.state

        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.num_rows - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.num_cols - 1, col + 1)

        self.state = (row, col)

        if row == 0 and col == 0:
            reward = 1  # Reached the goal
            done = True
        else:
            reward = 0  # Otherwise, no reward
            done = False

        return self.state, reward, done

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, state_space, action_space, alpha, gamma, epsilon):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((state_space, action_space))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )

# Initialize the environment and agent
env = GridWorld(num_rows=3, num_cols=4)
agent = QLearningAgent(state_space=env.state_space, action_space=env.action_space, alpha=0.1, gamma=0.9, epsilon=0.1)

# Training loop
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}")

# Test the trained agent
state = env.reset()
done = False

while not done:
    action = agent.select_action(state)
    next_state, _, done = env.step(action)
    state = next_state

print("Agent reached the goal!")
```

### Using Python and Libraries
```python
import numpy as np
import tensorflow as tf
import gym

# Define the Q-network model
class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.99
epsilon_initial = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
num_episodes = 1000

# Create the environment
env = gym.make('CartPole-v1')
state_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n

# Initialize the Q-networks
q_network = QNetwork(action_space_size)
target_network = QNetwork(action_space_size)
target_network.set_weights(q_network.get_weights())

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate)
huber_loss = tf.keras.losses.Huber()

# Function to select an action using epsilon-greedy policy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Choose a random action
    q_values = q_network(np.expand_dims(state, axis=0))
    return np.argmax(q_values)

# Training loop
epsilon = epsilon_initial

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        total_reward += reward

        # Calculate the target Q-value
        q_values_next = target_network(np.expand_dims(next_state, axis=0))
        target_q = reward + discount_factor * np.max(q_values_next)

        with tf.GradientTape() as tape:
            q_values = q_network(np.expand_dims(state, axis=0))
            selected_q = q_values[0][action]
            loss = huber_loss(selected_q, target_q)

        grads = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

        state = next_state

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, min_epsilon)

    # Update the target network weights periodically
    if episode % 10 == 0:
        target_network.set_weights(q_network.get_weights())
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Test the trained agent
state = env.reset()
total_reward = 0

while True:
    action = select_action(state, 0.0)  # Choose the best action (no exploration)
    next_state, reward, done, _ = env.step(action)

    total_reward += reward
    state = next_state

    env.render()

    if done:
        break

env.close()
```

## Resources

Find additional resources to enhance your understanding of Reinforcement Learning (RL).

### Books

- **"Reinforcement Learning: An Introduction"** by Richard S. Sutton and Andrew G. Barto - A widely recognized and comprehensive book on RL, covering theory and practical aspects.

- **"Deep Reinforcement Learning"** by Pieter Abbeel and John Schulman - This book focuses on deep RL, exploring the intersection of deep learning and RL.

- **"Reinforcement Learning"** by Csaba Szepesvári - A concise and practical book that provides insights into RL algorithms and applications.

### Online Courses

- **Coursera - "Reinforcement Learning Specialization"** - A specialization that includes courses on RL fundamentals, value-based methods, policy-based methods, and more.

- **edX - "Practical Deep Learning for Coders"** - An edX course that covers practical aspects of deep RL and its applications.

- **Coursera - "Practical Deep Learning for Recommender Systems"** - Learn how RL techniques are applied in recommender systems in this course.

### Additional Reading

- **[Reinforcement Learning (OpenAI)](https://openai.com/research/reinforcement-learning)** - A collection of research papers and articles on RL by OpenAI researchers.

- **[Deep Reinforcement Learning (Nature)](https://www.nature.com/articles/nature14236)** - This article in Nature discusses the combination of deep learning and RL and its impact.

- **[Introduction to Reinforcement Learning (Sutton and Barto)](http://incompleteideas.net/book/the-book-2nd.html)** - The online version of the book "Reinforcement Learning: An Introduction" by Sutton and Barto, available for free.

These resources cover a wide range of topics related to Reinforcement Learning (RL), from introductory materials to advanced concepts. Whether you're new to RL or looking to advance your knowledge in this field, these materials can be valuable for your journey into the world of RL.
