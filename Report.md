# Deep Reinforcement Learning Nanodegree - Project 3: Collaboration and Competition

To solve the problem, I've used the deep deterministic policy gradient (DDPG) adjusted to a multi-agent case. The task is episodic,
and in order to solve the environment, the agent must get an average score of 0.5. The algorithm applies noise.

![robot](https://im3.ezgif.com/tmp/ezgif-3-f48d4f7c4653.gif)

## Method: Deep Deterministic Policy Gradient (DDPG)
I've implemented an off-policy method called Deep Deterministic Policy Gradient and described in the paper ![Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971). Basically, the algorithm learns a Q-function using Bellman equation and off-policy data. After data, the Q-function is used to learn the policy. I've used two agents.


### Neural Network
#### Actor

    Hidden: (input, 256) - ReLU
    Hidden: (256, 256) - ReLU
    Output: TanH


#### Critic

    Hidden: (input, 256) - ReLU
    Hidden: (256 + action_size, 256) - ReLU
    Output: Linear

### Hyperparameters
#### Agent
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```
#### Train
```
n_episodes = 100        # maximum number of training episodes number of training episodes
max_Ds = 1000           # max step in each episode
```

### Noise
```
MU = 0.0
THETA = 0.15
SIGMA = 0.1

```

### Plot of Rewards

The result is shown in the figure. The environment is solved in 466 episodes, average score of 30.01. 

![score](./scores_plot.png)

## Improvement

- An evolution to this project would be to use other models such as Twin Delayed DDPG (TD3), Asynchronous Actor-Critic Agents (A3C) and Distributed Distributional Deterministic Policy Gradients (D4PG).
- The neural network could also be depper.
- The hyperparameters can be fine tuned.
- In addition, we could experiment with adding more elements to the neural networks, such as dropout layers. That is mentioned on ![this paper](https://arxiv.org/abs/1509.02971)
- We could also implement Prioritized Experience Replay (see ![this paper](https://arxiv.org/abs/1511.05952)). This prioritize the more important experiences (i.e. those experiences from which the network can learn the most) so the algorithm becomes more stable and the should obtain the score of 0.5 in less episodes.

## References
- Udacity's ![model.py](https://github.com/udacity/deep-reinforcement-learning/blob/master/finance/model.py) from `finance/model.py`
- Udacity's ![model.py](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/model.py) `from ddpg-bipedal/model.py`

