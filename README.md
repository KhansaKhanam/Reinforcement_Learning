# Reinforcement_Learning

Simply put, RL is a framework that gives our models (more commonly referred to as agents in RL) the ability to make “intelligent” decisions that help them achieve their goal which is approximately this one:

It’s super common to see a diagram like this one explaining, on a high level, what the framework is all about:
<img width="812" alt="Screenshot 2024-09-27 at 00 11 14" src="https://github.com/user-attachments/assets/33f30708-4d7b-46e1-93f8-f5316b0a962d">

You have an agent interacting with the environment. It makes some actions and the environment sends back the reward for that particular action and the next state of the environment.

To learn more about Reinforcement Learning check out the following references:
* https://gordicaleksa.medium.com/how-to-get-started-with-reinforcement-learning-rl-4922fafeaf8c
* https://www.davidsilver.uk/teaching/
* https://python-data-science.readthedocs.io/en/latest/reinforcement.html
* https://ubuffalo-my.sharepoint.com/personal/avereshc_buffalo_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Favereshc%5Fbuffalo%5Fedu%2FDocuments%2F2024%5FFall%5FRL%2F%5Fpublic%2FCourse%20Materials%2FRL%20Environment%20Visualization&ga=1
* https://ubuffalo-my.sharepoint.com/personal/avereshc_buffalo_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Favereshc%5Fbuffalo%5Fedu%2FDocuments%2F2024%5FFall%5FRL%2F%5Fpublic%2FCourse%20Materials%2FRL%20Environment%20Demo&ga=1
* https://huggingface.co/learn/deep-rl-course/unit2/introduction
* https://medium.com/data-science-in-your-pocket/how-to-create-a-custom-openai-gym-environment-with-codes-fb5de015de3c
* https://medium.com/@paulswenson2/an-introduction-to-building-custom-reinforcement-learning-environment-using-openai-gym-d8a5e7cf07ea
* https://www.datacamp.com/tutorial/reinforcement-learning-python-introduction
* https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187
* https://medium.com/@taeefnajib/hyperparameter-tuning-using-optuna-c46d7b29a3e
* https://medium.com/@qempsil0914/deep-q-learning-part2-double-deep-q-network-double-dqn-b8fc9212bbb2

### Defining RL Environments
### Environment Scenario in consideration:

### 1. Warehouse Robot

*Scenario:* 
A robot operates in a warehouse grid where its task is to pick up items from specified locations and deliver them to designated drop-off points. The warehouse has shelves that act as obstacles, and the
robot must navigate around them efficiently.

*Environment setup:*
• Grid size: 6x6 grid
• Obstacles: Shelves (static obstacles) that block the robot’s path
• Goal: The robot must pick up an item from an assigned location and deliver it to another specified location
• Actions: Up, Down, Left, Right, Pick-up, Drop-off
• Rewards:
* +20pts for successful delivery
* -1pt for each step taken to encourage efficiency
* -20pts for hitting an obstacle
• Terminal state: The robot successfully delivers the item
• Deterministic Setup: All the obstacles, drop-off points, and the outcome of the agent’s actions are fixed.
• Stochastic Setup: The outcomes of an agent’s actions are not fixed. For, if the agent takes the action to go right, the agent ends up in the grid block on the right with a probability of 0.9 (90%) but stays in the same grid block with a probability of 0.1 (10%).

*Environment Setup:*

This piece of code initiates the environment’s setup,
We’ve initialized the space to a 6*6 grid with the start position at [0,0] and the end/terminal position at [5,5]. The agent incurs a penalty of -1 for every step it takes towards the end position only to make sure it takes the least number of steps to reach the goal.
* We’ve set the shelves at positions [(0, 4), (2, 0), (2, 2), (3, 5), (4, 1)]. Upon hitting the shelf the agent receives a penalty of -20.
* We’ve declared the following actions: “DOWN” for 0, “UP” for 1,” RIGHT” for 2, “LEFT” for 3, “PICK” for 4 and “DROP” for 5. 
* We’ve set the maximum timesteps to 10 to make it feasible for the agent to reach the target with an optimal number of steps. 
* We’re setting specific values to the space to make it indicative of the specific positions of the entities like agents, shelves, pick-up locations, and drop-off locations. 

*Environment reset():*
The class reset() method is quite an essential one especially when the agent hits the terminal position or is truncated within the space. In such a situation we adjust the agent’s position back to the start and retract all variables(reward, pick_up,drop_off) back to the initial state to restart the training and start anew.

*Agent training through step():*
This method is the heart of our program and is essential for training the agent on how it should perform and interact with the environment. 

By default, we’ve set the environment type to ‘Deterministic’ where the agent takes the action as desired. It is called deterministic cause we can determine the actions the agent can take. The desired actions are listed as valid_actions. The actions the agent is allowed to take are going up, going down, going right or left, picking up an object, and dropping off the object at the goal position.

Consecutively, a reward/penalty is assigned for every move the agent makes. 
The following is a summary of the reward system of my model:
+5 upon picking up the object 
-1 for every move the agent makes 
+20 for successfully dropping off at the goal position 
-20 for every time the agent hits an obstacle 

np.clip() method also makes sure the agent navigates only within the bounds of the space and not outside of it. If the agent reaches the terminal position or gets truncated it will be sent back to the initial position. The goal is to make the agent make successful deliveries from the pick-up spot with the least number of steps.

In a “Stochastic” setting, however, the agent’s actions are not quite determined and the agent ends up taking any action from the list of actions. The model allows for a random probability to be chosen between 0 and 1 and if the random number is less than 0.1 it will stay in the same position else will take the desired action. The “Stochastic” setting would require one to keep a tab on the agent’s action and the observation space as the agent’s actions are undeterministic. The goal of “Stochastic” learning would be to conceive of the idea of unpredictability.

*Rendering the Space:*
The render() function gives a more graphical view of the agent space.

* How did you define the stochastic environment? 
The environment type is defined by the user and is by default set to “Deterministic”. The “Stochastic” environment behaves based on probability, here for my model I have defined the agent to take the intended action if the probability is more than 0.1 (else) and stay in the same position otherwise. 

* What is the difference between the deterministic and stochastic environments? 

A deterministic environment is one where the agent's actions always lead to a predictable outcome. The set of available actions is predefined, and each action results in a specific, expected environmental change. In other words, there’s no randomness in how actions are executed.

For example, deciding to submit a checkpoint is deterministic — you either submit it or don’t, with no uncertainty involved (I chose to submit, of course!).

On the other hand, a stochastic environment involves randomness, meaning the agent's actions don’t always lead to the same outcome. Each action might have a range of possible results based on probabilities. An example of this would be deciding what to do with a pizza — you could eat it, share it with a friend, or simply take pictures for social media. The action isn’t fixed, and multiple outcomes are possible, much like how randomness affects decisions in a stochastic environment.

* Safety in AI: Write a brief review explaining how you ensure the safety of your environments. E.g. how do you ensure that the agent chooses only actions that are allowed, that the agent is navigating within defined state-space, etc?
To ensure the safety of AI environments, constraints are placed on both the actions and state space of the agent. 
First, the agent's action space is restricted to predefined, valid actions, ensuring it cannot choose actions outside of what is allowed. 
For navigation, boundary conditions are enforced, so the agent remains within the defined grid or state-space limits (e.g., using functions like np. clip to keep positions within bounds). 
Additionally, checks are implemented to prevent the agent from interacting with forbidden states or triggering unsafe behaviors. 
These constraints, along with regular environment resets upon task completion or invalid actions, ensure the agent operates safely and predictably.

### 2. Applying Tabular Methods

### 2.1 Q-learning

Here we have initialized the Q-Learning code with the following parameters:
* Qtable - This will be the q-table that will be modified to obtain the best possible action 
* env_type='deterministic' - By default this is set to ‘deterministic’ and defines the environment type.
* n_episodes=1000 - defines the number of episodes over which the tuning happens, 1000 episodes gives the model the space to explore initially and then reduce to exploitation.
* max_iter_per_episode=100 - the maximum number of times the q-table gets updated per episode 
* epsilon_values=[] - this list stores the epsilon value that gets altered and reduced in every iteration.
* epsilon=0.99 - The initial epsilon value
* epsilon_decay_rate=0.01 - we reduce the epsilon by 0.01 in every iteration
* min_epsilon=0.01 - defines the minimum value the epsilon can attain           
* alpha=0.1 - initializing alpha value
* gamma=0.6 - initializing gamma value
* rewards=[] - list to store the reward obtained after every episode

### This is the standard formula that updates the q-table: 

qtable[current_state, action] = qtable[current_state, action] + alpha * (reward + gamma * np.max(qtable[obs, :]) - qtable[current_state, action])

The total reward based on the best course of action is computed per episode and the reward list is updated with it. Similarly, the epsilon decay is applied and the new epsilon value is appended to the epsilon_values list at the end of every episode. Epsilon starts high to encourage exploration and decays over time to favor the exploitation of the learned policy.
Initially, the total reward per episode will fluctuate, reflecting the agent's exploration. As the agent converges to the optimal policy (exploitation), the total reward should stabilize and increase, indicating better performance. 

Stochasticity introduces randomness into state transitions, making the environment more difficult to predict. The reward graph reflects the agent's performance despite the randomness.
When training the agent using Q-learning in a stochastic environment, the total reward per episode may be more variable compared to the deterministic case due to the randomness. It might take longer for the rewards to stabilize as the agent needs to adjust for unpredictability.

### Double Q-Learning:
I’ve applied Double Q-Learning as my choice of tabular method. This approach is designed to mitigate the overestimation bias of traditional Q-learning by maintaining two separate Q-tables. By maintaining two Q-tables, the function reduces the risk of overestimating the value of certain actions, which is a common pitfall in traditional Q-learning approaches. 
The use of an epsilon-greedy strategy balances exploration and exploitation, which is crucial for effective learning in reinforcement learning environments. 
The random selection of which Q-table to update introduces a degree of variability that can lead to more generalized learning.

### Double Q-Learning in a Deterministic Environment: 
In a deterministic environment, the outcomes of actions are predictable. Whenever the agent acts in a given state, it will always transition to the same next state and receive the same reward. This type of environment is well-suited for algorithms like Q-Learning or Double Q-Learning, as the agent can easily learn the optimal policy without having to deal with randomness.

### Double Q-Learning in a Stochastic Environment: 
In a stochastic environment, state transitions and rewards involve inherent randomness. When the agent performs the same action in a given state, it can result in different next states or rewards, based on probabilistic rules. This increased complexity presents a greater challenge for the agent's learning process.
The agent doesn't always know exactly which state it will transition to after acting. For example, moving left might sometimes move the agent to a different state than expected.

### Briefly explain the tabular methods, including Q-learning, that were used to solve the problems. Provide their updated functions and key features.

Tabular methods are effective in settings characterized by a limited and discrete number of states and actions. They utilize a matrix-like structure to store Q-values, representing state-action pairs, to determine the optimal course of action from the current state.

*Q-Learning*

Q-learning is a reinforcement learning algorithm designed to evaluate the value of specific actions taken in various states. Its objective is to identify the optimal action-selection policy through iterative updates of Q-values based on received rewards.

*Key Features:*
- Off-Policy: It learns the value of the optimal policy without depending on the actions taken by the agent.
- Exploration vs. Exploitation: Implements an epsilon-greedy strategy to strike a balance between exploring new actions and exploiting previously learned rewarding actions.
- Convergence: Q-Learning is guaranteed to converge to the optimal action-value function provided there is enough exploration and certain conditions are met.

*Update Function:*
qtable[current_state, action] = qtable[current_state, action] + alpha * (reward + gamma * np.max(qtable[obs, :]) - qtable[current_state, action])

*Double Q-Learning*

Double Q-learning tackles the overestimation bias present in standard Q-learning by keeping two distinct Q-value estimates. This approach enhances learning stability, particularly in environments with stochastic rewards.

*Key Features:*
- Reduced Overestimation Bias: By employing two separate Q-tables, it mitigates the bias that arises from using the same Q-values for both action selection and value updates.
- Off-Policy: Similar to Q-Learning, it learns the optimal policy independently of the actions chosen by the agent.
- Exploration Strategy: Adopts an epsilon-greedy policy akin to that used in Q-Learning.

*Update Function:* 
qtable1[current_state, action] += alpha * (reward + gamma * qtable2[obs, next_best_action] - qtable1[current_state, action])

(or)

qtable2[current_state, action] += alpha * (reward + gamma * qtable1[obs, next_best_action] - qtable2[current_state, action])

### Summary:
Both Q-learning and Double Q-learning serve as effective tabular methods for addressing reinforcement learning challenges. Q-learning is simple and commonly utilized, while Double Q-learning enhances stability and performance in environments with stochastic rewards by alleviating the overestimation bias typical in standard Q-learning. These algorithms learn adaptively from their interactions with the environment, progressively refining their policies to optimize cumulative rewards over time.

### Briefly explain the criteria for a good reward function.

A good reward function in reinforcement learning (RL) encourages the agent to complete tasks efficiently while avoiding undesirable actions. 
* The reward should guide the agent towards the desired behavior, like reaching the end goal in your warehouseBotEnv. For example, rewarding for successful pickups and drop-offs directs the agent toward completing tasks.
* The reward should encourage exploration while favoring exploitation.
* Negative rewards (penalties) help the agent avoid unwanted actions. In your environment, punishing the agent for hitting shelves is an example.
It’s beneficial to reward incremental progress toward the goal. 

In the case of my environment:
* I have assigned -1 as a reward for every step the agent takes, making the agent take the minimum number of steps.
* The agent gets a reward of +5 for successful pickup and +20 for successful drop-off instigating the agent to cover oaths that maximize the reward.
* The agent gets -20 as a reward every time it hits a shelf, as an indication to avoid hitting the shelf while moving across the grid. 

### Part 3 - Solve the Stock Trading Environment

### Given State: 

In each trade, you can either buy/sell/hold. You will start with an investment capital of $100,000 and your performance is measured as a percentage of the return on investment. Save the Q-table as a pickle file and attach it to your assignment submission.

### Stock trading Environment:

This environment is based on the dataset on the historical stock price for Nvidia for the last 2 years. The dataset has 523 entries starting 07/01/2022 to 07/31/2024. The features include information such as the price at which the stock opened, the intraday high and low, the price at which the stock closed, the adjusted closing price, and the volume of shares traded for the day. The environment that calculates the trends in the stock price is provided to you along with the documentation in the .ipynb file. Your task is to use the Q-learning algorithm to learn a trading strategy and increase your total account value over time.

The agent is trained using the QlearningStockEnv. It interacts with the StockTradingEnvironment by choosing actions based on either exploration (random choice) or exploitation (best-known action). The total reward for each episode is stored to track the agent's performance, showing how the cumulative reward evolves as the agent gets better at trading.

The goal of applying Q-learning to the stock trading environment is to train an intelligent agent that can maximize profit over time by deciding when to buy, sell, or hold stocks based on historical price data.

Specifically, the agent aims to:
* Maximize total account value
* Balance exploration and exploitation
* Learn optimal trading strategies
