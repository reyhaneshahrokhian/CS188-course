# Project3

* ## Reinforcement Learning
* ### This project contains 10Questions:
   + #### Q1  : Value Iteration
     * I have implemented a value iteration agent in ValueIterationAgent of the valueIterationAgents.py. Value iteration computes k-step estimates of the optimal values, Vk.
   + #### Q2  : Bridge Crossing Analysis
     * I have implemented question2() of analysis.py. Only ONE of the discount and noise parameters will change so that the optimal policy causes the agent to attempt to cross the bridge.
   + #### Q3  : Policies
     * I have implemented question3a() through question3e() which each return a 3-item tuple of (discount, noise, living reward) in analysis.py. It chooses settings of the discount, noise, and living reward parameters for this MDP to produce optimal policies of several different types.
   + #### Q4  : Asynchronous Value Iteration
     * I have implemented a value iteration agent in AsynchronousValueIterationAgent of the valueIterationAgents.py. It finds that the value of the start state (V(start)) and the empirical resulting average reward are quite close.
   + #### Q5  : Prioritized Sweeping Value Iteration
     * I have implemented the PrioritizedSweepingValueIterationAgent of the valueIterationAgents.py. It is actually a simplified version of the standard prioritized sweeping algorithm.
   + #### Q6  : Q-Learning
     * I have implemented a Q-learning agent, which does very little on construction, but instead learns by trial and error from interactions with the environment through its update method. it is in QLearningAgent in qlearningAgents.py
   + #### Q7  : Epsilon Greedy
     * I have implemented epsilon-greedy action selection in getAction to complete the Q-learning agent, meaning it chooses random actions an epsilon fraction of the time, and follows its current best Q-values otherwise.
   + #### Q8  : Bridge Crossing Revisited
     * I have first trained a completely random Q-learner with the default learning rate on the noiseless BridgeGrid for 50 episodes and observe whether it finds the optimal policy. Then, tried the same experiment with an epsilon of 0. Is there an epsilon and a learning rate for which it is highly likely (greater than 99%) that the optimal policy was learned after 50 iterations? Implementing question8() in analysis.py return EITHER a 2-item tuple of (epsilon, learning rate) OR the string 'NOT POSSIBLE'.
   + #### Q9  : Q-Learning and Pacman
     * Time to play some Pacman!
   + #### Q10 : Approximate Q-Learning
     * I have implemented an approximate Q-learning agent that learns weights for features of states, where many states might share the same features. It is in ApproximateQAgent class in qlearningAgents.py, which is a subclass of PacmanQAgent.
