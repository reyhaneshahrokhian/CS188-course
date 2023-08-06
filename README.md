# Project5

* ## Ghostbuster
* ### This project contains 11Questions:
   + #### Q0  : DiscreteDistribution Class
     * I have implemented the normalize method, which normalizes the values in the distribution to sum to one, but keeps the proportions of the values the same and sample method, which draws a sample from the distribution, where the probability that a key is sampled is proportional to its corresponding value. they are in DiscreteDistribution class defined in inference.py.
   + #### Q1  : Observation Probability
     * I have implemented the ObservationProb method in the InferenceModule base class in inference.py. this method returns the probability of the noisy distance reading given Pacman's position and the ghost's position.
   + #### Q2  : Exact Inference Observation
     * I have implemented the observeUpdate method in ExactInference class of inference.py to correctly update the agent's belief distribution over ghost positions.
   + #### Q3  : Exact Inference with Time Elapse
     * I have implemented implement the elapseTime method in ExactInference. The elapseTime step, update the belief at every position on the map after one time step elapsing.
   + #### Q4  : Exact Inference Full Test
     * I have implemented the chooseAction method in GreedyBustersAgent in bustersAgents.py. the agent first find the most likely position of each remaining uncaptured ghost, then choose an action that minimizes the maze distance to the closest ghost.
   + #### Q5  : Approximate Inference Initialization and Beliefs
     * I have first implemented the functions initializeUniformly and getBeliefDistribution in the ParticleFilter class in inference.py. The getBeliefDistribution method then takes the list of particles and converts it into a DiscreteDistribution object.
   + #### Q6  : Approximate Inference Observation
     * I have implemented the observeUpdate method in the ParticleFilter class in inference.py. this method constructs a weight distribution over self.particles.
   + #### Q7  : Approximate Inference with Time Elapse
     * I have implemented the elapseTime function in the ParticleFilter class in inference.py. This function construct a new list of particles that corresponds to each existing particle in self.particles advancing a time step.
   + #### Q8  : Joint Particle Filter Initialization
     * I have implemented the initializeUniformly method in JointParticleFilter in inference.py.A particle filter that tracks multiple ghosts simultaneously. 
   + #### Q9  : Joint Particle Filter Observation
     * I have implemented the observeUpdate method in the JointParticleFilter class of inference.py. the implementation weight and resample the entire list of particles based on the observation of all ghost distances.
   + #### Q10  : Joint Particle Filter Time Elapse and Full Test
     * I have implemented the elapseTime method in JointParticleFilter in inference.py to resample each particle correctly for the Bayes net.
