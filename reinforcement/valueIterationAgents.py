# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(self.iterations):
            states = self.mdp.getStates()
            count = util.Counter()
            for state in states:
                current_max = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, action)
                    if q > current_max:
                        current_max = q
                if current_max == float('-inf'):
                    count[state] = 0
                else:
                    count[state] = current_max
            self.values = count

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = 0
        for next, prob in transitions:
            reward = self.mdp.getReward(state, action, next)
            sum += prob * (reward + (self.discount * self.values[next]))
        return sum
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        best = None
        current_max = float('-inf')
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        for action in actions:
            q_val = self.computeQValueFromValues(state, action)
            if q_val > current_max:
                current_max = q_val
                best = action
        return best
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        for i in range(self.iterations):
            cyl_value = i % len(states)
            state = states[cyl_value]
            if not self.mdp.isTerminal(state) or self.mdp.getPossibleActions(state):
                current_max = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    qval = self.computeQValueFromValues(state, action)
                    if qval > current_max:
                        current_max = qval
                self.values[state] = current_max

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        pred = {}
        queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                pred[state] = set()

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for next, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0: # nonzero probability
                        if next not in pred:
                            pred[next] = set()
                        pred[next].add(state)
        
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                current_max = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q_val = self.computeQValueFromValues(state, action)
                    if q_val > current_max:
                        current_max = q_val
                diff = abs(self.values[state] - current_max)
                queue.update(state, -diff)
        
        for i in range(self.iterations):
            # terminate if empty
            if queue.isEmpty():
                break
            # pop s
            state = queue.pop()
            if not self.mdp.isTerminal(state):
                current_max = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q_val = self.computeQValueFromValues(state, action)
                    if q_val > current_max:
                        current_max = q_val
                self.values[state] = current_max
                    
            for curr_p in pred[state]:
                if not self.mdp.isTerminal(curr_p):
                    current_max = float('-inf')
                    for action in self.mdp.getPossibleActions(curr_p):
                        q_val = self.computeQValueFromValues(curr_p, action)
                        if q_val > current_max:
                            current_max = q_val
                    diff = abs(self.values[curr_p] - current_max)
                    if diff > self.theta:
                        queue.update(curr_p, -diff)
                