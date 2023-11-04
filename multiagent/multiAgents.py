# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print(str(successorGameState))
        newPos = successorGameState.getPacmanPosition()
        #print(str(newPos))
        newFood = successorGameState.getFood()
        #print(str(newFood))
        newGhostStates = successorGameState.getGhostStates()
        #print(str(newGhostStates))
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        new_food = newFood.asList()
        closest_food = float('inf')
        for food in new_food:
            closest = manhattanDistance(newPos, food)
            if closest < closest_food:
                closest_food = closest
        not_scared = []
        scared = []
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                not_scared.append(ghost) # get not scared ghosts
            else:
                scared.append(ghost) # get scared ghosts
        closet_ghost = float('inf') # LOL i have a typo
        for ghost in not_scared:
            closest = manhattanDistance(newPos, ghost.getPosition())
            if closest < closet_ghost:
                closet_ghost = closest
            
        closest_scared = float('inf')
        for ghost in scared:
            closest = manhattanDistance(newPos, ghost.getPosition())
            if closest < closest_scared:
                closest_scared = closest
        
        # subtract score from remaining food
        score = successorGameState.getScore() - len(new_food)
        score += 1.0 / (closest_food + 5.0)
        score -= 1.0 / (closet_ghost + 5.0) # pacman doesnt go to ghosts :D
        score += 1.0 / (closest_scared + 5.0)
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def value(gameState, agentIndex, depth):
            if agentIndex == gameState.getNumAgents():
                depth += 1
                agentIndex = 0
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max_value(gameState, agentIndex, depth)
            else:
                return min_value(gameState, agentIndex, depth)

        def max_value(gameState, agentIndex, depth):
            v = float('-inf')
            for successor in gameState.getLegalActions(agentIndex):
                v = max(v, value(gameState.generateSuccessor(agentIndex, successor), agentIndex + 1, depth))
            return v
        
        def min_value(gameState, agentIndex, depth):
            v = float('inf')
            for successor in gameState.getLegalActions(agentIndex):
                v = min(v, value(gameState.generateSuccessor(agentIndex, successor), agentIndex + 1, depth))
            return v
        best_value = float('-inf')
        best_action = 0
        for action in gameState.getLegalActions(0):
            current_value = value(gameState.generateSuccessor(0, action), 1, 0)
            if current_value > best_value:
                best_value = current_value
                best_action = action
        return best_action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
 
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def value(gameState, agentIndex, depth, alpha, beta):
            if agentIndex == gameState.getNumAgents():
                depth += 1
                agentIndex = 0
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max_value(gameState, agentIndex, depth, alpha, beta)
            else:
                return min_value(gameState, agentIndex, depth, alpha, beta)
        
        def max_value(gameState, agentIndex, depth, alpha, beta):
            v = float('-inf')
            for successor in gameState.getLegalActions(agentIndex):
                v = max(v, value(gameState.generateSuccessor(agentIndex, successor), agentIndex + 1, depth, alpha, beta))
                if v > beta: # pseudo code says >= beta tho :( spent so much time debugging this
                    return v
                alpha = max(alpha, v)
            return v
        
        def min_value(gameState, agentIndex, depth, alpha, beta):
            v = float('inf')
            for successor in gameState.getLegalActions(agentIndex):
                v = min(v, value(gameState.generateSuccessor(agentIndex, successor), agentIndex + 1, depth , alpha, beta))
                if v < alpha: # same as above :(
                    return v
                beta = min(beta, v)
            return v
        
        best_value = float('-inf')
        best_action = 0
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            current_value = value(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if current_value > best_value:
                best_value = current_value
                best_action = action
            alpha = max(alpha, best_value)
        return best_action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def value(gameState, agentIndex, depth):
            if agentIndex == gameState.getNumAgents():
                depth += 1
                agentIndex = 0
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max_value(gameState, agentIndex, depth)
            else:
                return exp_value(gameState, agentIndex, depth)
        
        def max_value(gameState, agentIndex, depth):
            v = float('-inf')
            for successor in gameState.getLegalActions(agentIndex):
                v = max(v, value(gameState.generateSuccessor(agentIndex, successor), agentIndex + 1, depth))
            return v
        
        def exp_value(gameState, agentIndex, depth):
            v = 0
            p = 1 / len(gameState.getLegalActions(agentIndex))
            for successor in gameState.getLegalActions(agentIndex):
                v += p * value(gameState.generateSuccessor(agentIndex, successor), agentIndex + 1, depth)
            return v
        current = float('-inf')
        current_best = 0
        for action in gameState.getLegalActions(0):
            cur_val = value(gameState.generateSuccessor(0, action), 1, 0)
            if cur_val > current:
                current = cur_val
                current_best = action
        return current_best
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: hiiii this is the best evaluation function!!!! jk it's the same :D
    """
    newPos = currentGameState.getPacmanPosition()
    #print(newPos)
    newFood = currentGameState.getFood()
    #print(newFood)
    newGhostStates = currentGameState.getGhostStates()
    #print(newGhostStates)
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    new_food = newFood.asList()
    closest_food = float('inf')
    for food in new_food:
        closest = manhattanDistance(newPos, food)
        if closest < closest_food:
            closest_food = closest
    not_scared = []
    scared = []
    for ghost in newGhostStates:
        if ghost.scaredTimer == 0:
            not_scared.append(ghost) # get not scared ghosts
        else:
            scared.append(ghost) # get scared ghosts
    closet_ghost = float('inf') # LOL i have a typo
    for ghost in not_scared:
        closest = manhattanDistance(newPos, ghost.getPosition())
        if closest < closet_ghost:
            closet_ghost = closest
        
    closest_scared = float('inf')
    for ghost in scared:
        closest = manhattanDistance(newPos, ghost.getPosition())
        if closest < closest_scared:
            closest_scared = closest
    
    # subtract score from remaining food
    score = currentGameState.getScore() - len(new_food)
    score += 1.0 / (closest_food + 5.0)
    score -= 1.0 / (closet_ghost + 5.0) # pacman doesnt go to ghosts :D
    score += 1.0 / (closest_scared + 5.0)
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
