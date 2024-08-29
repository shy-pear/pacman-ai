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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        #Successor game state capsules
        newCapsules = successorGameState.getCapsules()
        #score
        score = successorGameState.getScore()

        #Find distance to every food, add to foodDistances   
        foodList = newFood.asList()
        foodDistances = [manhattanDistance(newPos, food) for food in foodList]

        #prefer foods that are closer
        if foodDistances:
            score += 10 / min(foodDistances)

        # Find distance to every capsule, add to capsuleDistances
        capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in newCapsules]
        
        #prefer eating capsules if they are closer
        if capsuleDistances:
            score += 50 / min(capsuleDistances)

        #Find the distance of every ghost, add to ghostDistances
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        
        #Get the distance to every ghost and scared times of the ghosts. If scaredTime is greater
        # than zero, power pellet has been eaten and Pacman should prefer to eat scared ghosts
        #Otherwise, Pacman should stronger prefer to avoid ghosts (especially if the ghost is close)
        for index in range(len(newGhostStates)):
            ghostDistance = ghostDistances[index]
            scaredTime = newScaredTimes[index]

            #If power pellet has been eaten
            if scaredTime > 0:
                score += 100 / (ghostDistance + 1)
            #else if ghost is extremely close
            elif ghostDistance <= 1: 
                    score -= 1000
            #else prefer further distance from ghosts
            else:
                score -= 1 / (ghostDistance + 1)

        
        #Return the final Score
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
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agent):
            #The number of agents, the next agent, and the next depth for minimax
            numAgents = state.getNumAgents()
            nextAgent = (agent + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            #If terminal state or max depth is reached, return the evaluation function value
            if depth == self.depth or state.isWin() or state.isLose() or not state.getLegalActions(agent):            
                return (self.evaluationFunction(state), None)

            #If agent is Pacman, goal is to maximize value of state
            if (agent == 0):
                #set bestValue to -infinity
                bestValue = float("-inf")
                bestAction = None
                #for each action, generate successor and call minimix on next agent
                for action in state.getLegalActions(agent):
                    value, _ = minimax(state.generateSuccessor(agent, action), nextDepth, nextAgent)
                    #Find found value is greater than best value so far, adjust bestValue
                    if(value > bestValue):
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            
            #else, the agent is a ghost
            else:
                #set bestValue to +infinity
                bestValue = float("inf")
                bestAction = None
                #for each action, generate successor and call minimix on next agent         
                for action in state.getLegalActions(agent):
                    value, _ = minimax(state.generateSuccessor(agent, action), nextDepth, nextAgent)
                    #Find found value is smaller than best value so far, adjust bestValue
                    if(value < bestValue):
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
        
        #for getAction, return the action for minimax with gamestate, starting depth 0,
        #starting agent 0
        return minimax(gameState, 0, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta(state, depth, agent, alpha, beta):
            #The number of agents, the next agent, and the next depth for alpha_beta
            numAgents = state.getNumAgents()
            nextAgent = (agent + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            # If terminal state or max depth reached, evaluate state
            if depth == self.depth or state.isWin() or state.isLose() or not state.getLegalActions(agent):
                return (self.evaluationFunction(state), None)
            
            #If agent is Pacman, goal is to maximize value of state
            if (agent == 0):
                bestValue = float("-inf")
                bestAction = None
                for action in state.getLegalActions(agent):
                    #for each action, generate successor and call alpha_beta on next agent  
                    value, _ = alpha_beta(state.generateSuccessor(agent, action), nextDepth, nextAgent, alpha, beta)

                    #Find found value is greater than best value so far, adjust bestValue
                    if value > bestValue:
                        bestValue = value
                        bestAction = action

                    #if best value exceeds beta, prune branch
                    if bestValue > beta:
                        return bestValue, bestAction
                    #update alpha with best value
                    alpha = max(alpha, bestValue)
                return bestValue, bestAction
            #else, the agent is a ghost
            else:
                bestValue = float("inf")
                bestAction = None

                for action in state.getLegalActions(agent):
                    #for each action, generate successor and call alpha_beta on next agent 
                    value, _ = alpha_beta(state.generateSuccessor(agent, action), nextDepth, nextAgent, alpha, beta)

                    #Find found value is smaller than best value so far, adjust bestValue
                    if value < bestValue:
                        bestValue = value
                        bestAction = action
                    
                    #if best value is less than alpha, prune branch
                    if bestValue < alpha:
                        return bestValue, bestAction
                    #update beta with best value
                    beta = min(beta, bestValue)
                return bestValue, bestAction
        
        #for getAction, return the action for alpha_beta with gamestate, starting depth 0,
        #starting agent 0, alpha set to -infinity, and beta set to +infinity
        return alpha_beta(gameState, 0, 0, float("-inf"), float("inf"))[1]


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
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agent):
            #The number of agents, the next agent, and the next depth for expectimax
            numAgents = state.getNumAgents()
            nextAgent = (agent + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            # If terminal state or max depth reached, evaluate state
            if depth == self.depth or state.isWin() or state.isLose() or not state.getLegalActions(agent):
                return (self.evaluationFunction(state), None)
            
            #If agent is Pacman, goal is to maximize value of state
            if (agent == 0):
                bestValue = float("-inf")
                bestAction = None
                #for each action, generate successor and call expectimax on next agent
                for action in state.getLegalActions(agent):
                    value, _ = expectimax(state.generateSuccessor(agent, action), nextDepth, nextAgent)
                    #Find found value is greater than best value so far, adjust bestValue
                    if(value > bestValue):
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            else:
                expValue = 0
                actions = state.getLegalActions(agent)
                for action in actions:
                    value, _ = expectimax(state.generateSuccessor(agent, action), nextDepth, nextAgent)
                    expValue += value / len(actions)
                return expValue, None

        return expectimax(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    This function evalues the state and bases the score on Pacman's proximity
    to ghosts, scared ghosts, food, and capsules.
    Manhattan distances to all foods are calculated, and Pacman is given a
    reward proportional to the inverse of the closest food and a penalty
    based on the number of foods left to eat.
    Manhattan distances to all capsules are calculated as well. Pacman is given
    a reward proportional to the inverse of the closest capsule, a penalty
    based on the number of capsules left to eat. 
    Finally, manhattan distances to ghosts are calculated, and if the ghost is 
    scared then Pacman is given a reward proportional to the inverse of the ghost's
    distance. If the ghost is not scared, then Pacman is given a penalty 
    proportional to the inverse of the ghost's distance.
    """
    "*** YOUR CODE HERE ***"
    #state info
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsulePositions = currentGameState.getCapsules()
    score = currentGameState.getScore()

    #Penalties and rewards
    ghostPenalty = 200
    scaredGhostReward = 200
    foodReward = 10
    capsuleReward = 20

    "-----penalities/rewards for food---------"
    # Calculate food distances
    if foodList:
        minFoodDistance = min(manhattanDistance(pacmanPosition, food) for food in foodList)
        #Reward based on closest food
        score += foodReward / minFoodDistance

    #Penalty for remaining foods
    score -= len(foodList) * foodReward

    "-----penalities/rewards for capsules---------"
    # Calculate capsule distances
    if capsulePositions:
        minCapsuleDistance = min(manhattanDistance(pacmanPosition, capsule) for capsule in capsulePositions)
        # Reward based on closest capsule
        score += capsuleReward / minCapsuleDistance
    
    # Penalty for remaining capsules
    score -= len(capsulePositions) * capsuleReward

    "-----penalities/rewards for ghosts---------"
     # Calculate ghost distances
    for ghost in ghostStates:
        distanceToGhost = manhattanDistance(pacmanPosition, ghost.getPosition())
        if distanceToGhost > 0:
            if ghost.scaredTimer > 0:
                # Reward for scared ghosts
                score += scaredGhostReward / distanceToGhost
            else:
                # Penalty for normal ghosts
                score -= ghostPenalty / distanceToGhost

    return score

# Abbreviation
better = betterEvaluationFunction
