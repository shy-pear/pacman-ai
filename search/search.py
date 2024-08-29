# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()
    stack = util.Stack()

    actions = []
    visited = []
    stack.push((root, actions))

    while not stack.isEmpty():
        # get current node, actions list
        currentNode, actions = stack.pop()

        # if current node is a goal state, return actions list
        if problem.isGoalState(currentNode): return actions

        # if current node is not in visited list, append to visited
        if currentNode not in visited:
            visited.append(currentNode)
        
            # get all successors, and if not visited, push to stack
            successors = problem.getSuccessors(currentNode)
            for nextNode, action, stepCost in successors:
                if nextNode not in visited:
                    nextActions = actions + [action]
                    stack.push((nextNode, nextActions))
    
    return None

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()
    queue = util.Queue()

    actions = []
    visited = []

    # use queue instead of stack
    queue.push((root, actions))

    while not queue.isEmpty():
        # get current node, actions list
        currentNode, actions = queue.pop()

        # if current node is a goal state, return actions list
        if problem.isGoalState(currentNode): return actions
        
        # if current node is not in visited list, append to visited
        if currentNode not in visited:
            visited.append(currentNode)
        
             # get all successors, and if not visited, push to queue
            successors = problem.getSuccessors(currentNode)
            for nextNode, action, stepCost  in successors:
                if nextNode not in visited:
                    nextActions = actions + [action]
                    queue.push((nextNode, nextActions))
    
    return None
    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    pqueue = util.PriorityQueue()
    root = problem.getStartState()
    actions = []
    visited = {}

    # use priority queue, cost = priority
    pqueue.push((root, actions, 0), 0)

    while not pqueue.isEmpty():
        # get current node, actions list, cost
        currentNode, actions, cost = pqueue.pop()

        # if current node is a goal state, return actions list
        if problem.isGoalState(currentNode):
            return actions
        
        # if current node is not in visited dictionary, append
        # key = current node, value = cost
        if currentNode not in visited or cost < visited[currentNode]:
            visited[currentNode] = cost

            # get all successors, and if not visited or if new cost less
            # than previously assigned, push to priority queue
            successors = problem.getSuccessors(currentNode)

            for nextNode, action, stepCost in successors:
                nextCost = cost + stepCost
                if nextNode not in visited or nextCost < visited[nextNode]:
                    nextActions = actions + [action]
                    pqueue.push((nextNode, nextActions, nextCost), nextCost)

    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    pqueue = util.PriorityQueue()
    root = problem.getStartState()
    actions = []
    visited = {}

    # use priority queue, cost = priority + heuristic
    pqueue.push((root, [], 0), 0 + heuristic(root, problem))


    while not pqueue.isEmpty():
         # get current node, actions list, cost
        currentNode, actions, cost = pqueue.pop()

        # if current node is a goal state, return actions list
        if problem.isGoalState(currentNode):
            return actions
        
        # if current node is not in visited dictionary, append
        # key = current node, value = cost
        if currentNode not in visited or cost < visited[currentNode]:
            visited[currentNode] = cost

            # get all successors, and if not visited or if new cost less
            # than previously assigned, push to priority queue
            # take heuristic into account for priority
            successors = problem.getSuccessors(currentNode)

            for successor, action, stepCost in successors:
                nextCost = cost + stepCost
                fCost = nextCost + heuristic(successor, problem)
                if successor not in visited or nextCost < visited[currentNode]:
                    nextActions = actions + [action]
                    # priority = cost to reach successor + heuristic of successor
                    pqueue.push((successor, nextActions, nextCost), fCost)

    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
