README

Question 1: Finding a Fixed Food Dot using DFS

A depth first search means that nodes are expanded as far down the tree as possible. For this, I decided to use a stack since it allows me to store successor nodes at the top and pull from these nodes to expand further nodes.
I initialized a list of actions and a list of all visited nodes so I can avoid revisiting nodes (since this is a graph DFS), and I pushed a tuple of the starting state and the actions list to the stack. Then I iterated over the stack, first checking if the current position was a goal state so I could return the actions taken. Then I checked if the current node had already been visited, and if not, I added it to the visited nodes list and found the successors of the current node. After checking if each successor's position was not already in the visited nodes list, I created a new list of actions including the action to reach the successor and pushed a tuple with the position and actions to the stack.

Question 2: Breadth First Search

Breadth first search is similar to DFS, except instead of expanding nodes as far as possible, it expands all nodes on one level before moving to the next. To do this, I changed the stack data structure to a queue, since its FIFO properties allow all successors of one node to be expanded before moving onto deeper nodes of successors. The rest of the code remained the same as DFS.

Question 3: Varying the Cost Function

Uniform cost search takes the cost to get to successors into account as it expands nodes. To do this, I used a priority queue which uses the cost of nodes as its priority. In order to keep track of the cost, I added it to the tuples I pushed onto the priority queue (so it would be a tuple of the node state, the actions list, and the cost). I also changed the visited node list to dictionary so I could assign each node to its respective cost. I then iterated over the queue until it was empty, first checking if the current node was a goal state and returning the list of actions if so. Then I checked if current node was in the visited dictionary. If not, I assigned the current node as the key and the cost as its value. I then found the node's successors. If the successor nodes were not in the visited dictionary, or if the total cost to reach the successor was lower than previously assigned, I pushed a tuple of the successor node, its list of actions, and the new total cost to stack (along with the new cost as the priority).

Question 4: A* search

A* search takes the cost to get to each node as well as the heuristic of each node into account when determining which nodes to expand. I used a priority queue for A* as well, since it determines where to go based on lowest total node costs. The priority used here is the added cost of the node's heuristic plus the cost to reach the node. The tuple pushed to the priority queue remained the same (node state, actions list, cost to reach the node). In order to find nodes' heuristic, I used the heuristic function provided in the arguments.
When adding successor nodes to the queue, I first found the heuristic of successors and added that to the cost of actions to reach the successor. If the successor node was not in the visited dictionary or the cost to reach the next node was less than currently assigned, I pushed a tuple with the next node, next actions list, and next cost to stack. The priority was the total cost (including heuristic), denotes as fCost.

Question 5: Finding All the Corners

To implement the CornersProblem, I initialized a list cornersStatus containing boolean values to indicate whether each corner has been visited or not (initialized to False). To define getStartState, I returned a tuple of the the starting position as well as the cornersStatus list (as a tuple).
The goal is for all corners to have been visited, so I implemented isGoalState by checking if any of the values in cornersStatus are still False. If so, that corner has not been visited and the goal hasn't been reached. Otherwise, it should return True.
The getSuccessors method should return the next states available to Pacman, as well as the actions to get to the next positions and the cost (which is 1). It should also check if any of the successors are a corner, so it can update the cornersStatus list. To get all successor positions, I iterated to the actions and found the next x and y that Pacman would reach going in that direction. If this next position was not a wall, I checked if it was a corner and updated cornersStatus accordingly, and then appended the next state, action, and cost to the successors list.

Question 6: Corners Problem: Heuristic

The goal is to create a heuristic function for admissible and consistent heuristics. Pacman will maintain a manhattan distance to each unvisited corner from every position, and one of the corners wold be the furthest. If the heuristic is the manhattan distance to the furthest unvisited corner, then it should remain less or equal to the total cost to visit all corners as well as be less or equal to the cost of getting to the nearest successor plus the heuristic of the successor. This is the logic implemented in the heuristic function.

Question 7: Eating All The Dots

Similar to the heuristic implemented in for the corners problem, my logic is that the distance to the furthest uneaten food on the foodGrid should be less than or equal to the total cost of eating all foods as well as less or equal to the the cost of getting to the next successor plus the heuristic of the successor. To implement this, I iterated all the food items in food list. For all the items not eaten, I found the maze distance to those items from the current position and returned the largest distance.

Question 8: Suboptimal Search

To implement the goal state in AnyFoodSearchProblem, I checked if there was food at the the current state. If so, I returned True and otherwise False. In findPathtoClosestDot, problem is already defined as the AnyFoodSearchProblem. I used the breadth first search function to solve the problem.






