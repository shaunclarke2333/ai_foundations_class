"""
Name: Shaun Clarke
Course: CSC6313 Ai Foundations
Instructor: Margaret Mulhall
Module: 3
Assignment: The Smart Navigator (A vs. BFS)

In this project, you will implement two fundamental AI search algorithms:
Breadth-First Search (BFS) and A* Search. You will test these algorithms across four
different maze environments to observe the efficiency of Informed Search (using a heuristic) compared to Uninformed Search.
"""


import heapq
from collections import deque
from typing import Dict, List, Tuple



def get_neighbors(node: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Docstring for get_neighbors
    
    :param node: Current node position as tuple (row, col)
    :type node: Tuple[int, int]
    :param grid: A 2D list that represents the state space where 0 is a path and 1 is a wall
    :type grid: List[List[int]]
    :return: Returns valid North, South, East, West neighbors (0 = path, 1 = wall).
    :rtype: List[Tuple[int, int]]
    """
    neighbors: List[Tuple[int, int]] = []

    # Getting the grid dimensions 
    rows: int = len(grid) # Total number of rows
    cols: int = len(grid[0])# Total number of columes

    # Unpacking each tuple so we can check all four directions North (-1,0), South (1,0), West (0,-1), East (0,1)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        # Calculating new row position by adding the row change to the current row
        r = node[0] + dr
        # Calculating new column position by adding column change to the current column
        c = node[1] + dc
        # CHecking if the new position calculated above is within the grid boundaries and is not a wall
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
            # If it is a valid path add this neighbor to the neighbors list
            neighbors.append((r, c))
    # Return a list of all valid neighboring nodes/positions
    return neighbors

### Heuristic Function: Implement the Manhattan Distance formula to guide your A* search. ###
def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    TASK: Implement Manhattan Distance h(n)
    Formula: |x1 - x2| + |y1 - y2|
    
    The term Manhattan distance represents the minumum number of moves needed or the shortest distance between <br>
    point b and point a in a grid where only horizontal and vertical moves are alllowed.
    
    :param a: Starting position as a tuple (row, col)
    :type a: Tuple[int, int]
    :param b: End goal position as a tuple (row, col)
    :type b: Tuple[int, int]
    :return: Returns the Manhattan distance as an integer
    :rtype: int
    """
    # Getting coordinates for point a
    x1: int = a[0] # Row coordinate
    y1: int = a[1] # Column coordinate
    # Getting coordinates for point b
    x2: int = b[0] # Row coordinate
    y2: int = b[1] # Column coordinate 

    """
    Calculating the manhattan distance between the cordinates.
    This manhattan distance will be used as the heuristic h(n) in the A* search algorithm.
    """
    manhattan_distance: int = abs(x1 - x2) + abs(y1 - y2)
    # Returning the manhattan distance
    return manhattan_distance


### BFS Implementation: Use a deque (Double-Ended Queue) to implement a First-In-First-Out (FIFO) frontier. ###
def breadth_first_search(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    TASK: Implement Breadth-First Search.
    - Use 'deque' for the frontier.
    - Return the total count of nodes visited.
    This function implemetns BFS which is an uninformed search algorithm.
    - it explores the state space level by level, and guarantess the shortest paths in unweighted graphs
    - it uses a FIFO queue to process each state(node) in the state space in the order they were discovered.
    
    :param grid: The grid represents the state space where 0 is a path and 1 is a wall
    :type grid: List[List[int]]
    :param start: The starting position representing the present state in the state space
    :type start: Tuple[int, int]
    :param goal: The end position that represent the goal state in the state space
    :type goal: Tuple[int, int]
    :return: The number of moves(legal actions) it took to reach the goal
    :rtype: int
    """
    # Using fifo queue to store nodes that are waiting to be visited
    frontier: deque = deque()
    #Adding start position to the queue because this is the starting point
    frontier.append(start)
    # This will be used to track nodes that we already visited.
    visited: set = set()
    # Adding starting node to visited because starting here means its automatically visited
    visited.add(start)
    # Keeping count of visited nodes
    nodes_visited: int = 0

    # While the queue is not empty and we havent hit the goal node
    while frontier :
        # Get the next node in the queue by popping from left
        current_node: Tuple[int, int] = frontier.popleft() 
        # Incrementing visited node counter 
        nodes_visited += 1 # Mental note for me: node visited count should increment after it is removed from the queue.
        # Checking if we reached the goal
        if current_node == goal:
            # This way we can see the number of nodes we visited before hitting the goal.
            return nodes_visited
        
        # Getting all the neighbors of the current node
        neighbors: List = get_neighbors(current_node, grid)
        # looping through neighbors to see if we already visited them
        for neighbor_node in neighbors:
            # checking if the neighboring node was visited.
            if neighbor_node not in visited:
                # It was not visited so we will add it to visited to mark it as visited
                visited.add(neighbor_node) # mental note for me, a node is marked as visited when it gets added tot he queue
                # Lets also add it to the queue
                frontier.append(neighbor_node)
    # Returning the count of nodes visited to see how many steps it took to hit the goal
    return nodes_visited

### *A Implementation:** Use heapq (Priority Queue) to always expand the node with the lowest total estimated cost $f(n) = g(n) + h(n)$. ###
def a_star_search(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
     TASK: Implement A* Search.
    - Use 'heapq' for the priority queue.
    - Use the tie-breaker: priority = (g + h) + (h * 0.001)
    - Return the total count of nodes visited.

    This function calls the A* search algorithm:
    - A* uses a heuristic function (the manhattan distance function above) to traverse the state space more efficiently to reach the goal state.
    - It does this by always choosing the total esitamted cost of the best path(state):
        - It calculates the lowest cost using:
        - f(n) = g(n)+h(n)
            - g(n) the number of actions(moves) from the initial state(starting point) to the present state(current node).
            - h(n) the number of actions(moves) from the initial state(current node) to the goal state, number of moves is know as the manhtattan distance.
            - f(n) the total estimated actions(moves) of the next best state(path)
    
    :param grid: The grid represents the state space where 0 is a path and 1 is a wall
    :type grid: List[List[int]]
    :param start: The starting position representing the present state in the state space
    :type start: Tuple[int, int]
    :param goal: The end position that represent the goal state in the state space
    :type goal: Tuple[int, int]
    :return: The number of moves(legal actions) it took to reach the goal
    :rtype: int
    """
    frontier: List = []
    counter: int = 0

    visited: set = set()

    # Dictionary to track g(n) which is the cost from start to the node
    g_cost: Dict[Tuple[int, int], int] = {}

    # Setting the g score for the start node
    g_cost[start] = 0
    # Calculating the heuristic(h) value for the start node using manhattan distance
    h_start: int = manhattan_distance(start, goal) 
    # Calculating f (f = g + h) wich is the total estimated cost from start to finish
    f_start: int = g_cost[start] + h_start
    # Calculating the node that takes priority using the tie-breaker
    priority: float = f_start + (h_start * 0.001)
    # Adding the starting node and details to frontier using heappush
    heapq.heappush(frontier, (priority, counter, start))
    # Incrementing counter
    counter += 1
    nodes_visited: int = 0

    # Starting the search loop party
    while frontier: # While the list is not empty
        # Getting the node with the lowest priority by popping and unpacking the node with the best priority
        current_priority: float
        current: Tuple[int, int]
        current_priority, _, current = heapq.heappop(frontier)
        # If we didnt hit the goal node but we already visited this node, skip it by continuing
        if current in visited:
            continue
        # We checked the current node and it wasnt the goal node so adding it to the visited set
        visited.add(current)
        # Incrementing node visited because it was popped
        nodes_visited += 1
         # Checking if we hit the goal node
        if current == goal:
            return nodes_visited

        # We had no luck with the node we check so now we need to find the neibors of that node
        neighbors: List[Tuple[int, int]] = get_neighbors(current, grid)
        # looping through neighbors to calculate the A* search params
        for neighbor in neighbors:
            # Calculating a somewhat tentstive g_cost
            new_g_cost: int = g_cost[current] + 1
            # If we havent visited this node yet or we have seen it before but this path is cheaper(cheaper g_cost)
            if neighbor not in g_cost or new_g_cost < g_cost[neighbor]:
                # Updating the g_cost for this neighbor because one of the conditions were satisfied
                g_cost[neighbor] = new_g_cost
                # Calling manhattan distance func to calculate the heuristic(roughly how far is this present node from the goal) for this neighbor
                h: int = manhattan_distance(neighbor, goal)
                # Calculating the total cost from node to goal for this neighbot
                f: int = new_g_cost + h
                # Calculating the node that takes priority using the tie-breaker
                priority: float = f + (h * 0.001)
                # addding current node and othe rdetails to to frontier using heappush
                heapq.heappush(frontier, (priority, counter,neighbor))
                # Incrementing counter
                counter += 1

    # TODO: Your code here
    return nodes_visited


# 1. WINDING MAZE (10x10)
m1 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# 2. CHECKERBOARD (10x10)
m2 = [[(r+c)%2 if (0<r<9 and 0<c<9) else 0 for c in range(10)] for r in range(10)]

# 3. THE SPIRAL (12x12)
m3 = [[0]*12 for _ in range(12)]
for i in range(2, 10):
    m3[2][i] = m3[i][10] = m3[10][11-i] = m3[11-i][2] = 1
m3[3][2] = 0 

# 4. CORRIDORS (10x10)
m4 = [[0]*10 for _ in range(10)]
for r in [2, 4, 6, 8]:
    for c in range(0, 9): m4[r][c] = 1
for r in [1, 3, 5, 7]:
    m4[r][9] = 1

start = (0, 0)

print(f"1. WINDING MAZE - BFS: {breadth_first_search(m1, start, (9,9))} | A*: {a_star_search(m1, start, (9,9))}")
print(f"2. CHECKERBOARD - BFS: {breadth_first_search(m2, start, (9,9))} | A*: {a_star_search(m2, start, (9,9))}")
print(f"3. THE SPIRAL   - BFS: {breadth_first_search(m3, start, (6,6))} | A*: {a_star_search(m3, start, (6,6))}")
print(f"4. CORRIDORS    - BFS: {breadth_first_search(m4, start, (9,0))} | A*: {a_star_search(m4, start, (9,0))}")