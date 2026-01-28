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
from typing import Dict, List


def get_neighbors(node, grid):
    """Returns valid North, South, East, West neighbors (0 = path, 1 = wall)."""
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = node[0] + dr, node[1] + dc
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
            neighbors.append((r, c))
    return neighbors

### Heuristic Function: Implement the Manhattan Distance formula to guide your A* search. ###


def manhattan_distance(a, b):
    """
    TASK: Implement Manhattan Distance h(n)
    Formula: |x1 - x2| + |y1 - y2|
    """
    # TODO: Your code here
    return 0


### BFS Implementation: Use a deque (Double-Ended Queue) to implement a First-In-First-Out (FIFO) frontier. ###
def breadth_first_search(grid, start, goal):
    """
    TASK: Implement Breadth-First Search.
    - Use 'deque' for the frontier.
    - Return the total count of nodes visited.
    """
    # Using fifo queue to store nodes that are waiting to be visited
    queue: deque = deque()
    #Adding start position to queue
    queue.append(start)
    # This will be used to track visited nodes
    visited: set = set()
    # Adding starting node to visited
    visited.add(start)
    # Keeping count of visited nodes
    nodes_visited: int = 0

    # While the queue is not empty
    while queue :
        # Getting the next node in the queue by popping from left
        current_node = queue.popleft() 
        # Incrementing visited node counter 
        nodes_visited += 1 # Mental note for me: node visited count should increment after it is removed from the queue.
        # Checking if we reached the goal
        if current_node == goal:
            # This way we can see the end node
            return nodes_visited
        
        # Getting all the neighbors of the current node
        neighbors: List = get_neighbors(current_node, grid)
        # looping through neighbors to see if we already visited them
        for neighbor_node in neighbors:
            # checking if the neighboring node was visited.
            if neighbor_node not in visited:
                # Lets add it to visited to mark it as visited
                visited.add(neighbor_node) # mental note for me, a node is marked as visited when it gets added tot he queue
                # Lets also add it to the queue
                queue.append(neighbor_node)
        
    
    return nodes_visited

### *A Implementation:** Use heapq (Priority Queue) to always expand the node with the lowest total estimated cost $f(n) = g(n) + h(n)$. ###


def a_star_search(grid, start, goal):
    """
    TASK: Implement A* Search.
    - Use 'heapq' for the priority queue.
    - Use the tie-breaker: priority = (g + h) + (h * 0.001)
    - Return the total count of nodes visited.
    """
    nodes_visited = 0
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
