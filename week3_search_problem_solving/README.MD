*Project #3: The Smart Navigator (A vs. BFS)**
Objective:

In this project, you will implement two fundamental AI search algorithms: Breadth-First Search (BFS) and A* Search. You will test these algorithms across four different maze environments to observe the efficiency of Informed Search (using a heuristic) compared to Uninformed Search.

Requirements:

BFS Implementation: Use a deque (Double-Ended Queue) to implement a First-In-First-Out (FIFO) frontier.

*A Implementation:** Use heapq (Priority Queue) to always expand the node with the lowest total estimated cost $f(n) = g(n) + h(n)$.

Heuristic Function: Implement the Manhattan Distance formula to guide your A* search.

The Tie-Breaker: To achieve optimal efficiency in dense grids, your A* priority should be calculated as: priority = (g + h) + (h * 0.001).

Performance Goal: Your code must correctly return the number of nodes visited for each maze. For "Informed" search to be successful, your A* should visit significantly fewer nodes than BFS in most scenarios.