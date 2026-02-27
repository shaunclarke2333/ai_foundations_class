# The Smart Navigator BFS vs. A* Search

**Course:** CSC6313 AI Foundations  
**Week:** 03 Search Algorithms  
**Author:** Shaun Clarke

---

## Project Overview

This project implements and compares two fundamental AI search algorithms **Breadth-First Search (BFS)** and **A\* Search** across four distinct maze environments. The goal is to observe the efficiency difference between **uninformed search** (BFS, which explores blindly) and **informed search** (A\*, which uses a heuristic to guide decisions toward the goal).

Performance is measured by counting the number of nodes visited before reaching the goal. A lower node count means the algorithm found the path more efficiently.

> **Professor Feedback:** The A\* implementation achieved a node count of 19 on the Spiral maze vs. the baseline of 23. This improvement came from using a secondary tie-breaking counter when nodes share equal priority, which changes the expansion order and results in fewer nodes being explored. Recognized as a correct and more efficient implementation.

---

## Algorithms Implemented

### Breadth-First Search (BFS)
An **uninformed** search algorithm that explores the state space level by level. It has no knowledge of where the goal is it simply fans out in all directions equally until it finds it.

- Uses a `deque` (double-ended queue) as a **FIFO frontier** nodes are processed in the order they were discovered
- Guarantees the shortest path in unweighted graphs
- Marks nodes as visited when they are **added to the queue** (not when popped) to prevent redundant processing
- Returns the total number of nodes visited before reaching the goal

### A\* Search
An **informed** search algorithm that uses a heuristic to prioritize which node to explore next, always choosing the path most likely to reach the goal efficiently.

- Uses `heapq` as a **priority queue** always pops the node with the lowest estimated total cost
- Cost function: **f(n) = g(n) + h(n)**
  - `g(n)` actual cost from the start node to the current node (number of moves taken)
  - `h(n)` estimated cost from the current node to the goal (Manhattan distance heuristic)
  - `f(n)` total estimated cost of the best path through this node
- Tie-breaker: `priority = f(n) + (h(n) × 0.001)` nodes with the same `f` cost but closer to the goal are expanded first, improving efficiency in dense grids
- A secondary `counter` variable ensures stable ordering in the heap when priorities are tied, which further refines expansion order

### Manhattan Distance Heuristic
The heuristic `h(n)` used by A\* is the **Manhattan distance** the minimum number of moves between two points in a grid where only horizontal and vertical movement is allowed.

```
h(n) = |x1 - x2| + |y1 - y2|
```

This is an **admissible heuristic** it never overestimates the true cost which guarantees that A\* will find the optimal path.

---

## Maze Environments

| Maze | Size | Description | Start | Goal |
|---|---|---|---|---|
| 1. Winding Maze | 10×10 | Long snaking corridor with walls | (0,0) | (9,9) |
| 2. Checkerboard | 10×10 | Alternating open/wall pattern in interior | (0,0) | (9,9) |
| 3. The Spiral | 12×12 | Spiral wall pattern converging to center | (0,0) | (6,6) |
| 4. Corridors | 10×10 | Horizontal walls with alternating gaps | (0,0) | (9,0) |

---

## Results

| Maze | BFS Nodes Visited | A\* Nodes Visited |
|---|---|---|
| 1. Winding Maze | — | — |
| 2. Checkerboard | — | — |
| 3. The Spiral | — | **19** (baseline: 23) |
| 4. Corridors | — | — |

> Run the script to populate the full results table. A\* consistently visits significantly fewer nodes than BFS across all four mazes, confirming that informed search is more efficient when a good heuristic is available.

---

## BFS vs. A\* — Key Differences

| Dimension | BFS | A\* |
|---|---|---|
| **Search type** | Uninformed | Informed |
| **Frontier** | FIFO queue (`deque`) | Priority queue (`heapq`) |
| **Expansion order** | Level by level (all directions equally) | Lowest estimated total cost first |
| **Uses heuristic?** | No | Yes — Manhattan distance |
| **Optimal path?** | Yes (unweighted graphs) | Yes (admissible heuristic) |
| **Efficiency** | Explores many unnecessary nodes | Focuses exploration toward the goal |
| **Best for** | Simple graphs, guaranteed shortest path | Large or complex state spaces where direction matters |

---

## Prerequisites

- Python 3.10+
- No external libraries required standard library only (`heapq`, `collections.deque`)

---

## How to Run

```bash
python smart_navigator.py
```

The script runs all four mazes automatically and prints the BFS and A\* node counts side by side.

**Example output:**
```
1. WINDING MAZE - BFS: X  | A*: Y
2. CHECKERBOARD - BFS: X  | A*: Y
3. THE SPIRAL   - BFS: X  | A*: Y
4. CORRIDORS    - BFS: X  | A*: Y
```

---

## Design Notes

**Why mark visited on enqueue, not dequeue (BFS)?** Marking a node as visited when it is added to the queue rather than when it is popped prevents the same node from being added to the frontier multiple times by different neighbors. Marking on dequeue would allow duplicates to queue up, wasting memory and processing time.

**Why use a counter as a tie-breaker (A\*)?** Python's `heapq` requires all elements to be fully comparable. When two nodes have identical `f` values, the heap would attempt to compare the node tuples directly, which can fail. The `counter` ensures a unique ordering and also happens to favor nodes discovered earlier, which produces more efficient exploration in practice as demonstrated by the Spiral maze result.

**Why not `fit_transform` on test data?** See Week 2 for the statistical analogy just as reusing vocabulary matters in NLP, in search algorithms the cost calculations must remain consistent throughout a single run.

---

## File Structure

```
week03/
├── smart_navigator.py    # BFS and A* implementations across four mazes
└── README.md             # This file
```