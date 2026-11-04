"""
astar.py generic A* search, used by Wampus and firetruck

 Wumpus uses it on the 50x50 grid (nodes are cells)
the firetruck uses it on the PRM roadmap (nodes are roadmap poses)

A*  needs three things from caller, passed in as functions:
  neighbors(node)  -> yields (next_node, step_cost)
  is_goal(node)    -> True if we are done
  heuristic(node)  -> optimistic guess of the remaining cost (never too big)
"""
import heapq
import itertools


def astar(start, is_goal, neighbors, heuristic):
    #return cheapest path - list of nodes
    tie = itertools.count()                      #breaks ties so the heap never compares nodes
    open_heap = [(heuristic(start), next(tie), start)]
    cost_so_far = {start: 0.0}                   #g(n): best known cost from start to n
    came_from = {start: None}
    closed = set()

    while open_heap:
        _, _, node = heapq.heappop(open_heap)    #node with the smallest f = g + h
        if node in closed:
            continue                             #stale heap entry, skip it
        if is_goal(node):
            path = []                            # walk the parent links backwards
            while node is not None:
                path.append(node)
                node = came_from[node]
            return path[::-1]
        closed.add(node)

        for nxt, step_cost in neighbors(node):
            new_cost = cost_so_far[node] + step_cost
            if new_cost < cost_so_far.get(nxt, float("inf")):
                cost_so_far[nxt] = new_cost
                came_from[nxt] = node
                heapq.heappush(open_heap, (new_cost + heuristic(nxt), next(tie), nxt))
    return None                                  #goal unreachable
