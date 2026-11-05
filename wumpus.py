"""
wumpus.py DISCRETE (grid) planner: A* on the 50x50 cell grid

behavior is state machine with three modes
'arson'  (normal game)
      Pick an obstacle that can still be lit, A* to a free cell next to it.
      Any ignitable obstacle in an adjacent cell gets lit as it walks past.
      Target choice: close to me AND far from the truck (the Wumpus can see the truck).
'attack' (endgame: every obstacle is burning or burned)
      Sprint to a point just BEHIND the truck (flanking -- the truck turns slowly).
      While within WUMPUS_BURN_RANGE the truck takes burn damage.
'flee'   (endgame, getting doused)
      Once the truck's water meter on me reaches 50%, run to the free cell
      farthest from the truck. When I'm dry again and far enough away -> 'attack'.
"""
import math
import time
import numpy as np

import config as C
from astar import astar
from world import IGNITABLE

# 8-connected grid moves
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def cell_center(cell):
    return (cell[0] + 0.5) * C.CELL, (cell[1] + 0.5) * C.CELL


def cell_of(x, y):
    ix = min(C.GRID_N - 1, max(0, int(x // C.CELL)))
    iy = min(C.GRID_N - 1, max(0, int(y // C.CELL)))
    return ix, iy


def octile(a, b):
    """Exact grid distance with diagonal moves and no obstacles -> admissible heuristic."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (max(dx, dy) - min(dx, dy) + math.sqrt(2) * min(dx, dy)) * C.CELL


class Wumpus:
    def __init__(self, world):
        self.world = world
        self.free = world.owner < 0                        #true where it can stand
        self.free_cells = [tuple(c) for c in np.argwhere(self.free)]
        self.free_centers = (np.array(self.free_cells) + 0.5) * C.CELL

        self.cell = cell_of(*C.WUMPUS_START)
        self.x, self.y = cell_center(self.cell)
        self.path = []              #cells still to walk through
        self.target = None          #obstacle index heading for (arson mode)
        self.unreachable = set()    #obstacles A* could not reach (grid never changes)
        self.mode = "arson"
        self.alive = True
        self.douse = 0.0            # seconds of water the truck has sprayed on it
        self.replan_timer = 0.0
        self.cpu = 0.0              # CPU seconds spent planning (for the bar chart)

        #for each obstacle: the free cells touching it(where it can stand to light it)
        self.light_spots = []
        for o in world.obstacles:
            spots = set()
            for ix, iy in o.cells:
                for dx, dy in MOVES:
                    c = (ix + dx, iy + dy)
                    if 0 <= c[0] < C.GRID_N and 0 <= c[1] < C.GRID_N and self.free[c]:
                        spots.add(c)
            self.light_spots.append(spots)

    #grid A*
    def grid_neighbors(self, cell):
        ix, iy = cell
        for dx, dy in MOVES:
            nx, ny = ix + dx, iy + dy
            if not (0 <= nx < C.GRID_N and 0 <= ny < C.GRID_N) or not self.free[nx, ny]:
                continue
            if dx and dy and not (self.free[ix + dx, iy] and self.free[ix, iy + dy]):
                continue                                     
            yield (nx, ny), math.hypot(dx, dy) * C.CELL

    def plan_to(self, goal_cells):
        """A* from where I am (or the cell I'm stepping into) to any of goal_cells."""
        t0 = time.process_time()
        goals = set(goal_cells)
        start = self.path[0] if self.path else self.cell
        path = astar(start,
                     is_goal=lambda c: c in goals,
                     neighbors=self.grid_neighbors,
                     heuristic=lambda c: min(octile(c, g) for g in goals))
        self.cpu += time.process_time() - t0
        return path

    def nearest_free_cell(self, x, y):
        i = np.argmin(np.hypot(self.free_centers[:, 0] - x, self.free_centers[:, 1] - y))
        return self.free_cells[i]

    #one time step
    def step(self, t, dt, truck):
        if not self.alive:
            return
        self.light_adjacent(t)
        dist_to_truck = math.hypot(self.x - truck.center()[0], self.y - truck.center()[1])

        #pick a mode
        endgame = self.world.nothing_left_to_ignite() and truck.alive
        old_mode = self.mode
        if not endgame:
            self.mode = "arson"
        elif self.mode == "arson":
            self.mode = "attack"
        elif self.mode == "attack" and self.douse >= C.WUMPUS_FLEE_AT * C.WUMPUS_DOUSE_TIME:
            self.mode = "flee"
        elif self.mode == "flee" and self.douse == 0.0 and dist_to_truck > C.WUMPUS_SAFE_DIST:
            self.mode = "attack"
        if self.mode != old_mode:
            self.replan_timer = 0.0                          # replan right away

        #2. plan
        if self.mode == "arson":
            target_ok = (self.target is not None and
                         self.world.obstacles[self.target].state in IGNITABLE)
            if not target_ok or not self.path:
                self.choose_arson_target(truck)
        else:
            self.replan_timer -= dt
            if self.replan_timer <= 0.0:
                self.replan_timer = C.WUMPUS_REPLAN_PERIOD
                goals = self.flank_cells(truck) if self.mode == "attack" else [self.escape_cell(truck)]
                self.path = self.plan_to(goals) or []

        #move along the path
        speed = C.WUMPUS_SPEED if self.mode == "arson" else C.WUMPUS_SPRINT_SPEED
        self.walk(speed * dt)

    #helpers
    def light_adjacent(self, t):
        """Light every ignitable obstacle in the 8 cells around me."""
        ix, iy = self.cell
        for dx, dy in MOVES:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < C.GRID_N and 0 <= ny < C.GRID_N and self.world.owner[nx, ny] >= 0:
                self.world.ignite(self.world.obstacles[self.world.owner[nx, ny]], t)

    def choose_arson_target(self, truck):
        """Rank ignitable obstacles by (distance to me) - 0.5 * (distance to truck)."""
        tx, ty = truck.center()
        candidates = [o for o in self.world.obstacles
                      if o.state in IGNITABLE and o.idx not in self.unreachable]

        def score(o):
            s = o.distance_to(self.x, self.y)
            if truck.alive:
                s -= C.WUMPUS_AVOID_TRUCK * o.distance_to(tx, ty)
            return s

        for o in sorted(candidates, key=score)[:5]:          # try the best few
            path = self.plan_to(self.light_spots[o.idx]) if self.light_spots[o.idx] else None
            if path:
                self.target, self.path = o.idx, path
                return
            self.unreachable.add(o.idx)
        self.target, self.path = None, []

    def flank_cells(self, truck):
        """
        Where to attack from: free cells close enough to burn the truck, preferring
        the ones BEHIND it (the truck can't turn around quickly to spray them).
        If nothing is close enough, just head for the cell nearest the flank point.
        """
        cx, cy = truck.center()
        dx = self.free_centers[:, 0] - cx
        dy = self.free_centers[:, 1] - cy
        close = np.hypot(dx, dy) <= C.WUMPUS_BURN_RANGE
        behind = close & (dx * math.cos(truck.th) + dy * math.sin(truck.th) < 0)
        chosen = behind if behind.any() else close
        if chosen.any():
            return [self.free_cells[i] for i in np.flatnonzero(chosen)]
        return [self.nearest_free_cell(cx - C.FLANK_DIST * math.cos(truck.th),
                                       cy - C.FLANK_DIST * math.sin(truck.th))]

    def escape_cell(self, truck):
        """Among free cells near it, the one farthest from the truck."""
        cx, cy = truck.center()
        near = np.hypot(self.free_centers[:, 0] - self.x,
                        self.free_centers[:, 1] - self.y) < C.FLEE_SEARCH_RADIUS
        far = np.hypot(self.free_centers[:, 0] - cx, self.free_centers[:, 1] - cy)
        far[~near] = -1.0
        return self.free_cells[int(np.argmax(far))]

    def walk(self, distance):
        """Move `distance` meters along the cell path (straight lines between centers)."""
        while distance > 0 and self.path:
            gx, gy = cell_center(self.path[0])
            d = math.hypot(gx - self.x, gy - self.y)
            if d <= distance:
                self.x, self.y = gx, gy
                self.cell = self.path.pop(0)
                distance -= d
            else:
                self.x += (gx - self.x) / d * distance
                self.y += (gy - self.y) / d * distance
                distance = 0
