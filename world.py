"""
world.py the obstacle field and the fire rules.
builds a random field of tetromino obstacles (10% coverage), and
runs the fire rules: intact -> burning -> (spreads after 10 s) -> burned,
unless the truck extinguishes it first.

Grid convention used: cell (ix, iy) covers x in [ix*CELL, (ix+1)*CELL),  y in [iy*CELL, (iy+1)*CELL)
and arrays are indexed  grid[ix, iy]  (x first).
"""
import numpy as np
import config as C

# the four obstacle states from the assignment
INTACT, BURNING, EXTINGUISHED, BURNED = "intact", "burning", "extinguished", "burned"
IGNITABLE = (INTACT, EXTINGUISHED)          # burned obstacles cant be relit

#the 7 standard tetrominoes, as (dx, dy) cell offsets
SHAPES = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "L": [(0, 0), (0, 1), (0, 2), (1, 0)],
    "J": [(1, 0), (1, 1), (1, 2), (0, 0)],
}


def rotate(cells, quarter_turns):
    for _ in range(quarter_turns):
        cells = [(-y, x) for x, y in cells]
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return [(x - min_x, y - min_y) for x, y in cells]


class Obstacle:

    def __init__(self, idx, cells):
        self.idx = idx
        self.cells = cells                                         # list of (ix, iy)
        self.centers = (np.array(cells) + 0.5) * C.CELL            # cell centers, meters
        self.center = self.centers.mean(axis=0)
        self.state = INTACT
        self.t_ignited = None        #sim time it last started burning
        self.has_spread = False      #did this fire already ignite its neighbors
        self.neighbors = []          # indices of obstacles within SPREAD_RADIUS

    def distance_to(self, x, y):
        half = C.CELL / 2
        dx = np.maximum(np.abs(self.centers[:, 0] - x) - half, 0.0)
        dy = np.maximum(np.abs(self.centers[:, 1] - y) - half, 0.0)
        return float(np.min(np.hypot(dx, dy)))


class World:
    def __init__(self, rng):
        # owner[ix, iy] = index of obstacle in that cell, or -1 if cell is free
        self.owner = -np.ones((C.GRID_N, C.GRID_N), dtype=int)
        self.obstacles = []
        self._place_tetrominoes(rng)
        self._find_fire_neighbors()
        self.score = {"wumpus": 0, "truck": 0}
        self.counts = {"ignited": 0, "burned": 0, "extinguished": 0}

    #building the field
    def _place_tetrominoes(self, rng):
        #drop random tetrominoes on free cells until 10% of the field is covered
        target_cells = int(np.ceil(C.COVERAGE * C.GRID_N * C.GRID_N))
        filled = 0
        names = list(SHAPES)
        while filled < target_cells:
            shape = rotate(SHAPES[names[rng.integers(len(names))]], int(rng.integers(4)))
            ox, oy = rng.integers(0, C.GRID_N, size=2)
            cells = [(ox + dx, oy + dy) for dx, dy in shape]
            if all(self._cell_ok(ix, iy) for ix, iy in cells):
                for ix, iy in cells:
                    self.owner[ix, iy] = len(self.obstacles)
                self.obstacles.append(Obstacle(len(self.obstacles), cells))
                filled += 4

    def _cell_ok(self, ix, iy):
        if not (0 <= ix < C.GRID_N and 0 <= iy < C.GRID_N) or self.owner[ix, iy] >= 0:
            return False
        x, y = (ix + 0.5) * C.CELL, (iy + 0.5) * C.CELL
        for sx, sy in (C.TRUCK_START[:2], C.WUMPUS_START):
            if np.hypot(x - sx, y - sy) < C.SPAWN_CLEAR_RADIUS:
                return False
        return True

    def _find_fire_neighbors(self):
        for a in self.obstacles:
            for b in self.obstacles:
                if a is b:
                    continue
                diff = a.centers[:, None, :] - b.centers[None, :, :]
                if np.min(np.hypot(diff[..., 0], diff[..., 1])) <= C.SPREAD_RADIUS:
                    a.neighbors.append(b.idx)

    #fire rules
    def ignite(self, obstacle, t):
        #set an obstacle on fire. Wumpus scores 1 point per ignition
        if obstacle.state in IGNITABLE:
            obstacle.state = BURNING
            obstacle.t_ignited = t
            obstacle.has_spread = False
            self.score["wumpus"] += 1
            self.counts["ignited"] += 1

    def extinguish(self, obstacle):
        #truck puts out a burning obstacle. Truck scores 2 points
        if obstacle.state == BURNING:
            obstacle.state = EXTINGUISHED
            self.score["truck"] += 2
            self.counts["extinguished"] += 1

    def update_fire(self, t):
        #advance every fire: spread after SPREAD_DELAY, burn out after BURN_OUT_TIME
        for o in self.obstacles:
            if o.state != BURNING:
                continue
            burning_for = t - o.t_ignited
            if not o.has_spread and burning_for >= C.SPREAD_DELAY:
                o.has_spread = True
                for j in o.neighbors:
                    self.ignite(self.obstacles[j], t)
            if burning_for >= C.BURN_OUT_TIME:
                o.state = BURNED                       #wumpus gets a bonus point
                self.score["wumpus"] += 1
                self.counts["burned"] += 1

    #helper endgame
    def any_burning(self):
        return any(o.state == BURNING for o in self.obstacles)

    def nothing_left_to_ignite(self):
        #true when every obstacle is burning or burned -> the endgame may start
        return all(o.state in (BURNING, BURNED) for o in self.obstacles)
