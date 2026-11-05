"""
firetruck.py -- the firetruck SAMPLING-BASED planner: a Probabilistic RoadMap (PRM).

Three pieces
CollisionChecker  "Is the truck body hitting anything at these poses?"
                    The truck rectangle is covered by 3 circles. A precomputed
                    clearance map (distance to nearest obstacle) makes each check
                    a simple array lookup.
Roadmap           Built ONCE before the simulation starts:
                      1. sample random collision-free poses (x, y, heading)
                      2. connect each pose to its nearest neighbors with a Dubins
                         path (dubins.py = the local planner, respects the 13 m
                         turning radius), keep the edge if the whole path is free
                    Queries then use the same A* as the Wumpus, but on this graph.
FireTruck         Behavior: pick the most important fire, drive the roadmap to
                    a node within 10 m of it, stop 5 s to extinguish. In the endgame,
                    once the Wumpus attacks, chase the Wumpus instead.

Truck pose (x, y, th) is the REAR AXLE (standard for car-like kinematics).
"""
import math
import time
import numpy as np
from scipy.ndimage import distance_transform_edt

import config as C
import dubins
from astar import astar
from world import BURNING, IGNITABLE

#truck body: rear overhang + wheelbase + front overhang = 4.9 m
REAR_OVERHANG = (C.TRUCK_LENGTH - C.TRUCK_WHEELBASE) / 2           # 0.95 m behind rear axle
BODY_CENTER = C.TRUCK_LENGTH / 2 - REAR_OVERHANG                   # 1.5 m ahead of rear axle
#three circles along the body that together cover the whole 4.9 x 2.2 rectangle
_part = C.TRUCK_LENGTH / 3
CIRCLE_OFFSETS = [-REAR_OVERHANG + _part * (k + 0.5) for k in range(3)]
CIRCLE_RADIUS = math.hypot(_part / 2, C.TRUCK_WIDTH / 2)            # ~1.37 m


class CollisionChecker:
    RES = 0.5   # clearance map resolution (m)

    def __init__(self, world):
        scale = int(C.CELL / self.RES)                                  # 10 pixels per cell
        occupied = np.kron(world.owner >= 0, np.ones((scale, scale), dtype=bool))
        occupied = np.pad(occupied, 1, constant_values=True)            # field edge = wall
        # clearance[i, j] = distance (m) from pixel to nearest obstacle pixel
        self.clearance = distance_transform_edt(~occupied) * self.RES
        self.max_index = occupied.shape[0] - 1

    def poses_free(self, xs, ys, ths):
        #true if the truck body is collision-free at every pose in the arrays
        cos, sin = np.cos(ths), np.sin(ths)
        for off in CIRCLE_OFFSETS:
            ix = np.clip(((xs + off * cos) / self.RES).astype(int) + 1, 0, self.max_index)
            iy = np.clip(((ys + off * sin) / self.RES).astype(int) + 1, 0, self.max_index)
            # + RES margin because the lookup pixel can be up to RES away from the point
            if np.any(self.clearance[ix, iy] <= CIRCLE_RADIUS + self.RES):
                return False
        return True


class Edge:
    """A drivable roadmap edge: the sampled Dubins path from one node to another."""
    __slots__ = ("to", "length", "s", "x", "y", "th")

    def __init__(self, to, length, s, x, y, th):
        self.to, self.length, self.s, self.x, self.y, self.th = to, length, s, x, y, th


class Roadmap:
    def __init__(self, checker, rng, start_pose):
        self.nodes = [tuple(start_pose)]           # node 0 = where the truck starts
        while len(self.nodes) < C.PRM_SAMPLES:     # 1. sample free poses
            x, y = rng.uniform(0, C.FIELD_SIZE, size=2)
            th = rng.uniform(-math.pi, math.pi)
            if checker.poses_free(np.array([x]), np.array([y]), np.array([th])):
                self.nodes.append((x, y, th))
        self.xy = np.array([n[:2] for n in self.nodes])

        # 2. connect neighbors. Dubins paths are one-way (truck drives forward),
        #    so the graph is DIRECTED: edges[i][j] is the path from i to j.
        #    "Nearest" is measured in Dubins path length, not straight-line distance:
        #    a pose 5 m away but facing the wrong way needs an 80 m loop to reach,
        #    while one 30 m straight ahead is an easy drive.
        #    Each node gets its K best edges OUT (i -> j) and its K best edges IN (j -> i).
        self.edges = [dict() for _ in self.nodes]
        for i in range(len(self.nodes)):
            d = np.hypot(*(self.xy - self.xy[i]).T)
            nearby = [int(j) for j in np.flatnonzero(d <= C.PRM_MAX_NEIGHBOR_DIST) if j != i]
            self._connect_best([(i, j) for j in nearby], checker)    # outgoing
            self._connect_best([(j, i) for j in nearby], checker)    # incoming

        # 3. prune. Because edges are one-way, the truck could drive into a node it can
        #    never leave. Keep only nodes it can reach from the start AND get back from.
        reverse = [[] for _ in self.nodes]                     # edges flipped around
        for a in range(len(self.nodes)):
            for b in self.edges[a]:
                reverse[b].append(a)
        self.usable = self._reachable(self.edges) & self._reachable(reverse)
        for i in range(len(self.nodes)):
            if i not in self.usable:
                self.edges[i] = {}
            else:
                self.edges[i] = {j: e for j, e in self.edges[i].items() if j in self.usable}

    def _connect_best(self, pairs, checker):
        """Of the candidate (a, b) pairs, add the K shortest collision-free Dubins edges."""
        R = C.TRUCK_MIN_TURN_RADIUS
        options = []
        for a, b in pairs:
            segments, length = dubins.shortest_path(self.nodes[a], self.nodes[b], R)
            if length <= C.PRM_MAX_EDGE_LENGTH:
                options.append((length, a, b, segments))
        options.sort(key=lambda o: o[0])
        for length, a, b, segments in options[:C.PRM_K_NEIGHBORS]:
            if b in self.edges[a]:
                continue                                       # already have this edge
            s, x, y, th = dubins.sample_path(self.nodes[a], segments, R, C.PRM_CHECK_STEP)
            if checker.poses_free(x, y, th):                   # whole path must be free
                self.edges[a][b] = Edge(b, length, s, x, y, th)

    @staticmethod
    def _reachable(adjacency):
        """All nodes reachable from node 0 (simple depth-first search)."""
        seen, stack = {0}, [0]
        while stack:
            for j in adjacency[stack.pop()]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return seen

    def n_edges(self):
        return sum(len(e) for e in self.edges)


class FireTruck:
    def __init__(self, world, rng):
        self.world = world
        t0 = time.process_time()                       #roadmap time counts as planning CPU
        self.checker = CollisionChecker(world)
        self.roadmap = Roadmap(self.checker, rng, C.TRUCK_START)
        rm = self.roadmap
        #body-center position of every roadmap node
        th = np.array([n[2] for n in rm.nodes])
        self.node_centers = rm.xy + BODY_CENTER * np.column_stack([np.cos(th), np.sin(th)])
        #for each obstacle: the roadmap nodes close enough to spray it
        self.spray_nodes = [
            {i for i in rm.usable
             if o.distance_to(*self.node_centers[i]) <= C.EXTINGUISH_RANGE}
            for o in world.obstacles]
        self.cpu_roadmap = time.process_time() - t0
        self.cpu = self.cpu_roadmap                    #total planning CPU (roadmap + queries)

        self.x, self.y, self.th = C.TRUCK_START
        self.node = 0                # roadmap node we are at / last passed
        self.route = []              # list of Edges still to drive
        self.edge_s = 0.0            # meters driven along route[0]
        self.speed = 0.0
        self.stopped_time = 0.0      # how long we've been standing still
        self.target = None           # obstacle index, or "wumpus"
        self.retry_timer = 0.0       # wait a bit after a failed plan
        self.alive = True
        self.attacked = False        # has the Wumpus started burning us (endgame)?
        self.burn = 0.0              # seconds of burn damage taken

    def center(self):
        return (self.x + BODY_CENTER * math.cos(self.th),
                self.y + BODY_CENTER * math.sin(self.th))

    #roadmap query
    def plan(self, is_goal, heuristic):
        """A* on the roadmap from the current node. Returns a list of Edges or None."""
        t0 = time.process_time()
        edges = self.roadmap.edges
        nodes = astar(self.node, is_goal,
                      neighbors=lambda i: ((j, e.length) for j, e in edges[i].items()),
                      heuristic=heuristic)
        self.cpu += time.process_time() - t0
        if nodes is None:
            return None
        return [edges[a][b] for a, b in zip(nodes[:-1], nodes[1:])]

    def plan_to_fire(self, o):
        goals = self.spray_nodes[o.idx]
        xy = self.roadmap.xy
        # straight-line distance minus a safe margin (obstacle size + 10 m range + body)
        h = lambda i: max(0.0, math.hypot(*(xy[i] - o.center)) - 25.0)
        return self.plan(lambda i: i in goals, h)

    def plan_to_wumpus(self, wumpus):
        w = np.array([wumpus.x, wumpus.y])
        reach = C.EXTINGUISH_RANGE - 1.0
        return self.plan(lambda i: math.hypot(*(self.node_centers[i] - w)) <= reach,
                         lambda i: max(0.0, math.hypot(*(self.roadmap.xy[i] - w)) - reach - 3.0))

    #one time step
    def step(self, t, dt, wumpus):
        if not self.alive:
            return
        if not self.world.nothing_left_to_ignite():
            self.attacked = False                     # endgame over -> back to firefighting
        hunting = self.attacked and wumpus.alive

        if not self.route:
            self.retry_timer -= dt
            self.decide(t, hunting, wumpus)           # parked at a node: choose what next
        if self.route:
            self.stopped_time = 0.0
            self.drive(dt, hunting)
        else:
            self.speed = 0.0
            self.stopped_time += dt
            if self.stopped_time >= C.EXTINGUISH_TIME:
                self.spray()
                self.stopped_time = 0.0

    def decide(self, t, hunting, wumpus):
        """Parked at a roadmap node. Stay (spraying) or plan a new route."""
        if hunting:
            self.target = "wumpus"
            cx, cy = self.center()
            if math.hypot(cx - wumpus.x, cy - wumpus.y) > C.EXTINGUISH_RANGE - 1.0:
                self.route = self.plan_to_wumpus(wumpus) or []
            return

        # Still waiting next to a burning target? Keep standing still.
        if (self.target not in (None, "wumpus")
                and self.world.obstacles[self.target].state == BURNING
                and self.node in self.spray_nodes[self.target]):
            return
        if self.retry_timer > 0:
            return
        self.target = None
        for o in self.rank_fires(t)[:3]:              # try the best few fires
            route = self.plan_to_fire(o)
            if route is not None:
                self.target, self.route = o.idx, route
                return
        self.retry_timer = 1.0                        # nothing reachable; try again in 1 s

    def rank_fires(self, t):
        """
        Most important fire first. score = distance - 15 * (threatened neighbors).
        A fire that hasn't spread yet and has lots of unburned neighbors is worth a
        detour. Skip fires that will burn out before we can get there.
        """
        cx, cy = self.center()
        ranked = []
        for o in self.world.obstacles:
            if o.state != BURNING or not self.spray_nodes[o.idx]:
                continue
            d = o.distance_to(cx, cy)
            time_left = C.BURN_OUT_TIME - (t - o.t_ignited)
            if d / C.TRUCK_MAX_SPEED + C.EXTINGUISH_TIME > time_left:
                continue
            threatened = 0 if o.has_spread else sum(
                1 for j in o.neighbors if self.world.obstacles[j].state in IGNITABLE)
            ranked.append((d - C.TRUCK_THREAT_WEIGHT * threatened, o.idx))
        return [self.world.obstacles[i] for _, i in sorted(ranked)]

    def drive(self, dt, hunting):
        """Follow the route at up to 10 m/s, accelerating/braking at TRUCK_ACCEL."""
        remaining = self.route[0].length - self.edge_s + sum(e.length for e in self.route[1:])
        self.speed = min(self.speed + C.TRUCK_ACCEL * dt,          # speed up gently
                         C.TRUCK_MAX_SPEED,                         # speed limit
                         math.sqrt(2 * C.TRUCK_ACCEL * remaining))  # brake to stop at the end
        self.speed = max(self.speed, 0.5)                           # always creep to the end
        self.edge_s += self.speed * dt

        while self.route and self.edge_s >= self.route[0].length:   # reached a node
            self.edge_s -= self.route[0].length
            self.node = self.route.pop(0).to
            if self.route and self.should_replan(hunting):
                self.route = []
        if not self.route:                                          # parked exactly on node
            self.edge_s = 0.0
            self.x, self.y, self.th = self.roadmap.nodes[self.node]
            return
        e = self.route[0]                                           # pose along the edge
        self.x = float(np.interp(self.edge_s, e.s, e.x))
        self.y = float(np.interp(self.edge_s, e.s, e.y))
        self.th = float(np.interp(self.edge_s, e.s, e.th))

    def should_replan(self, hunting):
        #At each node: replan if the Wumpus moved, or our fire is no longer burning
        if hunting:
            return True
        if self.target == "wumpus":
            return True                                             # chase is over
        return self.target is None or self.world.obstacles[self.target].state != BURNING

    def spray(self):
        """Stopped for 5 s: extinguish every burning obstacle within 10 m."""
        cx, cy = self.center()
        for o in self.world.obstacles:
            if o.state == BURNING and o.distance_to(cx, cy) <= C.EXTINGUISH_RANGE:
                self.world.extinguish(o)
