"""
visualize_stages.py -- see each building block on its own, BEFORE running the game.

    python visualize_stages.py --seed 1

Saves stages.png with four panels:
  1. the obstacle field (stage 1: world.py)
  2. the Wumpus's grid and one A* path (stage 2: wumpus.py + astar.py)
  3. the PRM roadmap (stage 3: firetruck.py + dubins.py)
  4. one truck path planned on the roadmap, with the truck body drawn along it
Good for checking things work, and for figures in your report.
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from world import World
from wumpus import Wumpus, cell_center
from firetruck import FireTruck
from animate import truck_outline


def draw_obstacles(ax, world):
    for o in world.obstacles:
        for ix, iy in o.cells:
            ax.add_patch(plt.Rectangle((ix * C.CELL, iy * C.CELL), C.CELL, C.CELL,
                                       fc="#4f8f3a", ec="#2f5f22", lw=0.4))
    ax.set_xlim(0, C.FIELD_SIZE)
    ax.set_ylim(0, C.FIELD_SIZE)
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = World(rng)
    truck = FireTruck(world, rng)
    wumpus = Wumpus(world)
    rm = truck.roadmap

    # a target that is far from both players, so the example paths are interesting
    target = max(world.obstacles, key=lambda o: min(o.distance_to(*C.TRUCK_START[:2]),
                                                     o.distance_to(*C.WUMPUS_START)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    a1, a2, a3, a4 = axes.ravel()

    # 1. obstacle field
    draw_obstacles(a1, world)
    a1.plot(*C.TRUCK_START[:2], "s", color="#2a78d6", ms=10, label="truck start")
    a1.plot(*C.WUMPUS_START, "o", color="#8b2fc9", ms=10, label="Wumpus start")
    cover = 100 * sum(len(o.cells) for o in world.obstacles) / C.GRID_N ** 2
    a1.set_title(f"1. Obstacle field: {len(world.obstacles)} tetrominoes, {cover:.1f}% coverage")
    a1.legend(loc="upper center")

    # 2. Wumpus grid A*
    draw_obstacles(a2, world)
    for k in range(C.GRID_N + 1):
        a2.axhline(k * C.CELL, color="0.85", lw=0.3)
        a2.axvline(k * C.CELL, color="0.85", lw=0.3)
    path = wumpus.plan_to(wumpus.light_spots[target.idx])
    xy = np.array([cell_center(c) for c in path])
    a2.plot(xy[:, 0], xy[:, 1], "-o", color="#8b2fc9", ms=3, lw=2)
    a2.add_patch(plt.Circle(target.center, 6, fill=False, ec="red", lw=2))
    a2.set_title(f"2. Wumpus: A* on 5 m grid ({len(path)} cells, {wumpus.cpu * 1000:.1f} ms)")

    # 3. PRM roadmap
    draw_obstacles(a3, world)
    for out in rm.edges:
        for e in out.values():
            a3.plot(e.x, e.y, color="#2a78d6", lw=0.3, alpha=0.5)
    usable = np.array(sorted(rm.usable))
    a3.plot(rm.xy[usable, 0], rm.xy[usable, 1], ".", color="#184f95", ms=2)
    a3.set_title(f"3. PRM: {len(rm.usable)} usable poses, {rm.n_edges()} Dubins edges "
                 f"({truck.cpu_roadmap:.2f} s to build)")

    # 4. one truck query
    draw_obstacles(a4, world)
    before = truck.cpu
    route = truck.plan_to_fire(target)
    a4.add_patch(plt.Circle(target.center, 6, fill=False, ec="red", lw=2))
    if route:
        for e in route:
            a4.plot(e.x, e.y, color="#2a78d6", lw=2)
            for k in range(0, len(e.x), 12):          # truck body every ~6 m
                a4.add_patch(plt.Polygon(truck_outline(e.x[k], e.y[k], e.th[k]),
                                         fc="none", ec="#184f95", lw=0.6))
        length = sum(e.length for e in route)
        a4.set_title(f"4. Truck: roadmap query, {len(route)} edges, {length:.0f} m "
                     f"({(truck.cpu - before) * 1000:.1f} ms)")
    else:
        a4.set_title("4. Truck: no roadmap path to that obstacle")

    fig.tight_layout()
    fig.savefig("stages.png", dpi=120)
    print("saved stages.png")


if __name__ == "__main__":
    main()
