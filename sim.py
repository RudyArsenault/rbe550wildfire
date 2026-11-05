"""
sim.py -- runs ONE simulation (one random obstacle field, 3600 s of sim time).

Each time step (DT = 0.1 s):
  1. Wumpus thinks + moves (lights adjacent obstacles)
  2. Truck thinks + moves (sprays if stopped 5 s near fire)
  3. Fires advance (spread after 10 s, burn out after BURN_OUT_TIME)
  4. Endgame contact: Wumpus burns truck / truck douses Wumpus
  5. Stop early if nothing can change any more (steady state)
"""
import math
import numpy as np

import config as C
from world import World, BURNING
from wumpus import Wumpus
from firetruck import FireTruck

STATE_CODE = {"intact": 0, "burning": 1, "extinguished": 2, "burned": 3}


def endgame_contact(world, truck, wumpus, dt):
    """The cat-and-mouse rules. Only active when nothing is left to ignite."""
    if not (truck.alive and wumpus.alive and world.nothing_left_to_ignite()):
        wumpus.douse = max(0.0, wumpus.douse - dt)
        return None
    cx, cy = truck.center()
    d = math.hypot(cx - wumpus.x, cy - wumpus.y)

    if wumpus.mode == "attack" and d <= C.WUMPUS_BURN_RANGE:
        truck.burn += dt                     # 1 s of contact = 1 s of burn damage
        truck.attacked = True                # truck now fights back
    if truck.attacked and d <= C.EXTINGUISH_RANGE:
        wumpus.douse += dt                   # douse needs only half as long (5 s vs 10 s)
    else:
        wumpus.douse = max(0.0, wumpus.douse - dt)   # Wumpus dries off when out of range

    if wumpus.douse >= C.WUMPUS_DOUSE_TIME:
        wumpus.alive = False
        return "wumpus extinguished"
    if truck.burn >= C.TRUCK_BURN_TIME:
        truck.alive = False
        return "truck burned"
    return None


def steady_state(world, truck, wumpus):
    """True when nothing can ever change again, so we may stop early."""
    if world.any_burning():
        return False
    wumpus_can_act = wumpus.alive and (not world.nothing_left_to_ignite() or truck.alive)
    return not wumpus_can_act


def run_simulation(seed, record=False):
    rng = np.random.default_rng(seed)
    world = World(rng)
    truck = FireTruck(world, rng)            # builds the PRM here (timed)
    wumpus = Wumpus(world)

    frames, next_frame = [], 0.0
    outcome, endgame_start = "time limit", None
    t = 0.0
    while t < C.SIM_TIME:
        wumpus.step(t, C.DT, truck)
        truck.step(t, C.DT, wumpus)
        world.update_fire(t)
        event = endgame_contact(world, truck, wumpus, C.DT)
        if event:
            outcome = event
        if endgame_start is None and world.nothing_left_to_ignite() and truck.alive:
            endgame_start = t

        if record and t >= next_frame:
            frames.append(snapshot(t, world, truck, wumpus))
            next_frame += C.FRAME_EVERY
        if steady_state(world, truck, wumpus):
            if outcome == "time limit":
                outcome = "steady state"
            break
        t += C.DT

    if record:
        frames.append(snapshot(t, world, truck, wumpus))
    ws, ts = world.score["wumpus"], world.score["truck"]
    result = {
        "seed": seed,
        "wumpus_score": ws,
        "truck_score": ts,
        "winner": "Wumpus" if ws > ts else ("Firetruck" if ts > ws else "Tie"),
        **world.counts,
        "wumpus_cpu_s": wumpus.cpu,
        "truck_cpu_s": truck.cpu,
        "truck_roadmap_cpu_s": truck.cpu_roadmap,
        "roadmap_nodes": len(truck.roadmap.nodes),
        "roadmap_edges": truck.roadmap.n_edges(),
        "end_time_s": round(t, 1),
        "endgame_start_s": None if endgame_start is None else round(endgame_start, 1),
        "outcome": outcome,
    }
    return result, frames, world, truck


def snapshot(t, world, truck, wumpus):
    """Everything the animation needs to draw one frame (small and cheap)."""
    return {
        "t": t,
        "states": np.array([STATE_CODE[o.state] for o in world.obstacles], dtype=np.int8),
        "truck": (truck.x, truck.y, truck.th, truck.alive, truck.burn),
        "wumpus": (wumpus.x, wumpus.y, wumpus.alive, wumpus.mode, wumpus.douse),
        "score": dict(world.score),
    }
