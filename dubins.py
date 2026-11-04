"""
dubins.py LOCAL PLANNER for firetruck's PRM

truck drives forward with a minimum turning radius R = 13 m, so 
shortest such path is "Dubins path": turn - straight - turn.

  L = turn left at radius R,  R = turn right at radius R,  S = straight

only use four "turn-straight-turn" words (LSL, RSR, LSR, RSL)
full Dubins set also has LRL and RLR, which only matter when two poses are very close together
leaving them out just means a few roadmap edges are slightly longer.\

Geometry, in words:
  1. Each pose has a left circle and a right circle it could turn on.
  2. Pick one circle at the start and one at the goal.
  3. Find the straight line that is tangent to both circles (in the right direction).
  4. Path = arc on circle 1 -> tangent line -> arc on circle 2.
  5. Do this for all four combinations and keep the shortest.
"""
import math
import numpy as np

TWO_PI = 2.0 * math.pi


def _circle_center(x, y, th, turn, R):
    """Center of the left ('L') or right ('R') turning circle of pose (x, y, th)."""
    if turn == "L":
        return x - R * math.sin(th), y + R * math.cos(th)
    return x + R * math.sin(th), y - R * math.cos(th)


def _arc_angle(th_from, th_to, turn):
    """How far (radians, 0..2pi) we turn to go from heading th_from to th_to."""
    if turn == "L":
        return (th_to - th_from) % TWO_PI          # left = counter-clockwise
    return (th_from - th_to) % TWO_PI              # right = clockwise


def shortest_path(start, goal, R):
    """
    start, goal: (x, y, heading).
    Returns (segments, total_length), where segments is
    [(turn1, length1), ('S', length2), (turn2, length3)] in meters.
    """
    best = None
    for t1, t2 in (("L", "L"), ("R", "R"), ("L", "R"), ("R", "L")):
        c1x, c1y = _circle_center(*start, t1, R)
        c2x, c2y = _circle_center(*goal, t2, R)
        vx, vy = c2x - c1x, c2y - c1y
        D = math.hypot(vx, vy)                     # distance between circle centers

        if t1 == t2:
            # Same turn direction: OUTER tangent, parallel to the line between centers.
            straight = D
            heading = math.atan2(vy, vx) if D > 1e-9 else start[2]
        else:
            # Opposite turns: INNER tangent, crosses between the circles.
            if D < 2 * R:
                continue                           # circles overlap -> no inner tangent
            straight = math.sqrt(D * D - 4 * R * R)
            tilt = math.atan2(2 * R, straight)
            heading = math.atan2(vy, vx) + (tilt if t1 == "L" else -tilt)

        arc1 = R * _arc_angle(start[2], heading, t1)
        arc2 = R * _arc_angle(heading, goal[2], t2)
        length = arc1 + straight + arc2
        if best is None or length < best[1]:
            best = ([(t1, arc1), ("S", straight), (t2, arc2)], length)
    return best


def sample_path(start, segments, R, step):
    """
    Turn a Dubins path into poses every `step` meters.
    Returns numpy arrays (s, x, y, th): distance along path, position, heading.
    These are exact kinematic formulas for driving an arc or a line.
    """
    x, y, th = start
    S, X, Y, TH = [np.array([0.0])], [np.array([x])], [np.array([y])], [np.array([th])]
    s_offset = 0.0
    for kind, length in segments:
        if length < 1e-9:
            continue
        n = max(2, int(math.ceil(length / step)) + 1)
        s = np.linspace(0.0, length, n)[1:]        # skip 0 (it's the previous end point)
        if kind == "S":
            xs, ys, ths = x + s * math.cos(th), y + s * math.sin(th), np.full_like(s, th)
        elif kind == "L":
            ths = th + s / R
            xs = x + R * (np.sin(ths) - math.sin(th))
            ys = y + R * (math.cos(th) - np.cos(ths))
        else:  # "R"
            ths = th - s / R
            xs = x + R * (math.sin(th) - np.sin(ths))
            ys = y + R * (np.cos(ths) - math.cos(th))
        S.append(s_offset + s); X.append(xs); Y.append(ys); TH.append(ths)
        x, y, th = xs[-1], ys[-1], ths[-1]
        s_offset += length
    return np.concatenate(S), np.concatenate(X), np.concatenate(Y), np.concatenate(TH)
