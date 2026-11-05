"""
animate.py -- make the simulation animation (assignment section 5.7).

    python animate.py --seed 3                       # first 300 s of run 3
    python animate.py --seed 3 --start 150 --duration 200
    python animate.py --seed 3 --endgame             # jump to the cat-and-mouse part

Each frame = 1 s of simulated time. --speed sets frames per second of video
(default 20 -> 20x faster than real time). Saves an .mp4 if ffmpeg is installed,
otherwise a .gif.
"""
import argparse
import math
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
import numpy as np

import config as C
from firetruck import REAR_OVERHANG
from sim import run_simulation

# intact, burning, extinguished, burned
STATE_COLORS = ListedColormap(["#4f8f3a", "#f0441c", "#6ab0f0", "#3a3a3a"])


def truck_outline(x, y, th):
    """Corners of the 4.9 x 2.2 m truck rectangle (pose = rear axle)."""
    back, front, half = -REAR_OVERHANG, C.TRUCK_LENGTH - REAR_OVERHANG, C.TRUCK_WIDTH / 2
    pts = np.array([[back, -half], [front, -half], [front, half], [back, half]])
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return pts @ rot.T + [x, y]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", type=float, default=0.0, help="sim time to start at (s)")
    ap.add_argument("--duration", type=float, default=300.0, help="sim seconds to show")
    ap.add_argument("--endgame", action="store_true", help="start 20 s before the endgame")
    ap.add_argument("--speed", type=int, default=20, help="video frames per second")
    ap.add_argument("--roadmap", action="store_true", help="draw the PRM roadmap faintly")
    args = ap.parse_args()

    print("simulating...")
    result, frames, world, truck = run_simulation(args.seed, record=True)
    print(result)
    start = args.start
    if args.endgame:
        if result["endgame_start_s"] is None:
            print("This run never reached the endgame; showing from --start instead.")
        else:
            start = max(0.0, result["endgame_start_s"] - 20.0)
    frames = [f for f in frames if start <= f["t"] <= start + args.duration]
    print(f"rendering {len(frames)} frames...")

    # map from grid cell -> obstacle index, used to paint obstacle states
    owner = world.owner.T                         # transpose: imshow wants [row=y, col=x]
    fig, ax = plt.subplots(figsize=(7, 7.4))
    ax.set_xlim(0, C.FIELD_SIZE)
    ax.set_ylim(0, C.FIELD_SIZE)
    ax.set_aspect("equal")
    ax.set_facecolor("#e9e4d4")
    if args.roadmap:
        for i, out in enumerate(truck.roadmap.edges):
            for e in out.values():
                ax.plot(e.x, e.y, color="#2a78d6", lw=0.3, alpha=0.25)

    def state_image(states):
        img = np.ma.masked_less(np.where(owner >= 0, states[owner], -1), 0)
        return img

    im = ax.imshow(state_image(frames[0]["states"]), origin="lower", cmap=STATE_COLORS,
                   vmin=0, vmax=3, extent=(0, C.FIELD_SIZE, 0, C.FIELD_SIZE),
                   interpolation="nearest")
    truck_patch = Polygon(truck_outline(*frames[0]["truck"][:3]), closed=True,
                          fc="#2a78d6", ec="black", lw=1, zorder=5)
    ax.add_patch(truck_patch)
    (spray_ring,) = ax.plot([], [], color="#2a78d6", lw=0.8, ls="--", zorder=4)
    (wumpus_dot,) = ax.plot([], [], "o", ms=9, mfc="#8b2fc9", mec="black", zorder=6)
    title = ax.set_title("")
    ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=c, ms=9, label=l)
                       for c, l in zip(STATE_COLORS.colors,
                                       ["intact", "burning", "extinguished", "burned"])]
              + [plt.Line2D([], [], marker="s", ls="", color="#2a78d6", ms=9, label="firetruck"),
                 plt.Line2D([], [], marker="o", ls="", color="#8b2fc9", ms=9, label="Wumpus")],
              loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=6, frameon=False, fontsize=8)
    ring = np.linspace(0, 2 * math.pi, 60)

    def draw(f):
        im.set_data(state_image(f["states"]))
        x, y, th, t_alive, burn = f["truck"]
        wx, wy, w_alive, mode, douse = f["wumpus"]
        truck_patch.set_xy(truck_outline(x, y, th))
        truck_patch.set_facecolor("#2a78d6" if t_alive else "black")
        cx, cy = x + 1.5 * math.cos(th), y + 1.5 * math.sin(th)
        spray_ring.set_data(cx + C.EXTINGUISH_RANGE * np.cos(ring),
                            cy + C.EXTINGUISH_RANGE * np.sin(ring))
        wumpus_dot.set_data([wx], [wy])
        wumpus_dot.set_markerfacecolor("#8b2fc9" if w_alive else "white")
        s = f["score"]
        extra = ""
        if mode != "arson" or burn > 0:
            extra = f"\nWumpus mode: {mode}   truck burn {burn:.1f}/{C.TRUCK_BURN_TIME:.0f} s" \
                    f"   Wumpus douse {douse:.1f}/{C.WUMPUS_DOUSE_TIME:.0f} s"
        title.set_text(f"t = {f['t']:6.0f} s     Wumpus {s['wumpus']}   Firetruck {s['truck']}{extra}")
        return im, truck_patch, spray_ring, wumpus_dot, title

    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    fig.tight_layout()
    if shutil.which("ffmpeg"):
        out = f"wildfire_seed{args.seed}.mp4"
        anim.save(out, writer=animation.FFMpegWriter(fps=args.speed))
    else:
        out = f"wildfire_seed{args.seed}.gif"
        anim.save(out, writer=animation.PillowWriter(fps=args.speed))
    print("saved", out)


if __name__ == "__main__":
    main()
