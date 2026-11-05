"""
main.py -- run the 5 simulations and make the deliverables.

    python main.py

Creates, in the results/ folder:
    results.csv       one row per run (scores, counts, CPU times, outcome)
    results.md        the same as a Markdown table + the trophy winner (paste into report)
    cpu_time.png      bar chart: total planning CPU time per player, summed over runs
"""
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window needed
import matplotlib.pyplot as plt

import config as C
from sim import run_simulation

OUT = "results"
TRUCK_BLUE, TRUCK_LIGHT, WUMPUS_ORANGE = "#2a78d6", "#86b6ef", "#eb6834"


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    for run in range(1, C.N_RUNS + 1):
        wall = time.time()
        r, _, _, _ = run_simulation(seed=run)
        r["run"] = run
        results.append(r)
        print(f"run {run}: Wumpus {r['wumpus_score']:4d}  Firetruck {r['truck_score']:4d}  "
              f"-> {r['winner']:9s} ({r['outcome']}, ended at {r['end_time_s']:.0f} s, "
              f"{time.time() - wall:.1f} s wall clock)")

    save_csv(results)
    save_markdown(results)
    plot_cpu(results)
    print(f"\nSaved {OUT}/results.csv, {OUT}/results.md, {OUT}/cpu_time.png")


def save_csv(results):
    keys = ["run"] + [k for k in results[0] if k != "run"]
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)


def champion(results):
    wins = {"Wumpus": 0, "Firetruck": 0}
    for r in results:
        if r["winner"] in wins:
            wins[r["winner"]] += 1
    for name, n in wins.items():
        if n >= 3:
            return name, wins
    return None, wins


def save_markdown(results):
    lines = ["| Run | Wumpus | Firetruck | Winner | Ignited | Burned | Extinguished | Outcome |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['run']} | {r['wumpus_score']} | {r['truck_score']} | {r['winner']} | "
                     f"{r['ignited']} | {r['burned']} | {r['extinguished']} | {r['outcome']} |")
    name, wins = champion(results)
    lines.append("")
    lines.append(f"Wins: Wumpus {wins['Wumpus']}, Firetruck {wins['Firetruck']}")
    lines.append(f"🏆 Champion: **{name}**" if name else "No player won 3 of 5 -- no trophy.")
    text = "\n".join(lines)
    with open(f"{OUT}/results.md", "w") as f:
        f.write(text + "\n")
    print("\n" + text)


def plot_cpu(results):
    wumpus = sum(r["wumpus_cpu_s"] for r in results)
    roadmap = sum(r["truck_roadmap_cpu_s"] for r in results)
    queries = sum(r["truck_cpu_s"] - r["truck_roadmap_cpu_s"] for r in results)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar("Wumpus\n(grid A*)", wumpus, color=WUMPUS_ORANGE, width=0.5)
    ax.bar("Firetruck\n(PRM)", roadmap, color=TRUCK_BLUE, width=0.5, label="roadmap build")
    ax.bar("Firetruck\n(PRM)", queries, bottom=roadmap, color=TRUCK_LIGHT, width=0.5,
           label="path queries", edgecolor="white", linewidth=2)
    ax.text(0, wumpus, f"{wumpus:.2f} s", ha="center", va="bottom")
    ax.text(1, roadmap + queries, f"{roadmap + queries:.2f} s", ha="center", va="bottom")
    ax.set_ylabel("CPU time (s), summed over all runs")
    ax.set_title(f"Planning CPU time per player ({len(results)} runs)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cpu_time.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
