"""
config.py -- every tunable number in one place.

Units: meters, seconds, radians.
Values marked (ASSIGNMENT) come straight from the PDF.
Values marked (CHOICE) are our own design decisions -- mention them in the report.
"""

#world
FIELD_SIZE = 250.0                    # (ASSIGNMENT) square field, 250 m per side
CELL = 5.0                            # (ASSIGNMENT) one tetromino square = 5 m
GRID_N = int(FIELD_SIZE / CELL)       # 50 x 50 cells
COVERAGE = 0.10                       # (ASSIGNMENT) 10% obstacle coverage

TRUCK_START = (15.0, 125.0, 0.0)      # (CHOICE) x, y, heading -- left side, facing right
WUMPUS_START = (235.0, 125.0)         # (CHOICE) right side (opposite the truck)
SPAWN_CLEAR_RADIUS = 20.0             # (CHOICE) keep obstacles away from the start spots

#fire
SPREAD_DELAY = 10.0                   # (ASSIGNMENT) burning 10 s -> ignites neighbors
SPREAD_RADIUS = 30.0                  # (ASSIGNMENT) ...within 30 m
BURN_OUT_TIME = 60.0                  # (CHOICE) burning this long -> "burned" (PDF doesn't say)

#simulation
SIM_TIME = 3600.0                     # (ASSIGNMENT) 3600 s of simulated time
DT = 0.1                              # (CHOICE) time step
N_RUNS = 5                            # (ASSIGNMENT) five runs, different random fields
FRAME_EVERY = 1.0                     # (CHOICE) sim seconds between animation frames

#firetruck 
TRUCK_WIDTH = 2.2                     # (ASSIGNMENT)
TRUCK_LENGTH = 4.9                    # (ASSIGNMENT)
TRUCK_WHEELBASE = 3.0                 # (ASSIGNMENT)
TRUCK_MIN_TURN_RADIUS = 13.0          # (ASSIGNMENT)
TRUCK_MAX_SPEED = 10.0                # (ASSIGNMENT)
TRUCK_ACCEL = 2.0                     # (CHOICE) gentle accel/brake, saves the transmission
EXTINGUISH_RANGE = 10.0               # (ASSIGNMENT) must be within 10 m of the fire...
EXTINGUISH_TIME = 5.0                 # (ASSIGNMENT) ...and stopped for 5 s


# PRM roadmap
PRM_SAMPLES = 1000                    # (CHOICE) number of random poses (x, y, heading)
PRM_K_NEIGHBORS = 12                  # (CHOICE) connect each pose to its 12 "nearest"
                                      #          (nearest = shortest Dubins path)
PRM_MAX_NEIGHBOR_DIST = 50.0          # (CHOICE) only consider poses within 50 m
PRM_MAX_EDGE_LENGTH = 70.0            # (CHOICE) reject loop-de-loop Dubins edges
PRM_CHECK_STEP = 0.5                  # (CHOICE) collision-check every 0.5 m along an edge

#wumpus
WUMPUS_SPEED = 4.0                    # (CHOICE) normal walking speed while setting fires
WUMPUS_SPRINT_SPEED = 12.0            # (CHOICE) endgame speed -- faster than the truck (12)
WUMPUS_AVOID_TRUCK = 0.5              # (CHOICE) arson target choice: prefer obstacles far
                                      #          from the truck (see wumpus.py)
WUMPUS_REPLAN_PERIOD = 0.5            # (CHOICE) endgame: replan every 0.5 s (truck moves)

#endgame (cat & mouse)
# Starts only when every obstacle is burning or burned (nothing left to ignite).
TRUCK_BURN_TIME = 10.0                # (CHOICE) wumpus needs 10 s of contact to burn truck
WUMPUS_DOUSE_TIME = TRUCK_BURN_TIME / 2   # truck douses wumpus twice as fast (5 s)
WUMPUS_BURN_RANGE = 7.5               # (CHOICE) "contact" = within 7.5 m of truck center
WUMPUS_FLEE_AT = 0.5                  # (CHOICE) flee when douse meter hits 50%
WUMPUS_SAFE_DIST = 25.0               # (CHOICE) attack again once this far away and dry
FLANK_DIST = 6.0                      # (CHOICE) fallback aim point: 6 m behind the truck
FLEE_SEARCH_RADIUS = 60.0             # (CHOICE) look this far for an escape cell
