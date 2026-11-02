
import traci

from evaluation.fixed_time_controller import (
    run_fixed_time_controller
)

sumo_cmd = [
    "sumo",
    "-n", "sumo/network/intersection.net.xml",
    "-r", "sumo/routes/routes.rou.xml",
    "--no-step-log", "true"
]

traci.start(sumo_cmd)

try:
    results = run_fixed_time_controller()

finally:
    traci.close()
