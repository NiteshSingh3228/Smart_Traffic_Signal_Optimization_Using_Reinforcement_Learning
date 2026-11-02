
import os
import sys
import traci

# Import our monitoring function
sys.path.append(os.path.abspath("environment"))

from traffic_monitor import get_traffic_data

# Start SUMO simulation
sumo_cmd = [
    "sumo",
    "-n", "sumo/network/intersection.net.xml",
    "-r", "sumo/routes/routes.rou.xml",
    "--end", "100"
]

traci.start(sumo_cmd)

try:
    while traci.simulation.getMinExpectedNumber() > 0:

        # Move simulation forward by one second
        traci.simulationStep()

        # Collect traffic information
        data = get_traffic_data()

        print("\nTraffic Information:")
        print(data)

finally:
    traci.close()
