
import os
import sys
import shutil

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

for possible_path in [
    "/Library/Frameworks/EclipseSUMO.framework/Versions/1.28.0/EclipseSUMO/share/sumo/tools",
    os.path.abspath(".venv/lib/python3.10/site-packages"),
]:
    if os.path.exists(possible_path) and possible_path not in sys.path:
        sys.path.append(possible_path)

import traci

from environment.traffic_monitor import get_traffic_data


def _get_sumo_binary():
    framework = (
        "/Library/Frameworks/EclipseSUMO.framework"
        "/Versions/1.28.0/EclipseSUMO/bin/sumo"
    )
    if os.path.exists(framework):
        return framework
    resolved = shutil.which("sumo")
    return resolved if resolved else "sumo"


# Start SUMO simulation for 10 steps to verify traffic monitor
sumo_cmd = [
    _get_sumo_binary(),
    "-n", "sumo/network/intersection.net.xml",
    "-r", "sumo/routes/routes.rou.xml",
    "--no-step-log", "true"
]

traci.start(sumo_cmd)

try:
    for _ in range(10):
        traci.simulationStep()
        data = get_traffic_data()
        assert isinstance(data, dict)
        assert "N_in" in data
        assert "queue_length" in data["N_in"]

    print("Traffic monitor test passed successfully!")
    print("Sample data:", data)

finally:
    traci.close()
