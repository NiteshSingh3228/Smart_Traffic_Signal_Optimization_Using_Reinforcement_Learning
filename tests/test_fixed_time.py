
import sys
import os
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

from evaluation.fixed_time_controller import (
    run_fixed_time_controller
)


def _get_sumo_binary():
    framework = (
        "/Library/Frameworks/EclipseSUMO.framework"
        "/Versions/1.28.0/EclipseSUMO/bin/sumo"
    )
    if os.path.exists(framework):
        return framework
    resolved = shutil.which("sumo")
    return resolved if resolved else "sumo"


sumo_cmd = [
    _get_sumo_binary(),
    "-n", "sumo/network/intersection.net.xml",
    "-r", "sumo/routes/routes.rou.xml",
    "--no-step-log", "true"
]

traci.start(sumo_cmd)

try:
    results = run_fixed_time_controller()
    assert results is not None
    assert "average_waiting_time" in results
    print("Fixed-time test passed successfully! Results:", results)
finally:
    traci.close()
