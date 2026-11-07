
import sys
import os

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

from evaluation.metrics import TrafficMetrics

# Test 1: Callback and initialization
updates = []


def on_metric_update(snap):
    updates.append(snap)


metrics = TrafficMetrics(on_update=on_metric_update)
assert metrics.on_update is on_metric_update

# Test 2: Results computation logic
metrics.vehicle_waiting_times = {"v1": 10.0, "v2": 20.0}
metrics.vehicle_travel_times = [25.0, 35.0]
metrics.total_queue_length = 15
metrics.total_samples = 3
metrics.completed_vehicles = 2
metrics.ambulance_waiting_times = {"amb1": 5.0, "amb2": 15.0}

results = metrics.get_results()
assert results["average_waiting_time"] == 15.0, f"Expected 15.0, got {results['average_waiting_time']}"
assert results["average_travel_time"] == 30.0, f"Expected 30.0, got {results['average_travel_time']}"
assert results["average_queue_length"] == 5.0, f"Expected 5.0, got {results['average_queue_length']}"
assert results["completed_vehicles"] == 2
assert results["average_ambulance_waiting_time"] == 10.0
assert results["maximum_ambulance_waiting_time"] == 15.0
assert results["ambulances_observed"] == 2

print("TrafficMetrics unit tests passed successfully!")
print("Results verified:", results)
