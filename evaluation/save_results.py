
import csv
import os

results = [
    {
        "Controller": "Fixed-Time",
        "Average Waiting Time": 8.19,
        "Average Queue Length": 1.09,
        "Completed Vehicles": 76,
        "Observed Vehicles": 80,
        "Simulation Time": 600
    },
    {
        "Controller": "Q-Learning",
        "Average Waiting Time": 3.92,
        "Average Queue Length": 0.52,
        "Completed Vehicles": 76,
        "Observed Vehicles": 80,
        "Simulation Time": 600
    }
]

os.makedirs("results/metrics", exist_ok=True)

with open(
    "results/metrics/controller_comparison.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print("Updated comparison results saved successfully!")
