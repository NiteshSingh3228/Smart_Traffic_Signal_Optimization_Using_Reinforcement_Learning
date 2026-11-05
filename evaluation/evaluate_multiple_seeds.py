
import csv
import os
import statistics

from evaluation.evaluate_controllers import (
    evaluate_fixed_time,
    evaluate_q_learning,
    evaluate_dqn
)

# FIX 6: Use seeds outside training range (training uses 1000+)
SEEDS = [42, 123, 200, 300, 400]

METRICS = [
    "average_waiting_time",
    "average_travel_time",
    "average_queue_length",
    "completed_vehicles",
    "observed_vehicles",
    "average_ambulance_waiting_time",
    "maximum_ambulance_waiting_time",
    "ambulances_observed",
    "simulation_time"
]

results = {
    "Fixed-Time": [],
    "Fixed-Time + Ambulance Priority": [],
    "Q-Learning + Ambulance Priority": [],
    "DQN + Ambulance Priority": []
}

for seed in SEEDS:

    print("\n" + "=" * 55)
    print("RUNNING ALL CONTROLLERS WITH SEED:", seed)
    print("=" * 55)

    results["Fixed-Time"].append(
        evaluate_fixed_time(False, seed)
    )

    results["Fixed-Time + Ambulance Priority"].append(
        evaluate_fixed_time(True, seed)
    )

    results["Q-Learning + Ambulance Priority"].append(
        evaluate_q_learning(seed)
    )

    results["DQN + Ambulance Priority"].append(
        evaluate_dqn(seed)
    )

os.makedirs("results/metrics", exist_ok=True)

# Save individual run results
with open(
    "results/metrics/controller_multi_seed_runs.csv",
    "w",
    newline=""
) as file:

    fields = ["Controller", "Seed"] + METRICS
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for controller, runs in results.items():

        for seed, metrics in zip(SEEDS, runs):

            row = {
                "Controller": controller,
                "Seed": seed
            }

            row.update(metrics)
            writer.writerow(row)

# Save mean and standard deviation
with open(
    "results/metrics/controller_multi_seed_summary.csv",
    "w",
    newline=""
) as file:

    fields = ["Controller", "Metric", "Mean", "Std Dev"]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for controller, runs in results.items():

        for metric in METRICS:

            values = [run[metric] for run in runs]

            writer.writerow({
                "Controller": controller,
                "Metric": metric,
                "Mean": round(statistics.mean(values), 3),
                "Std Dev": round(statistics.pstdev(values), 3)
            })

print("\n========== MULTIPLE-SEED EVALUATION COMPLETED ==========")
print("Individual results: results/metrics/controller_multi_seed_runs.csv")
print("Summary results: results/metrics/controller_multi_seed_summary.csv")
