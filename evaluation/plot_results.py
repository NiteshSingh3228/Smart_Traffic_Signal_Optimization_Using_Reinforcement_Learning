
import pandas as pd
import matplotlib.pyplot as plt
import os

# Read evaluation results
df = pd.read_csv("results/metrics/controller_comparison.csv")

# Create graph folder
os.makedirs("results/graphs", exist_ok=True)

# Graph 1: Average Waiting Time
plt.figure(figsize=(10, 5))
plt.bar(df["Controller"], df["Average Waiting Time"])
plt.title("Average Vehicle Waiting Time Comparison")
plt.xlabel("Controller")
plt.ylabel("Waiting Time (seconds)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("results/graphs/waiting_time_comparison.png")
plt.close()

# Graph 2: Average Queue Length
plt.figure(figsize=(10, 5))
plt.bar(df["Controller"], df["Average Queue Length"])
plt.title("Average Queue Length Comparison")
plt.xlabel("Controller")
plt.ylabel("Queue Length (vehicles)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("results/graphs/queue_length_comparison.png")
plt.close()

# Graph 3: Completed Vehicles
plt.figure(figsize=(10, 5))
plt.bar(df["Controller"], df["Completed Vehicles"])
plt.title("Traffic Throughput Comparison")
plt.xlabel("Controller")
plt.ylabel("Completed Vehicles")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("results/graphs/throughput_comparison.png")
plt.close()

# Graph 4: Ambulance Waiting Time
plt.figure(figsize=(10, 5))
plt.bar(
    df["Controller"],
    df["Average Ambulance Waiting Time"]
)
plt.title("Ambulance Waiting Time Comparison")
plt.xlabel("Controller")
plt.ylabel("Waiting Time (seconds)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("results/graphs/ambulance_waiting_comparison.png")
plt.close()

# Graph 5: Average Travel Time (if column exists)
if "Average Travel Time" in df.columns:
    plt.figure(figsize=(10, 5))
    plt.bar(df["Controller"], df["Average Travel Time"])
    plt.title("Average Travel Time Comparison")
    plt.xlabel("Controller")
    plt.ylabel("Travel Time (seconds)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("results/graphs/travel_time_comparison.png")
    plt.close()
    print("5 graphs saved successfully!")
else:
    print("4 graphs saved successfully (no travel time column in CSV).")

print("\nGenerated graph files:")
print(os.listdir("results/graphs"))
