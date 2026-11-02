
import pandas as pd
import matplotlib.pyplot as plt
import os

# Read evaluation results
df = pd.read_csv("results/metrics/controller_comparison.csv")

# Create graph folder
os.makedirs("results/graphs", exist_ok=True)

# Graph 1: Average Waiting Time
plt.figure(figsize=(8, 5))
plt.bar(df["Controller"], df["Average Waiting Time"])
plt.title("Average Waiting Time Comparison")
plt.xlabel("Controller")
plt.ylabel("Waiting Time (seconds)")
plt.tight_layout()
plt.savefig("results/graphs/waiting_time_comparison.png")
plt.close()

# Graph 2: Average Queue Length
plt.figure(figsize=(8, 5))
plt.bar(df["Controller"], df["Average Queue Length"])
plt.title("Average Queue Length Comparison")
plt.xlabel("Controller")
plt.ylabel("Queue Length (vehicles)")
plt.tight_layout()
plt.savefig("results/graphs/queue_length_comparison.png")
plt.close()

# Graph 3: Completed Vehicles
plt.figure(figsize=(8, 5))
plt.bar(df["Controller"], df["Completed Vehicles"])
plt.title("Traffic Throughput Comparison")
plt.xlabel("Controller")
plt.ylabel("Completed Vehicles")
plt.tight_layout()
plt.savefig("results/graphs/throughput_comparison.png")
plt.close()

print("All 3 graphs saved successfully!")

print(os.listdir("results/graphs"))
