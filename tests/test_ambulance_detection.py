from environment.traffic_environment import TrafficEnvironment

env = TrafficEnvironment(use_gui=False)

try:
    env.reset(seed=42)

    for i in range(15):

        info = env.get_ambulance_info()

        print("Simulation time:", i)
        print("Ambulance information:", info)

        env.advance_simulation(1)

finally:
    env.close()