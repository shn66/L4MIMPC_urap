import pickle
import numpy as np
import matplotlib.pyplot as plt

file1 = open(f"solve_times_naive.pkl", "rb")
solve_times_naive = pickle.load(file1)

file2 = open(f"solve_times_model.pkl", "rb")
solve_times_model = pickle.load(file2)

dist_naive, times_naive = zip(*solve_times_naive)
dist_model, times_model = zip(*solve_times_model)

print("\nData for solve_times_naive:")
print(f"len: {len(solve_times_naive)}")
print(f"min: {min(times_naive)}")
print(f"med: {np.median(times_naive)}")
print(f"max: {max(times_naive)}")
print(f"avg: {np.mean(times_naive)}")
print(f"std.dev: {np.std(times_naive)}")

print("\nData for solve_times_model:")
print(f"len: {len(solve_times_model)}")
print(f"min: {min(times_model)}")
print(f"med: {np.median(times_model)}")
print(f"max: {max(times_model)}")
print(f"avg: {np.mean(times_model)}")
print(f"std.dev: {np.std(times_model)}")

plt.figure()
plt.scatter(dist_naive, times_naive, label='Naive Solution', color='blue', marker='o')
plt.scatter(dist_model, times_model, label='Model Solution', color='red', marker='x')

plt.title('Solve Times vs Distance from Origin')
plt.xlabel('Distance from Origin (meters)')
plt.ylabel('Solve Time (seconds)')

plt.yscale('log')
plt.legend()
plt.show()