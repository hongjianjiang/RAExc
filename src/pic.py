# Three separate figures labeled as subdiagrams (a), (b), (c)
# (Matplotlib only, no subplots, no specific colors)

import matplotlib.pyplot as plt
import numpy as np

# # # -----------------------------
# # # (a) Runtime vs Maximum Length
# # # -----------------------------
# max_len = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
# time_ls = [2.46, 3.99, 4.56, 8.11, 12.65, 14.41, 18.05, 20.30, 30.59, 41.48]
# time_smt = [6.52, 11.53, 12.88, 18.49, 24.22, 25.11, 32.70, 32.53, 32.66, 41.18]

# plt.figure()
# plt.plot(max_len, time_ls, marker='o')
# plt.plot(max_len, time_smt, marker='s')
# plt.xlabel("Maximum Length")
# plt.ylabel("Time (seconds)")
# plt.title("Runtime vs Maximum Length")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(["LS", "SMT"])
# plt.show()


# # -----------------------------
# # (b) Runtime vs Number of States
# # -----------------------------
# states = np.arange(2, 11)
# smt_time_20 = np.array([3.74, 5.81, 8.83, 12.47, 42.74, 57.38, 64.28, 70.01, 300])
# ls_time_20 = np.array([1.92, 4.34, 5.20, 16.12, 33.69,  91.33, 120, 120, 120])
# ac_time_20 = np.array([2,5,6,7,11,14,22,32, 300])

# plt.figure()
# plt.plot(states, smt_time_20, marker='o')
# plt.plot(states, ls_time_20, marker='s')
# plt.plot(states, ac_time_20, marker='^')
# plt.xlabel("Number of States")
# plt.ylabel("Time (seconds)")
# plt.title("Runtime vs Number of States (MaxLen = 50)")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(["SMT", "LS", "ACL"])
# plt.show()


# # -----------------------------
# # (c) Runtime vs Number of Registers
# # -----------------------------
# registers = [1,2,3,4,5,6,7,8,9,10]
# times_smt = [2.50, 12.27, 37.57, 90.55, 312.18, 535.44, 676.67, 820.80, 1308.75, 1901.56]
# times_ls  = [1.87, 4.50, 3.30, 3.48, 3.70, 9.07, 150.44, 2000, 2000, 2000]

# plt.figure()
# plt.plot(registers, times_smt, marker='o')
# plt.plot(registers, times_ls, marker='s')
# plt.xlabel("Number of Registers")
# plt.ylabel("Time (seconds)")
# plt.title("Runtime vs Number of Registers (MaxLen = 10)")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(["SMT", "LS"])
# plt.show()

# -----------------------------
# (d) Avg Runtime per Vector vs Number of States
# -----------------------------
states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
avg_ms = [0.87, 28.81, 87.14, 176.25, 299.28, 455.31,
          643.21, 865.15, 1118.37, 1419.95, 1726.86]

plt.figure()
plt.plot(states, avg_ms, marker='o')
plt.xlim(1, 23)
plt.xticks(range(2, 23, 2))
plt.xlabel("Number of States")
plt.ylabel("Average Time per Vector (ms)")
plt.title("Average Runtime per Vector vs Number of States")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# -----------------------------
# (e) Avg Runtime per Vector vs Number of Registers
# -----------------------------
regs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
avg_per_vector_ms = [
    39.71, 135.69, 313.79, 568.82, 997.31,
    1522.20, 2213.33, 2447.11, 2470.22,
    2400.95, 2404.89
]

plt.figure()
plt.plot(regs, avg_per_vector_ms, marker='o')
plt.xlim(1, 23)
plt.xticks(range(2, 23, 2))
plt.xlabel("Number of Registers")
plt.ylabel("Average Time per Vector (ms)")
plt.title("Average Runtime per Vector vs Number of Registers")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()