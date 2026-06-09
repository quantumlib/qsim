import matplotlib.pyplot as plt
import numpy as np

memuse_files_q20 = [
    # baseline qsim and cirq
    "memtotal.733604.dat",
    "memtotal.736557.dat",
    # qsim and cirq with cirq-PR #8054
    "memtotal.742794.dat",
    "memtotal.743107.dat",
]

memuse_files_q24 = [
    # baseline qsim and cirq
    "memtotal.737548.dat",
    "memtotal.738662.dat",
    # qsim and cirq with cirq-PR #8054
    "memtotal.743990.dat",
    "memtotal.745873.dat",
]

t_mem_q20 = [col for f in memuse_files_q20 for col in np.loadtxt(f, unpack=True)]
t_mem_q24 = [col for f in memuse_files_q24 for col in np.loadtxt(f, unpack=True)]

fig, ax = plt.subplots(nrows=2)
lines_q20 = ax[0].plot(*t_mem_q20)
lines_q24 = ax[1].plot(*t_mem_q24)
# blue dashed line with dot markers for baseline cirq
plt.setp(lines_q20[0:2] + lines_q24[0:2], marker=".", linestyle=":", color="blue")
# green solid line with x markers for runs with cirq-PR #8054
plt.setp(lines_q20[2:] + lines_q24[2:], marker="x", markersize=7, color="green")
# hide repeated runs
plt.setp(lines_q20[1::2] + lines_q24[1::2], visible=False)
plt.setp(ax, xlim=(0, 80), ylabel="total memory use (B)")
ax[0].legend(lines_q20[0::2], ["baseline qsim", "qsim with cirq #8054"])
ax[0].set_title("qsim_simulation.py, 20 qubits")
ax[1].set_title("qsim_simulation.py, 24 qubits")
ax[1].set_xlabel("process time (s)")
plt.show()
