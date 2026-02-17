import matplotlib.pyplot as plt


federwaage = [30.0,  35.5, 38.0,   50.0]
frequenz =   [78.78, 84.05, 88.05, 101.29]
spannung =   [29.49, 33.56, 36.83, 48.74]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(federwaage, spannung, "o")
#ax.plot([0, 1], [0, 1], transform=ax.transAxes, linewidth=0.6, alpha=0.8)
ax.axline((30, 30), slope=1, linewidth=0.6, alpha=0.8, color="orange") 
ax.grid(True)
ax.set_title("Wire Tension Measurement")
ax.set_xlabel("Federwaage /cN")
ax.set_ylabel("Measured tension / cN")

plt.show()