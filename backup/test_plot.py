import matplotlib.pyplot as plt


federwaage = [30.0,  35.5, 38.0,   50.0]
frequenz =   [78.78, 84.05, 88.05, 101.29]
spannung =   [29.49, 33.56, 36.83, 48.74]

# Ordentliche Messung vom 18.2.26:
federwaage2  = [16.5,  26.0,  37.0,  48.0,  53.0,  57.0 ]
spannung_neu = [15.93, 25.22, 35.24, 46.58, 51.30, 56.09]
spannung_alt = [15.67, 25.49, 35.58, 47.15, 51.72, 56.74]

federwaage2 = [x * 0.981 for x in federwaage2] # gramm zu centinewton

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(federwaage2, spannung_neu, "s", label="new device")
ax.plot(federwaage2, spannung_alt, "X", label="old device")
#ax.plot([0, 1], [0, 1], transform=ax.transAxes, linewidth=0.6, alpha=0.8)
ax.axline((30, 30), slope=1, linewidth=0.6, alpha=0.8, color="gray") 
ax.grid(True)
ax.set_title("Wire Tension Measurement")
ax.set_xlabel("Spring Scale /cN")
ax.set_ylabel("Measured tension / cN")
ax.legend()

plt.show()