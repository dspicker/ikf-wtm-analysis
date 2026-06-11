from wtm_analysis import WtmData
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


if __name__ == "__main__":
    tdms_filepaths = [
        "data/2026_06_01-Test/WTD-Vibration-20260601-155536.tdms",
        "data/2026_06_01-Test/WTD-Vibration-20260602-133903.tdms",
        "data/2026_06_01-Test/WTD-Vibration-20260602-140015.tdms",
        "data/2026_06_01-Test/WTD-Vibration-20260602-142113.tdms",
    ]

    # Collect data
    n_wires = set()
    dataframes = list()
    for file in tdms_filepaths:
        wtm_data = WtmData(file)
        wtm_data.start_analysis()
        wtm_df = wtm_data.to_dataframe(True)
        n_wires.add(len(wtm_df.index))
        dataframes.append(wtm_df)
        print(" ")

    # All datasets have to be of the same size
    if len(n_wires) != 1:
        print("Error")

    n_wires = int(n_wires.pop())
    tensions = [0.0] * n_wires
    binsizes = [0.0] * n_wires

    # Loop over wires
    for idx in range(n_wires):
        tensions_i = list()
        binsizes_i = list()
        # Loop over datasets
        for df in dataframes:
            tensions_i.append(df.iloc[idx, df.columns.get_loc("wire_tension")])
            binsizes_i.append(df.iloc[idx, df.columns.get_loc("tension_binsize")] / 2.0 )
        if tensions_i and binsizes_i:
            t_mean = np.mean(tensions_i)
            t_std = np.std(tensions_i)
            b_mean = np.mean(binsizes_i)

            if t_std > 0.01:
                # Remove Outliers
                z_t = [np.abs((x - t_mean) / t_std) for x in tensions_i]  # z score
                tensions_i = [x for x, z in zip(tensions_i, z_t) if z < 1.0]
                binsizes_i = [x for x, z in zip(binsizes_i, z_t) if z < 1.0]
                t_mean = np.mean(tensions_i)
                b_mean = np.mean(binsizes_i)

            tensions[idx] = float(t_mean)
            binsizes[idx] = float(b_mean)

    # For this specific set of measurements, remove one wire
    n_wires -= 1
    del tensions[38]
    del binsizes[38]

    # Plotting
    tensions_stats = stats.describe(tensions)
    diff = tensions_stats.minmax[1] - tensions_stats.minmax[0]
    print(f"min,max = {tensions_stats.minmax}, deviation = {diff:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axline(
        (0, 0.47), slope=0, linewidth=0.6, alpha=0.8, color="green", label="Set tension"
    )
    ax.axline(
        (0, tensions_stats.mean),
        slope=0,
        linewidth=0.6,
        alpha=0.8,
        color="orange",
        label="Mean tension",
    )
    ax.errorbar(
        range(n_wires),
        tensions,
        yerr=binsizes,
        fmt="o",
        linewidth=0.6,
        capsize=5.0,
        label="Data",
        zorder=1,
    )
    ax.grid(True)
    ax.set_title("Wire Tension Measurement")
    ax.set_ylabel("Wire tension /N")
    ax.set_xlabel("Wire index")
    ax.text(
        0.21,
        0.15,
        f"total {n_wires} wires\n mean = {tensions_stats.mean:.5f} N\nvariance = {tensions_stats.variance:.5f} N",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    ax.legend()

    plt.show()
