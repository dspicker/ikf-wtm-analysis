import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from wtm_plot import plot_wire_tensions, filter_add_skipped, filter_remove_double


def merge_mean_csv(csv_filepaths: list[str]):
    # Collect data
    n_wires = set()
    dataframes = list()

    for file in csv_filepaths:
        wtm_df = pd.read_csv(file, index_col="index")
        wtm_df = filter_remove_double(wtm_df)
        #wtm_df = filter_add_skipped(wtm_df)
        n_wires.add(len(wtm_df.index))
        dataframes.append(wtm_df)

    # All datasets have to be of the same size
    if len(n_wires) != 1:
        print("Error")

    n_wires = int(n_wires.pop())
    tensions = [0.0] * n_wires
    binsizes = [0.0] * n_wires
    positions = [0.0] * n_wires

    # Loop over wires
    for idx in range(n_wires):
        tensions_i = list()
        binsizes_i = list()
        positions_i = list()
        # Loop over datasets
        for df in dataframes:
            tensions_i.append(df.iloc[idx, df.columns.get_loc("wire_tension")])
            binsizes_i.append(df.iloc[idx, df.columns.get_loc("tension_binsize")])
            positions_i.append(df.iloc[idx, df.columns.get_loc("wire_position")])
        if tensions_i and binsizes_i:
            t_mean = np.nanmean(tensions_i)
            t_std = np.nanstd(tensions_i)
            b_mean = np.nanmean(binsizes_i)

            if t_std > 0.01:
                # Remove Outliers
                z_t = [np.abs((x - t_mean) / t_std) for x in tensions_i]  # z score
                tensions_i = [x for x, z in zip(tensions_i, z_t) if z < 1.0]
                binsizes_i = [x for x, z in zip(binsizes_i, z_t) if z < 1.0]
                t_mean = np.nanmean(tensions_i)
                b_mean = np.nanmean(binsizes_i)

            tensions[idx] = float(t_mean)
            binsizes[idx] = float(b_mean)

        if positions_i:
            p_mean = np.nanmean(positions_i)
            positions[idx] = float(p_mean)

    tensions_stats = stats.describe(tensions)
    diff = tensions_stats.minmax[1] - tensions_stats.minmax[0]
    print(f"Tension range [{tensions_stats.minmax[0]:.3f}, {tensions_stats.minmax[1]:.3f}] deviation = {diff:.4f}")

    new_dataframe = pd.DataFrame(
        {
            "wire_position": positions,
            "wire_tension": tensions,
            "tension_binsize": binsizes,
        }
    )
    return new_dataframe


if __name__ == "__main__":
    # tdms_filepaths = [
    #    "data/2026_06_01-Test/WTD-Vibration-20260601-155536.tdms",
    #    "data/2026_06_01-Test/WTD-Vibration-20260602-133903.tdms",
    #    "data/2026_06_01-Test/WTD-Vibration-20260602-140015.tdms",
    #    "data/2026_06_01-Test/WTD-Vibration-20260602-142113.tdms",
    # ]
    # for file in tdms_filepaths:
    #    wtm_data = WtmData(file)
    #    wtm_data.start_analysis()
    #    wtm_df = wtm_data.to_dataframe(True)
    #    n_wires.add(len(wtm_df.index))
    #    dataframes.append(wtm_df)
    #    print(" ")

    directory = (
        #"data/2026_09_15-BP1-006/"
        "data/2026_06_17-BP1-006/"
    )

    #csv_filepaths = [
    #    directory + "WTD-Vibration-20260915-135113_result.csv",
    #    directory + "WTD-Vibration-20260915-155820_result.csv",
    #    directory + "WTD-Vibration-20260916-130434_result.csv",
    #    directory + "WTD-Vibration-20260916-150246_result.csv",
    #]

    csv_filepaths = [
        directory + "20260617-102341.csv",
        directory + "20260617-140755.csv",
        directory + "20260617-180850.csv",
    ]

    my_dataframe = merge_mean_csv(csv_filepaths)

    plot_wire_tensions(
        my_dataframe,
        0.5,
        "Anode" + ", " + "BP1-006",
        directory + "tensions_merged.png",
    )

    my_dataframe.to_csv(directory + "merged_result.csv", index_label="index")

    ## Show Plots:
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close("all")
