import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle
from scipy import stats
import os.path


def load_csv_to_df(filepath: str):
    dataframe = pd.read_csv(filepath, index_col="index")
    return dataframe


def plot_wire_tensions(
    data: pd.DataFrame, set_tension=0.5, title_info="", fig_filename: str | None = None
):

    wire_indices = list(data.index)
    tensions_arr = data[["wire_tension"]].to_numpy().flatten()
    binsizes_arr = data[["tension_binsize"]].to_numpy().flatten()
    binsizes_arr /= 2.0

    # if "archive_tensions" in data.columns:

    tensions_stats = stats.describe(tensions_arr)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(
        set_tension,
        linewidth=0.6,
        alpha=0.8,
        color="green",
        label="Set tension",
    )
    ax.axhline(
        tensions_stats.mean,
        linewidth=0.6,
        alpha=0.8,
        color="orange",
        label="Mean tension",
    )
    ax.errorbar(
        wire_indices,
        tensions_arr,
        yerr=binsizes_arr,
        fmt="o",
        linewidth=0.6,
        capsize=5.0,
        label="Data",
        zorder=1,
    )
    ax.text(
        0.21,
        0.15,
        f"total {len(tensions_arr)} wires\n mean = {tensions_stats.mean:.4f} N\n std = {np.sqrt(tensions_stats.variance):.4f} N",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    ax.grid(True)
    ax.set_title("Wire Tension Measurement, " + title_info)
    ax.set_ylabel("Wire tension /N")
    ax.set_xlabel("Wire index")
    ax.legend()
    fig.tight_layout()

    binsize = np.mean(binsizes_arr) / 3.0
    bins = np.arange(
        tensions_stats.minmax[0] - binsize,
        tensions_stats.minmax[1] + binsize,
        binsize * 2.0,
    ).tolist()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(tensions_arr, bins=bins)
    ax2.axvline(
        set_tension,
        linewidth=0.6,
        alpha=0.8,
        color="green",
        label="Set tension",
    )
    ax2.axvline(
        tensions_stats.mean,
        linewidth=0.6,
        alpha=0.8,
        color="orange",
        label="Mean tension",
    )
    ax2.set_title("Wire Tension Distribution, " + title_info)
    ax2.set_ylabel("Wire Count")
    ax2.set_xlabel("Wire tension /N")
    # ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()

    if fig_filename:
        root, ext = os.path.splitext(fig_filename)
        fig.savefig(root + "_01" + ext, bbox_inches="tight")
        fig2.savefig(root + "_02" + ext, bbox_inches="tight")


def plot_pitches_hist(
    data: pd.DataFrame, set_pitch=2.5, title_info="", fig_filename: str | None = None
):
    # Drop the last entry:
    pitches_arr = data[["wire_pitch"]].to_numpy().flatten()[:-1]
    pitches_stats = stats.describe(pitches_arr)
    binsize = 2.5 / 100.
    bins = np.arange(
        0.0 - binsize/2.,
        3.0 + binsize/2.,
        binsize,
    ).tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pitches_arr, bins=bins)
    ax.text(
        0.21,
        0.15,
        f"total {len(data.index)} wires\n mean = {pitches_stats.mean:.4f} N\n std = {np.sqrt(pitches_stats.variance):.4f} N",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)
    ax.set_xlabel("Wire pitch /mm")
    ax.set_ylabel("Count")
    ax.set_title("Wire Pitch Distribution, " + title_info)

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(binsize, binsize/2.))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    #ax.set_xlim(-binsize / 2)

    fig.tight_layout()

    if fig_filename:
            fig.savefig(fig_filename, bbox_inches="tight")


def plot_wire_positions(
    data: pd.DataFrame, title_info="", fig_filename: str | None = None
):

     pitches_arr = data[["wire_pitch"]].to_numpy().flatten()[:-1]

     fig, ax = plt.subplots(figsize=(10, 6))
     ax.add_patch(
         Rectangle(
             (0.0, 2.45), float(pitches_arr.size + 1), 0.1, facecolor="0.8", alpha=0.5
         )
     )
     ax.plot(pitches_arr, ".-", linewidth=0.6)
     ax.set_title("Wire Pitch, " + title_info)
     ax.grid(True)
     ax.set_ylabel("Wire pitch /mm")
     ax.set_xlabel("Wire index")
     fig.tight_layout()

     if fig_filename:
             fig.savefig(fig_filename, bbox_inches="tight")



def plot_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame):
    tensions_before = df_before[["wire_tension"]].to_numpy().flatten()
    positions_before = df_before[["wire_position"]].to_numpy().flatten()
    positions_before -= positions_before[0]

    tensions_after = df_after[["wire_tension"]].to_numpy().flatten()
    positions_after = df_after[["wire_position"]].to_numpy().flatten()
    positions_after -= positions_after[0]

    position_offset = 10 * 0.0025 + 0.0013 # m
    print(f"Offset = {position_offset} m")
    index_offset = 9

    positions_after += position_offset
    index_after = df_after.index.to_numpy(np.int32)
    index_after += index_offset

    residuals = list()
    residuals_pos =list()

    #for idx, x in enumerate(positions_before):
    #    lower = x - 0.0012
    #    upper = x + 0.0012
    #    #print(f"[{lower:.5f}, {upper:.5f}]  {upper-lower}")
    #    i = np.where(np.logical_and(positions_after >= lower,  positions_after <= upper))
    #    if np.size(i)>0 :
    #        residuals.append(tensions_before[idx]-tensions_after[i[0]])
    #        residuals_pos.append(x)
    #    if np.size(i)>1 :
    #        print("multiple matches")
    #print(len(residuals))

    for index in index_after :
         residuals.append(tensions_after[index-index_offset] - tensions_before[index] )

    res_mean = np.mean(residuals)
    print(f"Residuals mean = {res_mean}")

    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,height_ratios=[3,1],figsize=(10, 6))
    #ax1.plot(positions_before,tensions_before, ".-", linewidth=0.6, label="on frame")
    #ax1.plot(positions_after,tensions_after, ".-", linewidth=0.6, label="in chamber")
    #ax2.plot(residuals_pos,residuals, ".")
    ax1.plot(df_before.index.to_numpy(),tensions_before, ".-", linewidth=0.6, label="on frame")
    ax1.plot(index_after,tensions_after, ".-", linewidth=0.6, label="in chamber")
    ax2.plot(index_after,residuals, ".")
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylabel("Wire tension /N")
    ax2.set_xlabel("Wire index")
    ax2.set_ylabel("diff")
    ax1.set_title("Comparison BP1-006")
    ax1.legend()
    fig.tight_layout()



def filter_remove_double(dataframe: pd.DataFrame):
    # nominal_pitch = 2.5  # mm
    pitch_interval = 0.15  # mm
    thresh_double = 0.1  # mm
    # thresh_skipped = nominal_pitch + pitch_interval
    n_before = dataframe.index.size

    # Remove double measured wires:
    dataframe = dataframe[
        dataframe["wire_pitch"] >= (thresh_double + pitch_interval)
    ].reset_index(drop=True)

    n_after = dataframe.index.size
    print(f"Filter removed {n_before - n_after} double measured wires, now {n_after} ")
    return dataframe


def filter_add_skipped(dataframe: pd.DataFrame):
    nominal_pitch = 2.5  # mm
    dataframe.at[dataframe.index[-1], "wire_pitch"] = np.nan
    # n_before = dataframe.index.size
    # print(f"Wires before filter: {n_before}")
    total_skipped = int(0)

    for index, row in dataframe.iterrows():
        index = hash(index) + total_skipped
        if np.isnan(row["wire_pitch"]):
            continue
        n_skipped = round(float(row["wire_pitch"]) / nominal_pitch) - 1
        if n_skipped > 0:
            # print(f"Skipped {n_skipped} wires at {index}")
            # print(dataframe.iloc[index-2:index+4])
            total_skipped += n_skipped
            insert_index = [i + index + 1 for i in range(n_skipped)]
            # print(f"Inserting at {insert_index}")
            new_line = pd.DataFrame({}, index=[insert_index])
            dataframe = pd.concat(
                [dataframe.iloc[: index + 1], new_line, dataframe.iloc[index + 1 :]]
            ).reset_index(drop=True)
            # print(dataframe.iloc[index-n_skipped-2:index+n_skipped+2])

    n_after = dataframe.index.size
    print(f"Filter inserted {total_skipped} wires (containing Nan), now {n_after} ")

    return dataframe


def test():
    wtm1 = load_csv_to_df("data/2026_09_15-BP1-006/WTD-Vibration-20260915-135113_result.csv")
    wtm2 = load_csv_to_df("data/2026_09_15-BP1-006/WTD-Vibration-20260915-155820_result.csv")
    wtm3 = load_csv_to_df("data/2026_09_15-BP1-006/WTD-Vibration-20260916-130434_result.csv")
    wtm4 = load_csv_to_df("data/2026_09_15-BP1-006/WTD-Vibration-20260916-150246_result.csv")

    print("# DF 1 #")
    wtm1 = filter_remove_double(wtm1)
    wtm1 = filter_add_skipped(wtm1)

    print("# DF 2 #")
    wtm2 = filter_remove_double(wtm2)
    wtm2 = filter_add_skipped(wtm2)
    print("# DF 3 #")
    wtm3 = filter_remove_double(wtm3)
    wtm3 = filter_add_skipped(wtm3)
    print("# DF 4 #")
    wtm4 = filter_remove_double(wtm4)
    wtm4 = filter_add_skipped(wtm4)

    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.plot(wtm1["wire_position"],wtm1["wire_tension"], "+-", linewidth=0.6)
    #ax.plot(wtm2["wire_position"],wtm2["wire_tension"], "x-", linewidth=0.6)
    #ax.plot(wtm3["wire_position"],wtm3["wire_tension"], "*-", linewidth=0.6)
    ax.plot(wtm1["wire_position"], "+-", linewidth=0.6)
    ax.plot(wtm2["wire_position"], "x-", linewidth=0.6)
    ax.plot(wtm3["wire_position"], "*-", linewidth=0.6)
    ax.plot(wtm4["wire_position"], ".-", linewidth=0.6)
    ax.grid(True)


if __name__ == "__main__":
    wtm_before = load_csv_to_df("data/2026_06_17-BP1-006/merged_result.csv")
    wtm_after = load_csv_to_df("data/2026_09_15-BP1-006/merged_result.csv")
    plot_before_after(wtm_before,wtm_after)
    #filtered = filter_df(wtm_result)
    #plot_wire_tensions(wtm_result, title_info="Anode, BP1-006")
    #plot_pitches_hist(wtm_result)
    #plot_wire_positions(wtm_result)
    #test()

    #df.to_csv(filename, index_label="index")

    ## Show Plots:
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close("all")
