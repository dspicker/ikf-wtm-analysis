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



if __name__ == "__main__":
    wtm_result = load_csv_to_df("data/2026_09_15-BP1-006/20260916-150246.csv")

    #plot_wire_tensions(wtm_result, title_info="Anode, BP1-006")
    #plot_pitches_hist(wtm_result)
    plot_wire_positions(wtm_result)

    plt.show()
