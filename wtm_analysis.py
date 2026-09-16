import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle


from wtm_data import WtmData

DEBUG = False


""" ----------- Plotting Functions ----------- """


def plot_pitches_histogram(data: WtmData, fig_filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Wire Pitch Histogram")
    bin_width = 0.125
    bins = [
        float(x)
        for x in np.arange(
            -bin_width / 2, max(data.wire_pitches) + bin_width, bin_width
        )
    ]
    ax.hist(data.wire_pitches, bins=bins)
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)
    ax.set_xlabel("Wire pitch /mm")
    ax.set_ylabel("Count")
    if data.wp_statistics:
        ax.text(
            0.95,
            0.95,
            f"total {data.num_wires} wires\nmean {data.wp_statistics.mean:.3f} mm\nvariance {data.wp_statistics.variance:.3f} mm",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
        )
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4 * bin_width))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(bin_width))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.set_xlim(-bin_width / 2)

    if fig_filename:
        fig.savefig(fig_filename, bbox_inches="tight")
    # plt.show()


def plot_wire_positions(data: WtmData, fig_filename=None):
    fig, ax_2 = plt.subplots(figsize=(10, 6))
    pitches_x = np.linspace(0.5, data.num_wires - 1.5, data.num_wires - 1)
    ax_2.add_patch(
        Rectangle(
            (0.0, 2.45), float(data.num_wires + 1), 0.1, facecolor="0.8", alpha=0.5
        )
    )
    ax_2.plot(pitches_x, data.wire_pitches, ".-", linewidth=0.6)
    ax_2.set_title("Wire Pitch")
    ax_2.grid(True)
    ax_2.set_ylabel("Wire pitch /mm")
    ax_2.set_xlabel("Wire number")

    if fig_filename:
        fig.savefig(fig_filename, bbox_inches="tight")
    # plt.show()


def plot_wire_tensions(data: WtmData, fig_filename=None):
    if not (data.tensions_stats and data.wire_tensions and data.wire_winlengths):
        print(
            "Error. No wire tensions calculated (yet) in the given instance of WtmData."
        )
        return
    yerrors = [x / 2 for x in data.tensions_binsizes]
    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.errorbar(
    #    range(len(data.wire_tensions)),
    #    data.wire_tensions,
    #    yerr=yerrors,
    #    fmt="o",
    #    linewidth=0.6,
    #    capsize=5.0,
    # )
    ax.axline(
        (0, 0.50), slope=0, linewidth=0.6, alpha=0.8, color="green", label="Set tension"
    )
    ax.axline(
        (0, data.tensions_stats.mean),
        slope=0,
        linewidth=0.6,
        alpha=0.8,
        color="orange",
        label="Mean tension",
    )
    ax.errorbar(
        data.wire_positions,
        data.wire_tensions,
        yerr=yerrors,
        fmt="o",
        linewidth=0.6,
        capsize=5.0,
        label="Data",
        zorder=1,
    )
    if hasattr(data, "archive_tensions"):
        ax.plot(
            data.wire_positions, data.archive_tensions, ".", label="Archive", zorder=2
        )

    ax.grid(True)
    ax.set_title("Wire Tension Measurement")
    ax.set_ylabel("Wire tension /N")
    ax.set_xlabel("Wire position /m")
    ax.text(
        0.21,
        0.15,
        f"total {len(data.wire_tensions)} wires\n mean = {data.tensions_stats.mean:.4f} N\nstd = {np.sqrt(data.tensions_stats.variance):.4f} N",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    ax.legend()

    if fig_filename:
        fig.savefig(fig_filename, bbox_inches="tight")
    # plt.show()


""" -----------     Misc      ----------- """


def plot_power_spectrum(data: WtmData, wire_no: int):
    spec = data.get_power_spectrum(wire_no)
    # peaks, _ = find_peaks(spec, distance=40.0, prominence=40.0)
    # tensions = list()
    # for i in range(len(peaks)):
    #    freq = peaks[i] * 2
    #    tension = calculate_wire_tension(freq, i + 1)
    #    tensions.append(tension)
    #    if i == 1:
    #        break
    # mean_tension = np.mean(tensions)
    # print(tensions)
    # print(mean_tension)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(0, len(spec) * 2, 2), spec, ".-", linewidth=0.6)
    # ax.plot(peaks * 2, spec[peaks], "x")
    ax.set_xlim(0.0, 400.0)
    # plt.show()


def analyse_single_wire(data: WtmData, wire_no: int):
    global DEBUG
    DEBUG = True
    data.debug = True
    spect = data.get_spectrum(wire_no)
    # spect = filter_spectrum(spect)
    x, y = data.do_fft(spect)
    freq, std, harmonic = data.find_frequency(x, y)
    tension, t_err = data.calculate_wire_tension(freq, harmonic)
    print(f" -| analyse_single_wire(wire_no={wire_no})")
    print(f"  | tension = {tension * 100:.2f} cN")
    DEBUG = False
    data.debug = False


def analyse_signal(data: WtmData, wire_no: int):
    sample_spacing = 1.0 / data.sampling_rate  # seconds
    spectrum = data.get_spectrum(wire_no)

    # butter_out = butter(4, 5000, fs=100000)
    # b, a = butter_out
    # filtered_spect = lfilter(b, a, spectrum)

    # comupte running mean over N samples:
    # N = 30
    # rm_spect = np.convolve(spectrum, np.ones(N) / N, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = [float(x) * sample_spacing for x in range(0, len(spectrum))]
    # x_filtered = [float(x) * sample_spacing for x in range(0, len(filtered_spect))]
    # x_rm = [float(x) * sample_spacing for x in range(0, len(rm_spect))]
    # y_oszi = [oszillator(x, 92.0) for x in x_vals]
    ax.plot(x_vals, spectrum, "-", linewidth=0.7)
    # ax.plot(x_filtered, filtered_spect, "-", linewidth=0.7)
    # ax.plot(x_rm, rm_spect, "-", linewidth=0.7)
    # ax.plot(x_vals,y_oszi, "-", linewidth=0.5, alpha=0.9)
    ax.grid(True)
    ax.set_title("Sensor Signal")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Signal /V")
    fig.tight_layout()
    # ax.set_xlim(0.15)

    # print(f"y mean value = {np.mean(spectrum)}")


def oszillator(t: float, freq: float):
    y_offset = 1.55

    r = 5.0
    amplitude = 0.6
    omega = 2 * np.pi * freq
    phi = 300.0 * np.pi / 180.0
    sinus = amplitude * np.cos(omega * t + phi) * np.exp(-t * r)

    return y_offset + sinus


""" ----------- Main Function ----------- """

if __name__ == "__main__":
    # data = WtmData("daten2/WTD-Vibration-20230705-120352.tdms")
    # my_data = WtmData("daten_bp1-007/WTD-Vibration-20251015-102505.tdms")
    # my_data = WtmData("daten_bp1-007/WTD-Vibration-20251016-111956.tdms")

    # zweite Hälfte Goldwicklung
    # my_data = WtmData("data/daten_bp1-007_b/WTD-Vibration-20251103-131030.tdms")
    # my_data.start_analysis()
    # plot_wire_tensions(my_data)

    # Einzelner Draht mit Gewicht
    # my_data = WtmData("test_daten/WTD-Vibration-20251030-154035.tdms")

    # Einzelner Draht im Teststand mit Federwaage
    # my_data = WtmData("2026_01_14_test/WTD-Vibration-20260115-170028.tdms")

    # measurements = ["2026_01_14_test/WTD-Vibration-20260114-130738.tdms",
    #                "2026_01_14_test/WTD-Vibration-20260114-132106.tdms",
    #                "2026_01_14_test/WTD-Vibration-20260114-134532.tdms"]
    #
    # my_data = WtmData(measurements[2])

    directory = "data/2026_09_15-BP1-006/"
    # directory = (
    #    "/Volumes/ikfhep/CBM/Drahtspannungsmessung/Messdaten/2026_06_17-BP1-006/"
    # )
    file = "WTD-Vibration-20260916-125715.tdms"
    my_data = WtmData(directory + file, wiretype="Anode", wirelength=0.960)
    my_data.start_analysis()
    # my_data.read_archive_tensions(directory + "26061513.13M", 0)
    # plot_pitches_histogram(my_data)
    # plot_wire_positions(my_data)
    # plot_wire_tensions(my_data)
    # my_data.to_csv_file(directory+"20260617-180850.csv")
    # my_data.to_dataframe()
    #analyse_single_wire(my_data, 0)
    # my_data.export_metadata_json("test.json")

    #analyse_signal(my_data, 0)
    # plot_pitches_histogram(my_data)
    # plot_wire_positions(my_data)
    # plot_wire_tensions(my_data)

    # print(
    #    f"Wire tension calculated: 9.81 kg*m/s^2 * 0.0509 kg = {9.81 * 0.0509 * 100:.2f} cN"
    # )

    # wire = 0
    # for wire in range(1):
    #    analyse_single_wire(my_data, wire)
    #    analyse_signal(my_data, wire)
    # test(my_data, wire)

    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close("all")
