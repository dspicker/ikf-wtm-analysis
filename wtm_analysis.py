import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle


from wtm_data import WtmData

DEBUG = False



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
    file = "WTD-Vibration-20260916-150246.tdms"
    my_data = WtmData(directory + file, wiretype="Anode", wirelength=0.960)
    my_data.start_analysis()
    # my_data.read_archive_tensions(directory + "26061513.13M", 0)
    #plot_pitches_histogram(my_data)
    #plot_wire_positions(my_data)
    #plot_wire_tensions(my_data)
    my_data.to_csv_file(directory+"20260916-150246.csv")
    # my_data.to_dataframe()
    #analyse_single_wire(my_data, 0)
    my_data.export_metadata_json(directory+"20260916-150246.json")

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
