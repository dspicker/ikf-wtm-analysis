import os.path
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import describe
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import find_peaks


DEBUG = False


class WtmData:
    def __init__(self, tdms_file_path: str) -> None:
        self.num_wires: int = 0
        self.wire_positions: list[float] = list()
        self.file_path = ""
        self._read_tdms(tdms_file_path)

        self.wire_pitches = self._calc_wire_pitches()
        self.wp_statistics = describe(self.wire_pitches)
        print(
            f"Wire pitch: mean={self.wp_statistics.mean:.3f} mm, variance={self.wp_statistics.variance:.3f} mm"
        )

    def _read_tdms(self, tdms_file_path: str):
        tdms_file_path = os.path.abspath(tdms_file_path)
        if os.path.isfile(tdms_file_path):
            print(f"Loading tdms file  {tdms_file_path} ")
            self.file_path = tdms_file_path
        with TdmsFile.read_metadata(tdms_file_path) as tdms_file:
            all_groups = tdms_file.groups()  # One group per wire
            self.num_wires = len(all_groups)
            print(f"File contains data for {self.num_wires} wires.")
            for group in all_groups:
                self.wire_positions.append(group.properties["Position_m"])

    def _calc_wire_pitches(self):
        """Calculates wire pitches in mm

        Returns:
            list[float]: Element 0 is the pitch between wire 0 and 1,
                Element 1 is the pitch between wire 1 and 2 and so on...
        """
        pitches: list[float] = list()
        for i in range(self.num_wires - 1):
            pitch = (self.wire_positions[i + 1] - self.wire_positions[i]) * 1000.0
            pitches.append(pitch)
        return pitches

    def get_spectrum(self, wire_no: int) -> np.ndarray:
        group_name = f"Wire_{wire_no}"
        spectrum = None
        with TdmsFile.open(self.file_path) as tdms_file:
            spectrum = tdms_file[group_name]["AI_Subset"][:]
        spectrum = spectrum[np.isfinite(spectrum)]  # get rid of nan entries
        return spectrum

    def do_fft(self, spectrum: np.ndarray):
        fft_ampl = fftshift(fft(spectrum)).real
        fft_freq = fftshift(fftfreq(spectrum.size, 1.0e-05))  # sampling rate
        return fft_freq, abs(fft_ampl)

    def find_frequency(self, x_vals: np.ndarray, y_vals: np.ndarray):
        peaks, _ = find_peaks(y_vals, height=100.0, prominence=50.0)
        peaks_x = x_vals[peaks]
        peaks_y = y_vals[peaks]
        freq_candidates = list()
        for x in peaks_x:
            if x > 60.0 and x < 200.0:
                freq_candidates.append(x)
        frequenzy = freq_candidates[0] if freq_candidates else 0.0

        if DEBUG:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_vals, y_vals, ".-", linewidth=0.6)
            ax.plot(peaks_x, peaks_y, "x")
            plt.show()

        return frequenzy

    def calculate_wire_tension(self, frequency: float):
        wire_rho = 19300
        wire_radius = 10 * 10**-6
        wire_length = 1.4
        tension = (
            4 * (frequency**2) * (wire_length**2) * wire_rho * np.pi * (wire_radius**2)
        )
        return tension  # Newton


def plot_pitches_histogram(data: WtmData):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Wire Pitch Histogram")
    ax.hist(data.wire_pitches, bins=24, range=(-0.125, 5.875))
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)
    # ax.set_ylabel("Number of wires")
    ax.set_xlabel("Wire pitch /mm")
    ax.set_ylabel("Count")
    ax.text(
        0.95,
        0.95,
        f"total {data.num_wires} wires\nmean {data.wp_statistics.mean:.3f} mm\nvariance {data.wp_statistics.variance:.3f} mm",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25, -0.125))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    plt.show()


def plot_wire_positions(data: WtmData):
    fig, (ax_1, ax_2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(15, 8),
        gridspec_kw={
            "height_ratios": [1, 2],
            "hspace": 0.05,
            "left": 0.07,
            "right": 0.95,
            "top": 0.95,
            "bottom": 0.09,
        },
    )
    ax_1.plot(range(data.num_wires), data.wire_positions)
    ax_2.plot(range(data.num_wires - 1), data.wire_pitches, ".-", linewidth=0.6)

    ax_1.set_title("Wire Pitch")
    ax_1.grid(True)
    ax_1.set_ylabel("Wire pos. /m")

    ax_2.grid(True)
    ax_2.set_ylabel("Wire pitch /mm")
    ax_2.set_xlabel("Wire number")
    plt.show()


def analyse_tension():
    wire_tensions = list()
    for i in range(data.num_wires):
        x, y = data.do_fft(data.get_spectrum(i))
        freq = data.find_frequency(x, y)
        tension = data.calculate_wire_tension(freq)
        wire_tensions.append(tension)
        #if tension == 0.0:
        #    print(i)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.wire_positions, wire_tensions, ".-", linewidth=0.6)
    ax.grid(True)
    ax.set_title("Wire Tension Measurement")
    ax.set_ylabel("Wire tension /N")
    ax.set_xlabel("Wire position /m")
    plt.show()


if __name__ == "__main__":
    data = WtmData("daten2/WTD-Vibration-20230705-120352.tdms")
    # plot_pitches_histogram(data)
    # plot_wire_positions(data)

    # spec = data.get_spectrum(317)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(spec)
    # plt.show()

    # x,y = data.do_fft(data.get_spectrum(317))
    # freq = data.find_frequency(x,y)
    analyse_tension()
