import os.path
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import describe
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import find_peaks, butter, lfilter


DEBUG = False

sampling_rate = 100.0e3  # 100 kHz
sample_duration = 1.5  # seconds


class WtmData:
    def __init__(self, tdms_file_path: str) -> None:
        self.num_wires: int = 0
        self.wire_positions: list[float] = list()
        self.file_path = ""
        self.window_length = 0.0  # Duration of vibration measurement in s
        self._read_tdms(tdms_file_path)

        self.wire_pitches = self._calc_wire_pitches()
        if self.wire_pitches:
            self.wp_statistics = describe(self.wire_pitches[1:]) # achtung erster draht wird ignoriert mit [1:]
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
            self.window_length = tdms_file["Wire_0"]["Power_Spectrum"].properties[
                "Duration_s"
            ]

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

    def get_power_spectrum(self, wire_no: int) -> np.ndarray:
        group_name = f"Wire_{wire_no}"
        spectrum = None
        with TdmsFile.open(self.file_path) as tdms_file:
            spectrum = tdms_file[group_name]["Power_Spectrum"][:]
        spectrum = spectrum[np.isfinite(spectrum)]  # get rid of nan entries
        return spectrum


""" ----------- Analysis Functions ----------- """


def filter_spectrum(spectrum: np.ndarray):
    # comupte running mean over N samples:
    N = 30
    rm_spect = np.convolve(spectrum, np.ones(N) / N, mode="valid")
    return rm_spect


def do_fft(spectrum: np.ndarray):
    sample_spacing = 1.0 / sampling_rate  # seconds
    fft_ampl = fftshift(fft(spectrum, norm="forward"))
    fft_freq = fftshift(fftfreq(spectrum.size, sample_spacing))
    nentries = spectrum.size // 2
    fft_freq = fft_freq[nentries:]  # use only positive half of freq spectrum
    fft_ampl = np.absolute(fft_ampl)[nentries:]
    fft_ampl = fft_ampl / fft_ampl[0]  # normalization
    if DEBUG:
        freq_incr = fft_freq[1] - fft_freq[0]
        print(" -| do_fft()")
        print(f"  | frequency increment = {freq_incr:.3f}")
        print(f"  | entries in frequency spectrum: {nentries}")

    return fft_freq, fft_ampl


def find_frequency(x_vals: np.ndarray, y_vals: np.ndarray):
    search_interval = [40.0, 250.0]  # Min and max frequency in Hz
    search_indices = [
        int(i // (x_vals[1] - x_vals[0])) for i in search_interval
    ]  # indices of interval values
    search_array = y_vals[search_indices[0] : search_indices[1]]
    peaks_idx, _ = find_peaks(search_array, prominence=0.002, wlen=20, distance=50)
    peaks_x = x_vals[peaks_idx + search_indices[0]]

    if DEBUG:
        print(f" -| find_frequency( x_vals[{x_vals.size}], y_vals[{y_vals.size}])")
        peaks_y = y_vals[peaks_idx + search_indices[0]]
        print(f"  | peaks x {peaks_x}")
        print(f"  | peaks y {peaks_y}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, y_vals, ".-", linewidth=0.6)
        ax.plot(peaks_x, peaks_y, "x")
        ax.grid(True)
        ax.set_yscale("log")
        ax.set_xlim(-1.0, 501.0)
        ax.set_title("Find Frequency")
        ax.set_xlabel("Frequency /Hz")
        ax.set_ylabel("Amplitude")
        # plt.show()

    # first peak is the frequency we want
    return float(peaks_x[0]) if peaks_x.size else 0.0


def calculate_wire_tension(frequency: float, harmonic: int = 1):
    wire_rho = 19289.58  # kg / m^3
    wire_radius = 10 * 10**-6  # m
    #wire_rho = 19020.0  # kg / m^3
    #wire_radius = (20.11 / 2 ) * 10**-6  # m
    wire_length = 1.4  # m
    tension = (
        4
        * (frequency**2)
        * (wire_length**2)
        * wire_rho
        * np.pi
        * (wire_radius**2)
        / (harmonic**2)
    )
    return tension  # Newton


def analyse_tension(data: WtmData):
    print("Analysing wire tensions. This may take some time")
    wire_tensions: list[float] = list()
    wire_frequencies = list()
    for i in range(data.num_wires):
        x, y = do_fft(data.get_spectrum(i))
        freq = find_frequency(x, y)
        if freq == 0.0:
            print(f"Could not find frequency for wire No {i}")
        tension = calculate_wire_tension(freq)
        wire_tensions.append(tension)
        wire_frequencies.append(freq)
        # if tension == 0.0:
        #    print(i)

    stats = describe(wire_tensions)
    print("Wire Tension Analysis finished:")
    print(f" mean = {stats.mean:.4f} N, variance = {stats.variance:.5f} N")

    # for i in range(len(wire_tensions)):
    #    if wire_tensions[i] < (0.25) :
    #        print(f"Wire {i}: tension {wire_tensions[i]}")

    return wire_tensions


""" ----------- Plotting Functions ----------- """


def plot_pitches_histogram(data: WtmData):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Wire Pitch Histogram")
    ax.hist(data.wire_pitches, bins=72, range=(-0.125, 5.875))
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)
    ax.set_xlabel("Wire pitch /mm")
    ax.set_ylabel("Count")
    ax.text(
        0.95,
        0.95,
        f"total {data.num_wires} wires\nmean {data.wp_statistics.mean:.3f} mm\nvariance {data.wp_statistics.variance:.3f} mm",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25, -0.125))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    # plt.show()


def plot_wire_positions(data: WtmData):
    fig, (ax_1, ax_2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 6),
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
    ax_1.set_title("Wire Pitch")
    ax_1.grid(True)
    ax_1.set_ylabel("Wire pos. /m")

    ax_2.plot(range(data.num_wires - 1), data.wire_pitches, ".-", linewidth=0.6)
    ax_2.grid(True)
    ax_2.set_ylabel("Wire pitch /mm")
    ax_2.set_xlabel("Wire number")
    # plt.show()


def plot_wire_tensions(wire_tensions: list[float]):
    stats = describe(wire_tensions)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(wire_tensions)), wire_tensions, ".-", linewidth=0.6)
    ax.grid(True)
    ax.set_title("Wire Tension Measurement")
    ax.set_ylabel("Wire tension /N")
    ax.set_xlabel("Wire number")
    ax.text(
        0.25,
        0.25,
        f"total {len(wire_tensions)} wires\n mean = {stats.mean:.4f} N\nvariance = {stats.variance:.5f} N",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}
    )
    # plt.show()


""" -----------     Misc      ----------- """


def test(data: WtmData, wire_no: int):
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
    spect = data.get_spectrum(wire_no)
    spect = filter_spectrum(spect)
    x, y = do_fft(spect)
    freq = find_frequency(x, y)
    tension = calculate_wire_tension(freq)
    print(f" -| analyse_single_wire(wire_no={wire_no})")
    print(f"  | tension = {tension * 100:.2f} cN")
    DEBUG = False


def analyse_signal(data: WtmData, wire_no: int):
    sample_spacing = 1.0 / sampling_rate  # seconds
    spectrum = data.get_spectrum(wire_no)

    #butter_out = butter(4, 5000, fs=100000)
    #b, a = butter_out
    #filtered_spect = lfilter(b, a, spectrum)

    # comupte running mean over N samples:
    #N = 30
    #rm_spect = np.convolve(spectrum, np.ones(N) / N, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = [float(x) * sample_spacing for x in range(0, len(spectrum))]
    #x_filtered = [float(x) * sample_spacing for x in range(0, len(filtered_spect))]
    #x_rm = [float(x) * sample_spacing for x in range(0, len(rm_spect))]
    # y_oszi = [oszillator(x, 92.0) for x in x_vals]
    ax.plot(x_vals, spectrum, "-", linewidth=0.7)
    #ax.plot(x_filtered, filtered_spect, "-", linewidth=0.7)
    #ax.plot(x_rm, rm_spect, "-", linewidth=0.7)
    # ax.plot(x_vals,y_oszi, "-", linewidth=0.5, alpha=0.9)
    ax.grid(True)
    ax.set_title("Sensor Signal")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Signal /V")
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
    #my_data = WtmData("daten_bp1-007_b/WTD-Vibration-20251103-131030.tdms")

    # Einzelner Draht mit Gewicht
    #my_data = WtmData("test_daten/WTD-Vibration-20251030-154035.tdms")

    # Einzelner Draht im Teststand mit Federwaage
    #my_data = WtmData("2026_01_14_test/WTD-Vibration-20260115-170028.tdms")

    #measurements = ["2026_01_14_test/WTD-Vibration-20260114-130738.tdms",
    #                "2026_01_14_test/WTD-Vibration-20260114-132106.tdms",
    #                "2026_01_14_test/WTD-Vibration-20260114-134532.tdms"]
    #
    #my_data = WtmData(measurements[2])

    my_data = WtmData("2026_01 Drahtspannung Testwicklung/WTD-Vibration-20260211-133449.tdms")
    plot_pitches_histogram(my_data)
    plot_wire_positions(my_data)
    tensions = analyse_tension(my_data)
    plot_wire_tensions(tensions)

    # print(
    #    f"Wire tension calculated: 9.81 kg*m/s^2 * 0.0509 kg = {9.81 * 0.0509 * 100:.2f} cN"
    # )

    #wire = 0
    #for wire in range(1):
    #    analyse_single_wire(my_data, wire)
    #    analyse_signal(my_data, wire)
    # test(my_data, wire)

    plt.show()
