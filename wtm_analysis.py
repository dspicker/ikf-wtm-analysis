import sys
import os.path
import numpy as np
import pandas as pd
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle
from scipy.stats import describe
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from windingmachine.archive import get_tension
from tqdm import tqdm
import re
import datetime as dt


DEBUG = False


class WtmData:
    """Class that represents the measurement results of a wire vibration measurement."""

    sampling_rate = 100.0e3  # 100 kHz
    available_wire_types = ("Anode", "Cathode")

    # Pitch threshold in mm
    # Below this value we assume that one wire got measured twice by the machine
    pitch_threshold = 0.2

    def __init__(
        self, tdms_file_path: str, wiretype: str = "Anode", wirelength: float = 1.4
    ) -> None:
        """Create a new instance of WtmData

        Args:
            tdms_file_path (str): path to tdms vibration file
            wiretype (str, optional): Cathode or Anode. Defaults to "Anode".
            wirelength (float, optional): Length of the wires in m. Defaults to 1.4.
        """
        self.file_path = ""

        if wiretype not in self.available_wire_types:
            print(f"Error. Wire type must be one of:  {self.available_wire_types}")
            sys.exit()
        self.wire_type: str = wiretype
        self.wire_length = wirelength

        self.num_wires: int = 0
        # Data-precision 1 micrometer, real ca. 10 micrometer:
        self.wire_positions: list[float] = list()
        # Duration of vibration measurement in s:
        self.wire_winlengths: list[float] = list()
        self.wire_pitches: list[float] = list()
        self.double_measured: list[int] = list()
        self.wp_statistics = None

        self._read_tdms_metadata(tdms_file_path)
        self._calc_wire_pitches()

        self.wire_tensions: list[float] = list()
        self.tensions_binsizes: list[float] = list()
        self.tensions_stats = None

    def _read_tdms_metadata(self, tdms_file_path: str):
        """Check if file is there and read necessary metadata

        Args:
            tdms_file_path (str): file path given to the constructor
        """
        tdms_file_path = os.path.abspath(tdms_file_path)
        if not os.path.isfile(tdms_file_path):
            print(f"Error. File not found:  {tdms_file_path}")
            sys.exit()
        print(f"Loading tdms file  {tdms_file_path} ")
        self.file_path = tdms_file_path
        with TdmsFile.read_metadata(tdms_file_path) as tdms_file:
            tdms_name = str(tdms_file.properties["name"])
            datetime_re = re.search(r"WTD-Vibration-(\d{8}-\d{6})", tdms_name)
            if datetime_re:
                self.datetime_measured = dt.datetime.strptime(
                    datetime_re.group(1), "%Y%m%d-%H%M%S"
                )

            all_groups = tdms_file.groups()  # One group per wire
            self.num_wires = len(all_groups)
            print(f"File contains data for {self.num_wires} wires.")
            for group in all_groups:
                self.wire_positions.append(group.properties["Position_m"])
                self.wire_winlengths.append(
                    group["Power_Spectrum"].properties["Duration_s"]
                )
                # wf_start_time = group['AI_Subset'].properties["wf_start_time"]
        # print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _calc_wire_pitches(self):
        """Calculate wire pitches in mm

        Returns:
            list[float]: Element 0 is the pitch between wire 0 and 1,
                element 1 is the pitch between wire 1 and 2 and so on...
        """
        self.wire_pitches.clear()
        for i in range(self.num_wires - 1):
            pitch = (self.wire_positions[i + 1] - self.wire_positions[i]) * 1000.0
            if pitch < self.pitch_threshold:
                print(f"* Wires {i} and {i + 1} possible double Measurement! *")
                print(
                    f"  at {self.wire_positions[i] * 1000.0:.4f} mm and {self.wire_positions[i + 1] * 1000.0:.4f}"
                )
            self.wire_pitches.append(pitch)

        self.wp_statistics = describe(self.wire_pitches)
        print(
            f"Wire pitch: mean={self.wp_statistics.mean:.3f} mm, variance={self.wp_statistics.variance:.3f} mm"
        )

    def to_dataframe(self, filter: bool = False):
        """Return analysis result as pandas dataframe

        Args:
            filter (bool, optional): If double measurements should be removed from the result. Defaults to False.

        Returns:
            pandas.DataFrame: Tension analysis results
        """
        if not (self.tensions_stats and self.wire_tensions and self.wire_winlengths):
            print(
                "Error. No wire tensions calculated (yet) in the given instance of WtmData."
            )
            return pd.DataFrame()
        pitches = self.wire_pitches
        pitches.append(self.wire_pitches[-1])
        dataframe = pd.DataFrame(
            {
                "wire_position": self.wire_positions,
                "wire_pitch": pitches,
                "wire_tension": self.wire_tensions,
                "tension_binsize": self.tensions_binsizes,
            }
        )
        if hasattr(self, "archive_tensions"):
            dataframe["archive_tension"] = self.archive_tensions
        if filter:
            # Filter out the double measured wires
            dataframe = dataframe[
                dataframe["wire_pitch"] >= self.pitch_threshold
            ].reset_index(drop=True)
        return dataframe

    def to_csv_file(self, filename: str):
        df = self.to_dataframe(True)
        df.to_csv(filename, index_label="index")

    def get_spectrum(self, wire_no: int) -> np.ndarray:
        """Read sensor data for the given wire from the file

        Args:
            wire_no (int): Wire index

        Returns:
            np.ndarray: Raw data 1D array with sampled sensor values
        """
        group_name = f"Wire_{wire_no}"
        spectrum = None
        with TdmsFile.open(self.file_path) as tdms_file:
            spectrum = tdms_file[group_name]["AI_Subset"][:]
        spectrum = spectrum[np.isfinite(spectrum)]  # get rid of nan entries
        return spectrum

    def get_power_spectrum(self, wire_no: int) -> np.ndarray:
        """Read fft result for the given wire from the file

        Fourier transform is also done in Labview and stored here.

        Args:
            wire_no (int): Wire index

        Returns:
            np.ndarray: 1D Array with result of fourier transform
        """
        group_name = f"Wire_{wire_no}"
        spectrum = None
        with TdmsFile.open(self.file_path) as tdms_file:
            spectrum = tdms_file[group_name]["Power_Spectrum"][:]
        spectrum = spectrum[np.isfinite(spectrum)]  # get rid of nan entries
        return spectrum

    def read_archive_tensions(self, archive_file: str, frame_no: int = 0):
        """Read archive file produced by the winding machine

        Args:
            archive_file (str): Path to the archive file that was created when winding.
            frame_no (int): Select if the upper or lower frame got measured. 0=upper

        Write results to self.archive_tensions on success.
        """
        archive_tensions = get_tension(archive_file, frame_no)
        archive_tensions = [float(i) / 100.0 for i in archive_tensions]

        if len(archive_tensions) == len(self.wire_tensions):
            self.archive_tensions = archive_tensions
            print("Successsfully read archive file")
        else:
            print(
                " read_archive_tensions() : Error. Datasets have different number of entries."
            )

    """ ----------- Analysis Functions ----------- """

    def filter_spectrum(self, spectrum: np.ndarray):
        # compute running mean over N samples:
        N = 30
        rm_spect = np.convolve(spectrum, np.ones(N) / N, mode="same")
        return rm_spect

    def do_fft(self, spectrum: np.ndarray):
        """Perform fast fourier transform on given data

        Args:
            spectrum (np.ndarray): Sensor data

        Returns:
            tuple(ndarray,ndarray): Frequqenzy and Amplitude values of fft result
        """
        sample_spacing = 1.0 / self.sampling_rate  # seconds
        fft_ampl = fftshift(fft(spectrum, norm="forward"))
        fft_freq = fftshift(fftfreq(spectrum.size, sample_spacing))
        nentries = spectrum.size // 2
        fft_freq = fft_freq[nentries:]  # use only positive half of freq spectrum
        fft_ampl = np.absolute(fft_ampl)[nentries:]
        fft_ampl = fft_ampl / fft_ampl[0]  # normalization
        if DEBUG:
            freq_incr = fft_freq[1] - fft_freq[0]
            print(" -| do_fft()")
            print(f"  | frequency increment = {freq_incr:.3f} Hz")
            print(f"  | entries in frequency spectrum: {nentries}")

        return fft_freq, fft_ampl

    def find_frequency(self, x_vals: np.ndarray, y_vals: np.ndarray):
        """Find peak in the fft output and try to fit it

        Args:
            x_vals (np.ndarray): frequency output of fft
            y_vals (np.ndarray): amplitude output of fft

        Returns:
            tuple(float,float, int): frequency, std, harmonic

        Returns (0.0, 0.0, 0) if no peaks are found. Then it may be necessary
        to adjust `search_interval` and the parameters of `find_peaks`
        """
        search_interval = [40.0, 350.0]  # Min and max frequency in Hz
        search_indices = [
            int(i // (x_vals[1] - x_vals[0])) for i in search_interval
        ]  # indices of interval values
        search_array = y_vals[search_indices[0] : search_indices[1]]
        peaks_idx, _ = find_peaks(search_array, prominence=0.002, wlen=20, distance=50)
        peaks_x = x_vals[peaks_idx + search_indices[0]]
        peaks_y = y_vals[peaks_idx + search_indices[0]]

        fig = None
        ax = None
        if DEBUG:
            print(f" -| find_frequency( x_vals[{x_vals.size}], y_vals[{y_vals.size}])")
            print(f"  | peaks x {peaks_x}")
            print(f"  | peaks y {peaks_y}")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_vals, y_vals, ".-", linewidth=0.6)
            ax.plot(peaks_x, peaks_y, "x")
            ax.grid(True)
            ax.set_yscale("log")
            ax.set_xlim(-1.0, 501.0)
            ax.set_title("Fast Fourier Transform of Signal")
            ax.set_xlabel("Frequency /Hz")
            ax.set_ylabel("Amplitude")
            fig.tight_layout()

        frequency: float = 0.0
        freq_std: float = 0.0
        harmonic: int = 0
        if peaks_x.size >= 2:
            harmonic = 2
        elif peaks_x.size == 1:
            harmonic = 1

        # Try to fit the second harmonic, if that fails try the first harmonic.
        # If that also fails, return just the peak position of the first peak.
        # If no peaks where found at all, return zero
        while harmonic > 0:
            p_x = float(peaks_x[harmonic - 1])
            p_y = float(peaks_y[harmonic - 1])
            initial_params = np.array([p_y, p_x, 10])
            try:
                params, pcov = curve_fit(
                    self._gauss_func, x_vals, y_vals, initial_params
                )
            except RuntimeError:  # Fit failed
                harmonic -= 1  # try next harmonic
                tqdm.write(" Fit in FFT failed. ")
                if harmonic == 0:  # if this was already the last try, return peak pos
                    frequency = float(peaks_x[0])
                    harmonic = 1
                    break
            else:  # Fit succeeded
                perr = np.sqrt(np.diag(pcov))
                frequency = float(params[1])
                freq_std = float(params[2])

                if DEBUG and ax:
                    fit_x = x_vals[
                        int(params[1] - 5 * params[2]) : int(params[1] + 5 * params[2])
                    ]
                    fit_y = self._gauss_func(fit_x, *params)
                    ax.plot(fit_x, fit_y)
                    print(f"  | FFT-Fit succeeded for harmonic {harmonic} ")
                    print(
                        f"  | mu    = {float(params[1]):.3f} +- {float(perr[1]):.5f} Hz"
                    )
                    print(
                        f"  | sigma = {float(params[2]):.4f} +- {float(perr[2]):.6f} Hz"
                    )
                break

        return (frequency, freq_std, harmonic)

    def calculate_wire_tension(
        self, frequency: float, harmonic: int = 2, freq_std: float = 0.0
    ):
        if self.wire_type == "Anode":
            wire_rho = 19289.58  # kg / m^3
            wire_radius = 10.055 * 10**-6  # m
        elif self.wire_type == "Cathode":
            wire_rho = 8230.22  # kg / m^3
            wire_radius = 37.5 * 10**-6  # m
        else:
            wire_rho = 0.0  # kg / m^3
            wire_radius = 0.0  # m

        wire_length = self.wire_length  # m
        tension = (
            4
            * (frequency**2)
            * (wire_length**2)
            * wire_rho
            * np.pi
            * (wire_radius**2)
            / (harmonic**2)
        )

        error = 0.0
        if freq_std > 0.0:
            error = abs(
                8
                * frequency
                * (wire_length**2)
                * wire_rho
                * np.pi
                * (wire_radius**2)
                * freq_std
                / (harmonic**2)
            )
        return (tension, error)  # Newton

    def analyse_tension(self, wire_no: int):
        """Analyse data of one wire

        Args:
            wire_no (int): Wire index

        Returns:
            tuple[float,float]: Tension of the wire in newton, binszize in newton
        """
        spectrum = self.get_spectrum(wire_no)
        x, y = self.do_fft(spectrum)
        freq, std, harmonic = self.find_frequency(x, y)
        if freq == 0.0:
            print(f"Could not find frequency for wire No {wire_no}")
            return (0.0, 0.0)
        tension, errorbar = self.calculate_wire_tension(freq, harmonic, std)
        if std == 0.0:  # use distance of values in fft
            freq_binsize = 1.0 / self.wire_winlengths[wire_no]
            tension_binsize, _ = self.calculate_wire_tension(freq + freq_binsize)
            errorbar = tension_binsize - tension
        return (tension, errorbar)

    def start_analysis(self):
        """Main analysis loop"""
        self.wire_tensions.clear()
        self.tensions_binsizes.clear()
        print("Analysing wire tensions. This may take some time...")
        for i in tqdm(range(self.num_wires)):
            tension, binszize = self.analyse_tension(i)
            self.wire_tensions.append(tension)
            self.tensions_binsizes.append(binszize)

        print("")
        self.tensions_stats = describe(self.wire_tensions)
        print("Wire Tension Analysis finished:")
        print(
            f" mean = {self.tensions_stats.mean:.4f} N, std = {np.sqrt(self.tensions_stats.variance):.5f} N"
        )

    def _gauss_func(self, x, a, mu, sigma):
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


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
    spect = data.get_spectrum(wire_no)
    # spect = filter_spectrum(spect)
    x, y = data.do_fft(spect)
    freq, std, harmonic = data.find_frequency(x, y)
    tension = data.calculate_wire_tension(freq, harmonic)
    print(f" -| analyse_single_wire(wire_no={wire_no})")
    print(f"  | tension = {tension * 100:.2f} cN")
    DEBUG = False


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

    directory = "data/2026_04_29-BP1-006/Anode/"
    # directory = (
    #    "/Volumes/ikfhep/CBM/Drahtspannungsmessung/Messdaten/2026_06_17-BP1-006/"
    # )
    file = "WTD-Vibration-20260429-140250.tdms"
    my_data = WtmData(directory + file, wiretype="Anode")
    # my_data.start_analysis()
    # my_data.read_archive_tensions(directory + "26061513.13M", 0)
    # plot_pitches_histogram(my_data)
    # plot_wire_positions(my_data)
    # plot_wire_tensions(my_data)
    # my_data.to_csv_file(directory+"20260617-180850.csv")
    # my_data.to_dataframe()
    # analyse_single_wire(my_data, 52)

    # analyse_signal(my_data, 52)
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
