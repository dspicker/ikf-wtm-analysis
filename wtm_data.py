import sys
import os.path
import copy
import numpy as np
import pandas as pd
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from scipy.stats import describe
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from windingmachine.archive import get_tension
from tqdm import tqdm
import re
import datetime as dt
import json


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
        self.debug = False

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
                print(f"- Wires {i} and {i + 1} possible double Measurement! -")
                print(
                    f"  at {self.wire_positions[i] * 1000.0:.4f} mm and {self.wire_positions[i + 1] * 1000.0:.4f}, pitch = {pitch:.4f} mm"
                )
            if pitch > (2.5 + self.pitch_threshold) :
                print(f"+ Possible missed wire at index {i+1}, pitch = {pitch:.4f} mm +")
            self.wire_pitches.append(pitch)

        self.wp_statistics = describe(self.wire_pitches)
        print(
            f"Wire pitch: mean={self.wp_statistics.mean:.3f} mm, variance={self.wp_statistics.variance:.3f} mm"
        )

    def analysis_completed(self):
        if self.tensions_stats and self.wire_tensions and self.wire_winlengths:
            return True
        else:
            print("No wire tensions calculated (yet) in the given instance of WtmData.")
            return False

    """ ----------- Output Functions ----------- """

    def to_dataframe(self, filter: bool = False):
        """Return analysis result as pandas dataframe

        Args:
            filter (bool, optional): If double measurements should be removed from the result. Defaults to False.

        Returns:
            pandas.DataFrame: Tension analysis results
        """
        if not self.analysis_completed():
            return pd.DataFrame()
        pitches = copy.deepcopy(self.wire_pitches) 
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
        df = self.to_dataframe(False)
        df.to_csv(filename, index_label="index")

    def export_metadata_json(self, filename: str):
        if not (self.analysis_completed() or self.wp_statistics or self.tensions_stats):
            return
        if self.wp_statistics:
            pitches_mean = self.wp_statistics.mean
        else:
            pitches_mean = 0.0
        if self.tensions_stats:
            tensions_mean = self.tensions_stats.mean
            tensions_std = np.sqrt(self.tensions_stats.variance)
        else:
            tensions_mean = 0.0
            tensions_std = 0.0
        export_dict = {
            "wire_type": self.wire_type,
            "wire_length": self.wire_length,
            "wire_number": self.num_wires,
            "tdms_path": self.file_path,
            "datetime_measurement": dt.datetime.strftime(
                self.datetime_measured, "%Y-%m-%d %H:%M:%S"
            ),
            "datetime_analysis": dt.datetime.strftime(
                dt.datetime.now(), "%Y-%m-%d %H:%M:%S"
            ),
            "wire_measureduration": self.wire_winlengths[0],
            "pitches_mean": pitches_mean,
            "tensions_mean": tensions_mean,
            "tensions_std": tensions_std,
        }
        with open(filename, "w") as json_file:
            json.dump(export_dict, json_file, indent=2)

    """ ----------- Input Functions ----------- """

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
        if self.debug:
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
        if self.debug:
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

                if self.debug and ax:
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
            tqdm.write(f"Could not find frequency for wire No {wire_no}")
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
        print("Analysing wire tensions. This may take some time... \n")
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


""" -----------     Misc      ----------- """


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

    # comupte running mean over N samples:
    # N = 30
    # rm_spect = np.convolve(spectrum, np.ones(N) / N, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = [float(x) * sample_spacing for x in range(0, len(spectrum))]
    # x_rm = [float(x) * sample_spacing for x in range(0, len(rm_spect))]
    # y_oszi = [oszillator(x, 92.0) for x in x_vals]
    ax.plot(x_vals, spectrum, "-", linewidth=0.7)
    # ax.plot(x_rm, rm_spect, "-", linewidth=0.7)
    # ax.plot(x_vals,y_oszi, "-", linewidth=0.5, alpha=0.9)
    ax.grid(True)
    ax.set_title("Sensor Signal")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Signal /V")
    fig.tight_layout()


def plot_power_spectrum(data: WtmData, wire_no: int):
    spec = data.get_power_spectrum(wire_no)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(0, len(spec) * 2, 2), spec, ".-", linewidth=0.6)

    ax.set_xlim(0.0, 400.0)
    # plt.show()


def oszillator(t: float, freq: float):
    y_offset = 1.55

    r = 5.0
    amplitude = 0.6
    omega = 2 * np.pi * freq
    phi = 300.0 * np.pi / 180.0
    sinus = amplitude * np.cos(omega * t + phi) * np.exp(-t * r)

    return y_offset + sinus
