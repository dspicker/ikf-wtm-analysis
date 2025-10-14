import os.path

# import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt


class WtmData:
    def __init__(self, tdms_file_path: str) -> None:
        self.num_wires: int = 0
        self.wire_positions: list[float] = list()
        self._read_tdms(tdms_file_path)

    def _read_tdms(self, tdms_file_path: str):
        tdms_file_path = os.path.abspath(tdms_file_path)
        if os.path.isfile(tdms_file_path):
            print(f"Loading tdms file  {tdms_file_path} ")
        with TdmsFile.read_metadata(tdms_file_path) as tdms_file:
            all_groups = tdms_file.groups()  # One group per wire
            self.num_wires = len(all_groups)
            print(f"File contains data for {self.num_wires} wires.")
            for group in all_groups:
                self.wire_positions.append(group.properties["Position_m"])

    def get_wire_pitches(self):
        """Get wire pitches in mm

        Returns:
            list[float]: Element 0 is the pitch between wire 0 and 1,
                Element 1 is the pitch between wire 1 and 2 and so on...
        """
        pitches: list[float] = list()
        for i in range(self.num_wires - 1):
            pitch = (self.wire_positions[i + 1] - self.wire_positions[i]) * 1000.0
            pitches.append(pitch)
        return pitches


def plot_pitches_histogram(data: WtmData):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data.get_wire_pitches(), bins=24, range=(-0.125, 5.875))
    ax.grid(axis='y', which='both')
    #ax.set_ylabel("Number of wires")
    ax.set_xlabel("Wire pitch /mm")
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
    ax_2.plot(range(data.num_wires - 1), data.get_wire_pitches(), ".-", linewidth=0.6)

    ax_1.grid(True)
    ax_1.set_ylabel("Wire pos. /m")

    ax_2.grid(True)
    ax_2.set_ylabel("Wire pitch /mm")
    ax_2.set_xlabel("Wire number")
    plt.show()


if __name__ == "__main__":
    data = WtmData("daten2/WTD-Vibration-20230705-120352.tdms")
    plot_pitches_histogram(data)
    #plot_wire_positions(data)
