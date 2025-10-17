import os.path
from nptdms import TdmsFile
import matplotlib.pyplot as plt


class WtmProfile:
    def __init__(self, tdms_file_path: str) -> None:
        self.file_path = ""
        self._read_tdms(tdms_file_path)

    def _read_tdms(self, tdms_file_path: str):
        tdms_file_path = os.path.abspath(tdms_file_path)
        if os.path.isfile(tdms_file_path):
            print(f"Loading tdms file  {tdms_file_path} ")
            self.file_path = tdms_file_path
        with TdmsFile.open(tdms_file_path) as tdms_file:
            x_profile = tdms_file["X-Profile"]
            self.x_profile_led = x_profile["LED"][:]
            self.x_profile_locations = x_profile["Locations"][:]
            self.x_profile_amplitudes = x_profile["Amplitudes"][:]
            self.x_profile_gaps = x_profile["Gaps"][:]


if __name__ == "__main__":
    data = WtmProfile("daten_bp1-007/WTD-Profile-20251013-145459.tdms")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.x_profile_led)
    # ax.plot(data.x_profile_locations)
    plt.show()
