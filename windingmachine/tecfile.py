import os.path
import re
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.interpolate import make_interp_spline


class TecFile:
    def __init__(self) -> None:
        self.w0_values = [0.0] * 3601
        self.w0_values[0] = 1.0
        self.w1_values = [1.0] * 3601
        self.params = {
            "wire_speed": 0.075,  # m/s
            "angle_move": 20.0,  # deg
            "wire_pitch": 2.5,  # mm
            "winding_start": 300.0,  # mm
            "winding_end": 1300.0,  # mm
            "wire_tension": 470.0,  # mN
            "wire_diameter": 0.02,  # mm
            "angle_init": 10.0,  # deg
        }

    def compose_line(self, group_idx: int, param_idx: int, value: float):
        if (
            group_idx not in [0, 1]
            or param_idx not in range(0, 3701)
            or (value < 0.0 or value > 9999.99)
        ):
            raise ValueError("Value not in allowed range.")
        string = f"{value: 9.3f}  ;  W {group_idx} - {param_idx} "
        return string

    def _check_file(self, filepath: str):
        filepath = os.path.abspath(filepath)
        if os.path.isfile(filepath):
            print(f"Existing .TEC file: {filepath}")
            return filepath
        else:
            path, filename = os.path.split(filepath)
            root, ext = os.path.splitext(filename)
            filename = root[0:8].upper() + ".TEC"
            filepath = os.path.join(path, filename)
            with open(filepath, mode="xb"):
                print(f"New .TEC file: {filepath}")
            return filepath

    def write_file(self, filepath: str):
        filepath = self._check_file(filepath)
        with open(filepath, mode="w", encoding="utf-8", newline="\r\n") as file:
            for idx, val in enumerate(self.w0_values):
                out_str = self.compose_line(0, idx, val)
                file.write(out_str + "\n")
            for i in range(3601, 3701):
                file.write(self.compose_line(0, i, 0.0) + "\n")
            for idx, val in enumerate(self.w1_values):
                out_str = self.compose_line(1, idx, val)
                file.write(out_str + "\n")
            for i in range(3601, 3670):
                file.write(self.compose_line(1, i, 0.0) + "\n")
            out_str = self.compose_line(1, 3670, self.params["wire_speed"])
            file.write(out_str + "\n")
            out_str = self.compose_line(1, 3671, self.params["angle_move"])
            file.write(out_str + "\n")
            out_str = self.compose_line(1, 3672, 0.0)
            file.write(out_str + "\n")
            out_str = self.compose_line(1, 3673, self.params["wire_pitch"])
            file.write(out_str + "\n")
            out_str = self.compose_line(1, 3674, self.params["winding_start"])
            file.write(out_str + "\n")
            out_str = self.compose_line(1, 3675, self.params["winding_end"])
            file.write(out_str + "\n")
            out_str = self.compose_line(1, 3676, self.params["wire_tension"])
            file.write(out_str + "\n")
            out_str = self.compose_line(1, 3677, self.params["wire_diameter"])
            file.write(out_str + "\n")
            out_str = self.compose_line(1, 3678, self.params["angle_init"])
            file.write(out_str + "\n")
            for i in range(3679, 3701):
                file.write(self.compose_line(1, i, 0.0) + "\n")
            file.write("171255  ; Dateiidentifikation")

    def read_file(self, filepath: str):
        new_w0_values: list[float] = list()
        new_w1_values: list[float] = list()
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                columns = line.split(";", 1)
                value = float(columns[0].strip())
                name = columns[1].strip()
                # print(f"{name} ; {value:07.3f}")
                regex_match = re.match(r"W (\d) - (\d{1,4})", name)
                if regex_match:
                    w = int(regex_match.group(1))
                    idx = int(regex_match.group(2))
                    if w == 0:
                        assert len(new_w0_values) == idx
                        new_w0_values.append(value)
                    elif w == 1:
                        assert len(new_w1_values) == idx
                        new_w1_values.append(value)
                    else:
                        print(regex_match)
        self.w0_values = new_w0_values[0:3601]
        self.w1_values = new_w1_values[0:3601]
        self.params["wire_speed"] = new_w1_values[3670]
        self.params["angle_move"] = new_w1_values[3671]
        self.params["wire_pitch"] = new_w1_values[3673]
        self.params["winding_start"] = new_w1_values[3674]
        self.params["winding_end"] = new_w1_values[3675]
        self.params["wire_tension"] = new_w1_values[3676]
        self.params["wire_diameter"] = new_w1_values[3677]
        self.params["angle_init"] = new_w1_values[3678]

    def set_rpm_profile(self, profile):
        new_profile = list(profile)
        assert len(new_profile) == 3601
        self.w1_values = new_profile

    def save_params(self, filename: str = "params.json"):
        with open(filename, mode="w", encoding="utf-8") as file:
            json.dump(self.params, file, indent=2)

    def load_params(self, filename: str = "params.json"):
        with open(filename, mode="r", encoding="utf-8") as file:
            new_params = json.load(file)
            self.params = new_params

    def get_basepoints(self):
        basepoints_angle = [
            float(i) / 10.0 for i, val in enumerate(self.w0_values[0:3601]) if val > 0.0
        ]
        basepoints_rpm = [val for val in self.w0_values[0:3601] if val > 0.0]
        ret_dict = dict(zip(basepoints_angle, basepoints_rpm))
        return ret_dict

    def reset_basepoints(self):
        self.w0_values = [0.0] * 3601
        self.w0_values[0] = 1.0

    def set_basepoints(self, basepoints: dict[float, float]):
        for angle, rpm in basepoints.items():
            angle = int(angle * 10.0)
            if (angle not in range(0, 3601)) or not (0.0 < rpm < 6.0):
                raise ValueError("Value out of range")
            # print(f"angle= {angle}, rpm= {rpm}")
            self.w0_values[angle] = rpm

    def save_basepoints(self, filename: str = "basepoints.json"):
        with open(filename, mode="w", encoding="utf-8") as file:
            json.dump(self.get_basepoints(), file, indent=2)

    def load_basepoints(self, filename: str = "basepoints.json"):
        with open(filename, mode="r", encoding="utf-8") as file:
            new_basepoints = json.load(file)
            # in json keys are str, so we convert to float:
            new_basepoints = {float(i): j for i, j in new_basepoints.items()}
            self.set_basepoints(new_basepoints)

    def interpolate_basepoints(self):
        x_spline = np.arange(0.0, 360.1, 0.1)
        bx, by = zip(*self.get_basepoints().items())
        spline = make_interp_spline(bx, by, k=3, bc_type="periodic")
        y_spline = spline(x_spline)
        return x_spline, y_spline

    def make_smooth(self, profile, window: int = 90):
        N = window
        # da wir ein periodisches signal haben, können wir so randeffekte beim glätten entfernen
        profile_expanded = profile[-N:] + profile + profile[:N]
        # glättung
        profile_smooth = np.convolve(profile_expanded, np.ones(N) / N, mode="same")
        # zurück zur richtigen länge
        profile_return = profile_smooth[N:-N]
        return list(profile_return)

    def create_plot(self):
        rpm_x = [float(i) for i in np.arange(0.0, 360.1, 0.1)]
        rpm_y = self.w1_values[0:3601]

        bx, by = zip(*self.get_basepoints().items())
        sx, sy = self.interpolate_basepoints()

        fig, ax = plt.subplots()
        ax.plot(bx, by, "o", label="Stützpunkte")
        ax.plot(rpm_x, rpm_y, ".", markersize=2.0, label="Geschwindigkeitsprofil")
        # ax.plot(rpm_profile_smooth, ".", markersize=0.9,)
        ax.plot(sx, sy, ".", markersize=1.0, label="Interpoliert")
        ax.set_xlabel("degree")
        ax.xaxis.set_major_locator(MultipleLocator(45))
        ax.set_ylim(0.0)
        ax.set_ylabel("rpm")
        ax.set_title("Sollwertkurve Vorgabegeschwindigkeit")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    myfile = TecFile()

    # myfile.write_file("test.tec")
    # myfile.read_file("test.tec")

    # myfile.save_params()
    # myfile.save_basepoints()

    myfile.load_params()
    myfile.load_basepoints()
    sx, sy = myfile.interpolate_basepoints()
    myfile.set_rpm_profile(sy)
    # myfile.set_rpm_profile(myfile.make_smooth(myfile.w1_values, 200))
    # myfile.write_file("AU-NEU2.TEC")

    myfile.create_plot()
