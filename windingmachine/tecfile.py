import os.path
import re
import json


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
        self.mode = ""  # "existing" or "new"

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
            self.mode = "existing"
            return filepath
        else:
            path, filename = os.path.split(filepath)
            root, ext = os.path.splitext(filename)
            filename = root[0:8].upper() + ".TEC"
            filepath = os.path.join(path, filename)
            with open(filepath, mode="xb"):
                print(f"New .TEC file: {filepath}")
                self.mode = "new"
            return filepath

    def write_file(self, filepath: str):
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
        new_w0_values = list()
        new_w1_values = list()
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

    def save_params(self, filename: str):
        with open(filename, mode="w", encoding="utf-8") as file:
            json.dump(self.params, file, indent=2)

    def load_params(self, filename: str):
        with open(filename, mode="r", encoding="utf-8") as file:
            new_params = json.load(file)
            self.params = new_params


if __name__ == "__main__":
    myfile = TecFile()
    # test = myfile.compose_line(0, 3400, 3.5)
    # print(f" >{test}< ")
    # myfile.write_file("test.tec")
    # myfile.read_file("test.tec")

    # myfile._check_file("ab3456789.cdef")
    #myfile.save_params("test.json")
    #myfile.load_params("test.json")
