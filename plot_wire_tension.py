from nptdms import TdmsFile
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, find_peaks
from scipy.fft import fft, ifft, fftfreq, fftshift, rfft
import numpy as np

# from typing import String

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

class WireProfileData:
    def __init__(self, tdms_file: str):
        self.led, self.loc, self.ampl, self.gaps = self._getData(tdms_file)

    def _getData(self, file: str):
        with TdmsFile.open(file) as tdms_file:
            # print(tdms_file.groups())
            x_profile = tdms_file["X-Profile"]
            return x_profile["LED"][:], x_profile["Locations"][:], x_profile["Amplitudes"][:], x_profile["Gaps"][:],

class WireVibrationData:
    def __init__(self, tdms_file: str):
        # self.led, self.loc, self.ampl, self.gaps = self._getData(tdms_file)
        self.wires = self._getData(tdms_file)

    def _getData(self, file: str):
        with TdmsFile.open(file) as tdms_file:
            ret_dict = {}
            wires = tdms_file.groups()
            print(next(wires[0]["AI_Subset"]._reader.read_raw_data()))
            for name, value in wires[0]["AI_Subset"].properties.items():
                print("{0}: {1}".format(name, value))
            for i in range(len(wires)):
                ret_dict[i] = {
                        "ai_subset": wires[i]["AI_Subset"][:],
                        "power_spectrum": wires[i]["Power_Spectrum"][:],
                        "harmonics": wires[i]["Harmonics"][:],
                        "amplitudes": wires[i]["Amplitudes"][:],
                        "position_m": wires[i].properties["Position_m"],
                        }
            return ret_dict


wire_length = 1.4
wire_density = 19300
wire_radius = 10*10**-6


if __name__ == "__main__":


    p = 19300 # Dichte AuW
    r = 10 * 10**-6
    l = 1.4 

    fig, ax = plt.subplots()

    '''
            Kammer 1, AuW auf Rahmen
    '''

    t = WireVibrationData('WTD-Vibration-20230629-214501.tdms')
    print("wires: {}".format(len(t.wires)))

    wires = len(t.wires)

    axis_x = []
    axis_y = []

    cf = 103

    for i in range(wires):
        print("Wire {}/{}".format(i,wires))
        ps = t.wires[i]["ai_subset"]
        raw = [x for x in ps if str(x) != 'nan']
        N = len(raw)
        T = 1.0/100000
        #print("  Samples: {}".format(N))
        # FFT
        yfs = abs(fft(raw))
        xfs = fftfreq(N, T)#[:N//2]
        xmax = 0
        ymax = 0
        for j in range(N//2):
            if xfs[j] > 52 and xfs[j]<cf+0.5*cf:
                if yfs[j] > ymax:
                    xmax = xfs[j]
                    ymax = yfs[j]
        print("  Frequency: {}".format(xmax))
        F = 4*(xmax**2)*(l**2)*p*np.pi*(r**2)
        axis_x.append(t.wires[i]["position_m"]);
        axis_y.append(F)



    ax.plot(axis_x, axis_y)

    '''
            Kammer 1, AuW auf Leisten
    '''

    t = WireVibrationData('WTD-Vibration-20230705-120352.tdms')
    print("wires: {}".format(len(t.wires)))

    wires = len(t.wires)

    axis_x = []
    axis_y = []

    l = 0.96 
    cf = 150


    for i in range(wires):
        print("Wire {}/{}".format(i,wires))
        ps = t.wires[i]["ai_subset"]
        raw = [x for x in ps if str(x) != 'nan']
        N = len(raw)
        T = 1.0/100000
        #print("  Samples: {}".format(N))
        # FFT
        yfs = abs(fft(raw))
        xfs = fftfreq(N, T)#[:N//2]
        xmax = 0
        ymax = 0
        for j in range(N//2):
            if xfs[j] > cf-0.5*cf and xfs[j]<cf+0.5*cf:
                if yfs[j] > ymax:
                    xmax = xfs[j]
                    ymax = yfs[j]
        #print("  Frequency: {}".format(xmax))
        F = 4*(xmax**2)*(l**2)*p*np.pi*(r**2)
        axis_x.append(t.wires[i]["position_m"]+0.136);
        axis_y.append(F)



    ax.plot(axis_x, axis_y)


    '''
            ENDE
    '''
    
    plt.xlabel("Wire")
    plt.ylabel("Tension [N]")
    plt.title("Wire tension on frame vs on chamber")
    plt.savefig("./plot_wire_tension.png",dpi=600)
    plt.show()

