import pandas as pd
import matplotlib.pyplot as plt






def read_archive(file: str):
    archive_data = pd.read_csv(
        file,
        sep=";",
        header=None,
        index_col=False,
        usecols=[0, 1, 2, 3, 5, 6],
        names=["wire", "angle", "tension", "speed", "time", "day"],
        parse_dates=["time", "day"],
        date_format={"time": "%H:%M:%S", "day": "$m-%d-%Y"}
    )
    return archive_data


def get_tension(file: str, frame: int=0):

    archive_data = read_archive(file)

    frame1 = archive_data[ archive_data["angle"] == 155.1 ]
    frame0 = archive_data[ archive_data["angle"] == 335.1 ]

    if frame == 1:
        return frame1["tension"].tolist()
    else:
        return frame0["tension"].tolist()
    

if __name__ == "__main__":

    #archive_file = "/Users/dspicker/Uni/wickelmaschine/archiv/26041313.24M"
    archive_file = "/Users/dspicker/Uni/drahtspannung/datenauswertung/data/2026_05_26-Test/26052114.32M"

    archive_data = read_archive(archive_file)
    frame1 = archive_data[ archive_data["angle"] == 155.1 ]
    frame0 = archive_data[ archive_data["angle"] == 335.1 ]

    fig, ax = plt.subplots()
    #ax.plot(frame1["wire"], frame1["tension"], "-o", linewidth=0.6)
    ax.plot(frame0["wire"], frame0["speed"], "-o", linewidth=0.6)
    ax.set_xlabel("Wire No.")
    ax.set_ylabel("Tension /cN")
    ax.set_title("Wire Tensions from Meteor at 155.1°")
    ax.grid(True)
    fig.tight_layout()
    plt.show()