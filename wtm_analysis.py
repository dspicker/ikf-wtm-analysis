import os.path
from matplotlib import pyplot as plt
from wtm_data import WtmData
from wtm_plot import plot_wire_positions, plot_pitches_hist, plot_wire_tensions


def analyse_single_tdms(
    filepath_tdms: str,
    wiretype: str,
    wirelength: float,
    set_tension: float,
    backpanel_id: str,
):
    filepath_tdms = os.path.abspath(filepath_tdms)
    path_dir, path_file = os.path.split(filepath_tdms)
    file_name, file_ext = os.path.splitext(path_file)
    if file_ext != ".tdms":
        print("Need path to tdms file!")

    my_data = WtmData(filepath_tdms, wiretype, wirelength)
    my_data.start_analysis()

    ## Uncomment, if you want to read archive data from the winding machine:
    # my_data.read_archive_tensions(path_dir + "26061513.13M", 0)

    ## Export results
    my_data.to_csv_file(os.path.join(path_dir, file_name + "_result.csv"))
    my_data.export_metadata_json(os.path.join(path_dir, file_name + "_info.json"))

    ## Plotting
    my_dataframe = my_data.to_dataframe(False)
    plot_wire_tensions(
        my_dataframe,
        set_tension,
        wiretype + ", " + backpanel_id,
        os.path.join(path_dir, file_name + "_tensions.png"),
    )
    plot_wire_positions(
        my_dataframe,
        wiretype + ", " + backpanel_id,
        os.path.join(path_dir, file_name + "_positions.png"),
    )
    plot_pitches_hist(
        my_dataframe,
        2.5,
        wiretype + ", " + backpanel_id,
        os.path.join(path_dir, file_name + "_pitches.png"),
    )


""" ----------- Main Function ----------- """

if __name__ == "__main__":
    analyse_single_tdms(
        "/Volumes/ikfhep/CBM/Drahtspannungsmessung/Messdaten/2026_09_15-BP1-006/WTD-Vibration-20260916-150246.tdms",
        "Anode",
        0.96,
        0.5,
        "BP1-006",
    )

    ## Show Plots:
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close("all")
