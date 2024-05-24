import matplotlib.pyplot as plt
import numpy as np
# import PySimpleGUI as sg
import os
from parse_data import *


def plot_data(data_td, data_fd, **kwargs):
    if "fig_num" not in kwargs:
        kwargs["fig_num"] = "Figure 1"
    if "label" not in kwargs:
        kwargs["label"] = ""
    if "title" not in kwargs:
        kwargs["title"] = ""
    if "fig_info" in kwargs:
        kwargs["title"] += "\n" + str(kwargs["fig_info"])

    t_axis, y_td = data_td[:, 0], data_td[:, 1]
    freqs, y_fd = data_fd[:, 0].real, data_fd[:, 1]
    y_fd_db = 10 * np.log10(np.abs(y_fd))

    fig_num = kwargs["fig_num"]
    if plt.fignum_exists(fig_num):
        fig = plt.figure(fig_num)
        ax0, ax1 = fig.get_axes()
    else:
        fig, (ax0, ax1) = plt.subplots(2, 1, num=fig_num)
        fig.suptitle(kwargs["title"])

        ax0.set_xlabel("Time (ps)")
        ax0.set_ylabel("Amplitude (arb. u.)")
        ax0.set_ylim((-1.65e-5, 1.1e-5))

        ax1.set_xlabel("Frequency (THz)")
        ax1.set_ylabel("Amplitude (dB)")
        ax1.set_xlim((-0.05, 4))
        ax1.set_ylim((-80, -35))

    ax0.plot(t_axis, y_td, label=kwargs["label"])
    ax0.legend()

    ax1.plot(freqs, y_fd_db, label=kwargs["label"])
    ax1.legend()


def plot_ref_and_sample():
    """
    ### Reference and sample (No magnet)
    """

    fig_kwargs = {"fig_num": "No magnet", "title": "Reference and sample (No magnet)",
                  "fig_info": f"Averaging_en: {en_avg_data}"}
    plot_data(ref1_td, ref1_fd, label="Reference lockin1", **fig_kwargs)
    plot_data(ref2_td, ref2_fd, label="Reference lockin2", **fig_kwargs)

    plot_data(sam1_td, sam1_fd, label="Sample lockin1", **fig_kwargs)
    plot_data(sam2_td, sam2_fd, label="Sample lockin2", **fig_kwargs)


def plot_magnet_with_and_without():
    """
    ### Magnet (B=0.7 T) without (ref) and with sample (sam)
    """


    fig_kwargs = {"fig_num": "With magnet", "title": "Reference(no sample) and sample (both with magnet)",
                  "fig_info": f"Averaging_en: {en_avg_data}"}
    #plot_data(ref1_td, ref1_fd, label="Reference lockin1", **fig_kwargs)
    #plot_data(ref2_td, ref2_fd, label="Reference lockin2", **fig_kwargs)

    plot_data(sam1_td_p, sam1_fd_p, label="Sample lockin1 (B=0.7 T)", **fig_kwargs)
    plot_data(sam2_td_p, sam2_fd_p, label="Sample lockin2 (B=0.7 T)", **fig_kwargs)

    plot_data(sam1_td_n, sam1_fd_n, label="Sample lockin1 (B=-0.7 T)", **fig_kwargs)
    plot_data(sam2_td_n, sam2_fd_n, label="Sample lockin2 (B=-0.7 T)", **fig_kwargs)


def main():
    plot_ref_and_sample()
    plot_magnet_with_and_without()
    plt.show()


if __name__ == '__main__':
    main()
