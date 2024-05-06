from consts import DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
# import PySimpleGUI as sg
import os
from functools import partial
from numpy.fft import rfft, rfftfreq

data_dir = DATA_DIR / "sample1_01_05_2024"

ref_path = data_dir / "free space measurement"  # reference
sam_path = data_dir / "With_sample_bigger_hole"  # sample on aperture (B=0)
ref_magnet_path = data_dir / "with_magnet_700"  # magnet only (B=0.7 T)
sam_magnet_p_path = data_dir / "with_magnet_sample_+"  # sample in magnet (B=0.7 T)
sam_magnet_n_path = data_dir / "with_magnet_sample_-"  # sample in magnet, magnet reversed (B=-0.7 T)

en_avg_data = True


def differentiate_data(data_arr):
    t_axis = data_arr[:, 0]
    y_td = np.gradient(data_arr[:, 1], t_axis)

    return np.array([t_axis, y_td], dtype=float).T


def fft(data_td):
    dt = np.mean(np.diff(data_td[:, 0]))
    y_fd = rfft(data_td[:, 1])
    freqs = rfftfreq(len(data_td[:, 0]), dt)

    return np.array([freqs, y_fd], dtype=complex).T


def load_data(file_paths, avg_data=True):
    t_axis = np.loadtxt(file_paths[0])[:, 0]

    amp_arrays = []
    for i, file_path in enumerate(file_paths):
        data_int_td = np.loadtxt(file_path)
        data_td = differentiate_data(data_int_td)

        amp_arrays.append(data_td[:, 1])

        if not avg_data:
            break

    amp_avg = np.mean(amp_arrays, axis=0)

    data_td = np.array([t_axis, amp_avg], dtype=float).T

    # data_td[:, 1] -= np.mean(data_td[:10, 1], axis=0)

    data_fd = fft(data_td)

    return data_td, data_fd


load_data = partial(load_data, avg_data=en_avg_data)


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
    ref_files1 = [ref_path / file for file in os.listdir(ref_path) if "lockin1" in file]
    ref_files2 = [ref_path / file for file in os.listdir(ref_path) if "lockin2" in file]

    sam_files1 = [sam_path / file for file in os.listdir(sam_path) if "lockin1" in file]
    sam_files2 = [sam_path / file for file in os.listdir(sam_path) if "lockin2" in file]

    ref1_td, ref1_fd = load_data(ref_files1)
    ref2_td, ref2_fd = load_data(ref_files2)

    sam1_td, sam1_fd = load_data(sam_files1)
    sam2_td, sam2_fd = load_data(sam_files2)

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
    ref_magnet_files1 = [ref_magnet_path / file for file in os.listdir(ref_magnet_path) if "--lockin1" in file]
    ref_magnet_files2 = [ref_magnet_path / file for file in os.listdir(ref_magnet_path) if "--lockin2" in file]

    sam_magnet_p_files1 = [sam_magnet_p_path / file for file in os.listdir(sam_magnet_p_path) if "data-lockin1" in file]
    sam_magnet_p_files2 = [sam_magnet_p_path / file for file in os.listdir(sam_magnet_p_path) if "data-lockin2" in file]

    sam_magnet_n_files1 = [sam_magnet_n_path / file for file in os.listdir(sam_magnet_n_path) if "--lockin1" in file]
    sam_magnet_n_files2 = [sam_magnet_n_path / file for file in os.listdir(sam_magnet_n_path) if "--lockin2" in file]

    ref1_td, ref1_fd = load_data(ref_magnet_files1)
    ref2_td, ref2_fd = load_data(ref_magnet_files2)

    sam1_td_p, sam1_fd_p = load_data(sam_magnet_p_files1)
    sam2_td_p, sam2_fd_p = load_data(sam_magnet_p_files2)

    sam1_td_n, sam1_fd_n = load_data(sam_magnet_n_files1)
    sam2_td_n, sam2_fd_n = load_data(sam_magnet_n_files2)

    fig_kwargs = {"fig_num": "With magnet", "title": "Reference(no sample) and sample (both with magnet)",
                  "fig_info": f"Averaging_en: {en_avg_data}"}
    plot_data(ref1_td, ref1_fd, label="Reference lockin1", **fig_kwargs)
    plot_data(ref2_td, ref2_fd, label="Reference lockin2", **fig_kwargs)

    plot_data(sam1_td_p, sam1_fd_p, label="Sample lockin1 (B=0.7 T)", **fig_kwargs)
    plot_data(sam2_td_p, sam2_fd_p, label="Sample lockin2 (B=0.7 T)", **fig_kwargs)

    plot_data(sam1_td_p, sam1_fd_n, label="Sample lockin1 (B=-0.7 T)", **fig_kwargs)
    plot_data(sam2_td_p, sam2_fd_n, label="Sample lockin2 (B=-0.7 T)", **fig_kwargs)


def main():
    plot_ref_and_sample()
    plot_magnet_with_and_without()
    plt.show()


if __name__ == '__main__':
    main()
