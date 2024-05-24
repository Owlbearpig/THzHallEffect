from consts import DATA_DIR
from functools import partial
from numpy.fft import rfft, rfftfreq
import numpy as np
import os

en_avg_data = True

data_dir = DATA_DIR / "sample1_01_05_2024"

ref_path = data_dir / "free space measurement"  # reference
sam_path = data_dir / "With_sample_bigger_hole"  # sample on aperture (B=0)
ref_magnet_path = data_dir / "with_magnet_700"  # magnet only (B=0.7 T)
sam_magnet_p_path = data_dir / "with_magnet_sample_+"  # sample in magnet (B=0.7 T)
sam_magnet_n_path = data_dir / "with_magnet_sample_-"  # sample in magnet, magnet reversed (B=-0.7 T)


def sort_files(dir_, kw1, kw2=None):
    if kw2 is None:
        res = [dir_ / file for file in os.listdir(dir_) if kw1 in file]
    else:
        res1 = [dir_ / file for file in os.listdir(dir_) if kw1 in file]
        res2 = [dir_ / file for file in os.listdir(dir_) if kw2 in file]
        res = (res1, res2)

    return res


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

# no magnet (without and with sample)
ref_files1, ref_files2 = sort_files(ref_path, "lockin1", "lockin2")
sam_files1, sam_files2 = sort_files(sam_path, "lockin1", "lockin2")

ref1_td, ref1_fd = load_data(ref_files1)
ref2_td, ref2_fd = load_data(ref_files2)

sam1_td, sam1_fd = load_data(sam_files1)
sam2_td, sam2_fd = load_data(sam_files2)

# magnet without sample
ref_magnet_files1, ref_magnet_files2 = sort_files(ref_magnet_path, "--lockin1", "--lockin2")

ref1_magnet_td, ref1_magnet_fd = load_data(ref_magnet_files1)
ref2_magnet_td, ref2_magnet_fd = load_data(ref_magnet_files2)

# magnet(+, -) with sample
sam_magnet_p_files1, sam_magnet_p_files2 = sort_files(sam_magnet_p_path, "data-lockin1", "data-lockin2")
sam_magnet_n_files1, sam_magnet_n_files2 = sort_files(sam_magnet_n_path, "--lockin1", "--lockin2")

# parallel (+)
sam1_td_p, sam1_fd_p = load_data(sam_magnet_p_files1)
sam2_td_p, sam2_fd_p = load_data(sam_magnet_p_files2)

# anti parallel (-)
sam1_td_n, sam1_fd_n = load_data(sam_magnet_n_files1)
sam2_td_n, sam2_fd_n = load_data(sam_magnet_n_files2)



