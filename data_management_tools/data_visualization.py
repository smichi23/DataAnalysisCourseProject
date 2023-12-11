import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams.update({'font.size': 18})
def plot_histogram(data, xlabel, ylabel, normalized=False):
    plt.figure(dpi=100, figsize=(10, 6))
    if normalized:
        plt.plot(data[:, 0], data[:, 1] / np.sum(data[:, 1] * (data[1, 0] - data[0, 0])))
    else:
        plt.plot(data[:, 0], data[:, 1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_multiple_histogram(datas, data_labels, xlabel, ylabel, bins, normalized=False, add_vline=None):
    plt.figure(dpi=100, figsize=(10, 6))
    it = 0
    for data in datas:
        plt.hist(data, bins=bins, density=normalized, label=data_labels[it], alpha=0.5, histtype='stepfilled',
                 ec='black')
        it += 1
    if add_vline is not None:
        plt.axvline(x=add_vline[0], color='black', linestyle='--', label=add_vline[1])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def show_image(image: np.ndarray, axis_titles, axis_ranges, vmin, vmax, index_to_show, axis_to_show):
    plt.figure(dpi=100)
    plt.imshow(np.take(image, indices=index_to_show, axis=axis_to_show),
               vmax=vmax, vmin=vmin, extent=axis_ranges)
    plt.xlabel(f"{axis_titles[0]}")
    plt.ylabel(f"{axis_titles[1]}")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(axis_titles[2], rotation=0, labelpad=30)
    plt.show()


def show_images_via_slider(images: np.ndarray, axis_titles, axis_ranges, vmin, vmax, axis_to_scroll):
    plt.figure(dpi=100)
    plot_image = plt.imshow(np.take(images, indices=images.shape[axis_to_scroll] // 2, axis=axis_to_scroll),
                            vmax=vmax, vmin=vmin, extent=axis_ranges)
    plt.xlabel(f"{axis_titles[0]}")
    plt.ylabel(f"{axis_titles[1]}")
    axslice = plt.axes([0.15, 0.9, 0.65, 0.02], facecolor='lightgoldenrodyellow')
    slice_index = Slider(axslice, 'Slice', valmin=-images.shape[axis_to_scroll] // 2,
                         valmax=images.shape[axis_to_scroll] // 2,
                         valinit=images.shape[axis_to_scroll] // 2, valstep=1)

    def update(val):
        slice_value = int(slice_index.val)
        plot_image.set_data(
            np.take(images, indices=images.shape[axis_to_scroll] // 2 + slice_value, axis=axis_to_scroll))
        plt.draw()

    slice_index.on_changed(update)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(axis_titles[2], rotation=0, labelpad=30)
    plt.show()
