import numpy as np
from data_management_tools.data_readers import read_cancer_data, read_time_dependant_cancer_data, read_historical_cancer_data
from data_management_tools.data_visualization import plot_histogram, show_image, show_images_via_slider


if __name__ == '__main__':
    data, labels = read_cancer_data('./data/cancer_data.txt')
    print(data.shape)
    print(labels.shape)
    patients_with_cancer = np.where(labels == 1)[0]
    show_images_via_slider(data[list(patients_with_cancer)],
                           ['x [cm]', 'y [cm]', 'Intensity [-]'],
                           [-7.5, 7.5, -7.5, 7.5],
                           0, data[list(patients_with_cancer)].max(), 0)
    data, labels = read_time_dependant_cancer_data('./data/cancer_data_timeDependency.txt')
    print(data.shape)
    print(labels.shape)
    patients_with_cancer = np.where(labels == 1)[0]
    print(data[list(patients_with_cancer)[0]].shape)
    show_images_via_slider(data[list(patients_with_cancer)[1]],
                           ['x [cm]', 'y [cm]', 'Intensity [-]'],
                           [-7.5, 7.5, -7.5, 7.5],
                           0, data[list(patients_with_cancer)].max(), 0)
    histo_dict = read_historical_cancer_data('./data/historical_cancer_data.txt', 0.2)
    for key in histo_dict.keys():
        for histo in histo_dict[key]:
            plot_histogram(histo, f'{key}', 'Intensity [-]')

