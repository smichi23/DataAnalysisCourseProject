from typing import Tuple, Dict

import numpy as np


def read_cancer_data(path_to_data: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the cancer data from the path specified by path_to_data.
    :param path_to_data: Path to the data
    :return: The data as a numpy array
    """
    open_file = open(path_to_data, 'r')
    lines = open_file.readlines()
    labels = []
    data = []
    for line in lines:
        if "_patient_" in line:
            data.append([])
            words = line[:-2].split('_')
            if words[0] == 'healthy':
                labels.append(0)
            else:
                labels.append(1)
        else:
            if labels != [] and ',' in line:
                data[-1].append(line[:-2].split(','))

    return np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def read_secret_cancer_data(path_to_data: str) -> np.ndarray:
    """
    Reads the cancer data from the path specified by path_to_data.
    :param path_to_data: Path to the data
    :return: The data as a numpy array
    """
    open_file = open(path_to_data, 'r')
    lines = open_file.readlines()
    data = []
    for line in lines:
        if "Patient_" in line:
            data.append([])
        else:
            if data != [] and ',' in line:
                data[-1].append(line[:-2].split(','))

    return np.asarray(data, dtype=np.float32)


def read_time_dependant_cancer_data(path_to_data: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the cancer data from the path specified by path_to_data.
    :param path_to_data: Path to the data
    :return: The data as a numpy array
    """
    open_file = open(path_to_data, 'r')
    lines = open_file.readlines()
    labels = []
    data = []
    current_patient = -1
    for line in lines:
        if "_patient_" in line:
            words = line[:-2].split('_')
            if int(words[2]) != current_patient:
                current_patient = int(words[2])
                data.append([])
                if words[0] == 'healthy':
                    labels.append(0)
                else:
                    labels.append(1)

            data[-1].append([])

        else:
            if labels != [] and ',' in line:
                data[-1][-1].append(line[:-2].split(','))

    return np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def read_historical_cancer_data(path_to_data: str, bin_width: float) -> Dict[str, np.ndarray]:
    """
    Reads the cancer data from the path specified by path_to_data.
    :param path_to_data: Path to the data
    :return: The data as a numpy array
    """
    open_file = open(path_to_data, 'r')
    lines = open_file.readlines()
    data = {}
    current_data_type = ''
    bin_iterator = -1
    for line in lines:
        if "_data" in line:
            words = line.split('_')
            current_data_type = words[2]
            if words[2] not in data.keys():
                data[words[2]] = []

            data[current_data_type].append([])
            bin_iterator = 0
        else:
            if current_data_type != "" and len(line) > 1:
                data[current_data_type][-1].append([bin_width * (bin_iterator + 0.5), float(line)])
                bin_iterator += 1

    array_dict = {}
    for key in data.keys():
        array_dict[key] = np.asarray(data[key], dtype=np.float32)

    return array_dict
