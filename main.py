import numpy as np
from torch.utils.data import DataLoader

from data_management_tools.data_readers import read_cancer_data, read_time_dependant_cancer_data, \
    read_historical_cancer_data
from data_management_tools.data_visualization import plot_histogram, show_image, show_images_via_slider
from diagnostic_models.max_diff_density_models import PreDefinedThresholdMaximumDiffDensity, \
    SigmoidScipyCurveFitMaximumDiffDensity, OptimalThresholdMaximumDiffDensity
from diagnostic_models.maximum_density_models import PreDefinedThresholdMaximumDensity, \
    SigmoidScipyCurveFitMaximumDensity
from model_evaluation.auc_curve_evaluation import AUCComparisonOfModelPerformances
from model_evaluation.test_performances import TestPerformances
from model_evaluation.p_value_comparison_of_model_perfoprmances import PValueComparisonOfModelPerformancesDistribution
import torch
from diagnostic_models.deep_learning_models import *

if __name__ == '__main__':
    # Data visualization

    # data, labels = read_cancer_data('./data/cancer_data.txt')
    # print(data.shape)
    # print(labels.shape)
    # patients_with_cancer = np.where(labels == 1)[0]
    # show_images_via_slider(data[list(patients_with_cancer)],
    #                        ['x [cm]', 'y [cm]', 'Intensity [-]'],
    #                        [-7.5, 7.5, -7.5, 7.5],
    #                        0, data[list(patients_with_cancer)].max(), 0)
    # data2, labels2 = read_time_dependant_cancer_data('./data/cancer_data_timeDependency.txt')
    # print(data2.shape)
    # print(labels2.shape)
    # patients_with_cancer = np.where(labels2 == 1)[0]
    # print(data2[list(patients_with_cancer)[0]].shape)
    # show_images_via_slider(data2[list(patients_with_cancer)[1]],
    #                        ['x [cm]', 'y [cm]', 'Intensity [-]'],
    #                        [-7.5, 7.5, -7.5, 7.5],
    #                        0, data2[list(patients_with_cancer)].max(), 0)
    # histo_dict = read_historical_cancer_data('./data/historical_cancer_data.txt', 0.2)
    # for key in histo_dict.keys():
    #     for histo in histo_dict[key]:
    #         plot_histogram(histo, f'{key}', 'Intensity [-]')
    #
    # patients_without_cancer = np.where(labels == 0)[0]
    # max_densities = data[list(patients_without_cancer)].max(axis=(1, 2))
    # histo = np.histogram(max_densities, bins=100)
    # histo_as_array = np.concatenate((histo[1][1:].reshape(-1, 1), histo[0].reshape(-1, 1)), axis=1)
    # plot_histogram(histo_as_array, 'Max Density', 'Frequency', normalized=True)
    #
    # patients_with_cancer = np.where(labels == 1)[0]
    # max_densities = data[list(patients_with_cancer)].max(axis=(1, 2))
    # histo = np.histogram(max_densities, bins=100)
    # histo_as_array = np.concatenate((histo[1][1:].reshape(-1, 1), histo[0].reshape(-1, 1)), axis=1)
    # plot_histogram(histo_as_array, 'Max Density', 'Frequency', normalized=True)
    #
    # max_minus_min_densities = data.max(axis=(1, 2)) - data.min(axis=(1, 2))
    # print(max_minus_min_densities.shape)
    # histo = np.histogram(max_densities, bins=100)
    # histo_as_array = np.concatenate((histo[1][1:].reshape(-1, 1), histo[0].reshape(-1, 1)), axis=1)
    # plot_histogram(histo_as_array, 'Max-Min Density', 'Frequency', normalized=True)

    # all_data = read_cancer_data('./data/cancer_data.txt')
    # max_model = PreDefinedThresholdMaximumDensity('Predefined Threshold on Max', 6, None)
    # max_diff_model = PreDefinedThresholdMaximumDiffDensity('Predefined Threshold on Max Diff', 5, None)
    # max_model_scipy = SigmoidScipyCurveFitMaximumDensity('Sigmoid Curve Fit on Max',
    #                                                      [1, np.median(all_data[0].max((-2,-1))),1,0],
    #                                                      [0, 1])
    # max_model_scipy.fit(all_data[0], all_data[1])
    # max_diff_model_scipy = SigmoidScipyCurveFitMaximumDiffDensity('Sigmoid Curve Fit on Max Diff',
    #                                                               [1, np.median(all_data[0].max((-2,-1)) - all_data[0].min((-2,-1))),1,0],
    #                                                      [0, 1])
    # max_diff_model_scipy.fit(all_data[0], all_data[1])
    # optimal_thresh_on_diff = OptimalThresholdMaximumDiffDensity('Optimal Threshold on Max Diff', None)
    # optimal_thresh_on_diff.fit(all_data[0], all_data[1])
    # optimal_thresh_on_max = OptimalThresholdMaximumDiffDensity('Optimal Threshold on Max', None)
    # optimal_thresh_on_max.fit(all_data[0], all_data[1])
    # evaluator = TestPerformances([max_model, max_diff_model, max_model_scipy, max_diff_model_scipy, optimal_thresh_on_diff,
    #                               optimal_thresh_on_max], all_data[0], all_data[1])
    # accuracy = evaluator.evaluate_single_model('Predefined Threshold on Max')
    # print(f'Accuracy: {accuracy}')
    # accuracy = evaluator.evaluate_single_model('Predefined Threshold on Max Diff')
    # print(f'Accuracy: {accuracy}')
    # accuracy = evaluator.evaluate_single_model('Sigmoid Curve Fit on Max')
    # print(f'Accuracy: {accuracy}')
    # accuracy = evaluator.evaluate_single_model('Sigmoid Curve Fit on Max Diff')
    # print(f'Accuracy: {accuracy}')
    # accuracy = evaluator.evaluate_single_model('Optimal Threshold on Max Diff')
    # print(f'Accuracy: {accuracy}')
    # accuracy = evaluator.evaluate_single_model('Optimal Threshold on Max')
    # print(f'Accuracy: {accuracy}')
    #
    # p_value_evaluator = PValueComparisonOfModelPerformancesDistribution([max_model,
    #                                                                      max_diff_model,
    #                                                                      max_model_scipy,
    #                                                                      max_diff_model_scipy, optimal_thresh_on_diff,
    #                                                                      optimal_thresh_on_max],
    #                                                                     all_data[0], all_data[1])
    # p_value_evaluator.evaluate_for_different_training_and_prediction_set(20)
    # p_value_evaluator.plot_statistics_for_single_model('Predefined Threshold on Max')
    # p_value_evaluator.plot_statistics_for_single_model('Predefined Threshold on Max Diff')
    # p_value_evaluator.plot_statistics_for_single_model('Sigmoid Curve Fit on Max')
    # p_value_evaluator.plot_statistics_for_single_model('Sigmoid Curve Fit on Max Diff')
    # p_value_evaluator.plot_statistics_for_single_model('Optimal Threshold on Max Diff')
    # p_value_evaluator.plot_statistics_for_single_model('Optimal Threshold on Max')
    #
    #
    # auc_comparison = AUCComparisonOfModelPerformances([max_model,
    #                                                    max_diff_model,
    #                                                    max_model_scipy,
    #                                                    max_diff_model_scipy, optimal_thresh_on_diff, optimal_thresh_on_max], all_data[0], all_data[1])
    # auc_comparison.evaluate_for_different_training_and_prediction_set(5, 100)
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Predefined Threshold on Max')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Predefined Threshold on Max Diff')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Sigmoid Curve Fit on Max')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Sigmoid Curve Fit on Max Diff')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Optimal Threshold on Max Diff')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Optimal Threshold on Max')
    # auc_comparison.plot_auc_on_all_models_on_multiple_splits()
    all_data = read_cancer_data('./data/cancer_data.txt')
    all_images = torch.as_tensor(all_data[0], dtype=torch.float32).unsqueeze(1)
    labels = torch.as_tensor(all_data[1], dtype=torch.float32)

    data = DataWrapper(all_images, labels)
    data_loader = DataLoader(data, batch_size=32, shuffle=True)
    model = CNNClassifier2D((30, 30), **{"activation_func": nn.ReLU(), 'padding': '',
                                            'kernel_initializer': 'glorot_normal', 'd_rate': 0.2,
                                            'initial_kernel_size': (4, 4), 'final_activation_func': nn.Sigmoid(),
                                            'kernel_size': (4, 4), "channels": 1,
                                            "down_sampling_nb_layers": 3})
    trainer = Trainer(model)
    trainer.train(data_loader, data_loader)
