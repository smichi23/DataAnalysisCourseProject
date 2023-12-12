import numpy as np
import torchvision as torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data_management_tools.data_readers import read_cancer_data, read_time_dependant_cancer_data, \
    read_historical_cancer_data
from data_management_tools.data_visualization import plot_histogram, show_images_via_slider, plot_multiple_histogram
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
    # data visualization
    data, labels = read_cancer_data('./data/cancer_data.txt')
    patients_with_cancer = np.where(labels == 1)[0]
    show_images_via_slider(data[list(patients_with_cancer)],
                           ['x [cm]', 'y [cm]', 'Intensity [-]'],
                           [-7.5, 7.5, -7.5, 7.5],
                           0, data[list(patients_with_cancer)].max(), 0)
    # Time dependant data
    # data2, labels2 = read_time_dependant_cancer_data('./data/cancer_data_timeDependency.txt')
    #
    # patients_with_cancer = np.where(labels2 == 1)[0]
    # show_images_via_slider(data2[list(patients_with_cancer)[1]],
    #                        ['x [cm]', 'y [cm]', 'Intensity [-]'],
    #                        [-7.5, 7.5, -7.5, 7.5],
    #                        0, data2[list(patients_with_cancer)].max(), 0)
    histo_dict = read_historical_cancer_data('./data/historical_cancer_data.txt', 0.2)
    for key in histo_dict.keys():
        print(key)
        for histo in histo_dict[key]:
            if key == "maxTissueDensity\n":
                label = 'Max Intensity [-]'
                vline = [6, "Previous work positive\ndetection threshold"]
            elif key == "maxDensityDifference\n":
                label = 'Max Intensity Difference [-]'
                vline = [5, "Previous work positive\ndetection threshold"]
            if histo[:, 1].sum() == 2000:
                title = "2000-patient cohort"
            elif histo[:, 1].sum() == 10000:
                title = "10000-patient cohort"
            else:
                title = ""
            plot_histogram(histo, title, label, 'Frequency [-]',
                           add_vline=vline, normalized=True)

    # Histogram of max densities
    patients_without_cancer = np.where(labels == 0)[0]
    patients_with_cancer = np.where(labels == 1)[0]
    max_densities_without = data[list(patients_without_cancer)].max(axis=(1, 2))
    max_densities_with = data[list(patients_with_cancer)].max(axis=(1, 2))
    plot_multiple_histogram([max_densities_without, max_densities_with], ['Without Cancer', 'With Cancer'],
                            'Max Intensity [-]', 'Frequency [-]', 25,
                            add_vline=[6, "Previous work positive\ndetection threshold"], normalized=True)

    # Histogram of max densities differences
    max_minus_min_densities_without = data[list(patients_without_cancer)].max(axis=(1, 2)) \
                                      - data[list(patients_without_cancer)].min(axis=(1, 2))
    max_minus_min_densities_with = data[list(patients_with_cancer)].max(axis=(1, 2)) \
                                   - data[list(patients_with_cancer)].min(axis=(1, 2))
    plot_multiple_histogram([max_minus_min_densities_without, max_minus_min_densities_with],
                            ['Without Cancer', 'With Cancer'],
                            'Max Intensity Difference [-]', 'Frequency [-]', 25,
                            add_vline=[5, "Previous work positive\ndetection threshold"], normalized=True)

    # all_data = read_cancer_data('./data/cancer_data.txt')
    # max_model = PreDefinedThresholdMaximumDensity('Predefined Threshold on Max', 6, None)
    # max_diff_model = PreDefinedThresholdMaximumDiffDensity('Predefined Threshold on Max Diff', 5, None)
    # max_model_scipy = SigmoidScipyCurveFitMaximumDensity('Sigmoid Curve Fit on Max',
    #                                                      [1, np.median(all_data[0].max((-2, -1))), 1, 0],
    #                                                      [0, 1])
    # max_model_scipy.fit(all_data[0], all_data[1])
    # max_diff_model_scipy = SigmoidScipyCurveFitMaximumDiffDensity('Sigmoid Curve Fit on Max Diff',
    #                                                               [1, np.median(
    #                                                                   all_data[0].max((-2, -1)) - all_data[0].min(
    #                                                                       (-2, -1))), 1, 0],
    #                                                               [0, 1])
    # max_diff_model_scipy.fit(all_data[0], all_data[1])
    # optimal_thresh_on_diff = OptimalThresholdMaximumDiffDensity('Optimal Threshold on Max Diff', None)
    # optimal_thresh_on_diff.fit(all_data[0], all_data[1])
    # optimal_thresh_on_max = OptimalThresholdMaximumDiffDensity('Optimal Threshold on Max', None)
    # optimal_thresh_on_max.fit(all_data[0], all_data[1])
    # torch.manual_seed(0)
    # transforms = torchvision.transforms.Compose([torchvision.transforms.RandomRotation(degrees=(0, 180)),
    #                                              torchvision.transforms.RandomHorizontalFlip(p=0.5),
    #                                              torchvision.transforms.RandomVerticalFlip(p=0.5)])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # deep_model = CNNClassifier2D((30, 30), **{"activation_func": nn.ReLU(), 'padding': '',
    #                                           'kernel_initializer': 'glorot_normal', 'd_rate': 0.2,
    #                                           'initial_kernel_size': (2, 2), 'final_activation_func': nn.Sigmoid(),
    #                                           'kernel_size': (3, 3), "channels": 1,
    #                                           "down_sampling_nb_layers": 3, "transform": transforms,
    #                                           "device": device})
    # deep_model.to(device)
    #
    #
    # def early_stoping_condition(accuracy_history):
    #     if len(accuracy_history) < 200:
    #         return False
    #     elif np.asarray(accuracy_history)[-50:].mean() < 0.7:
    #         return False
    #     elif np.asarray(accuracy_history)[-50:].std() < 0.015 and np.asarray(accuracy_history)[-1] > 2 * np.asarray(accuracy_history)[-50:].mean():
    #         return True
    #     else:
    #         return False
    #
    #
    # optimizer = torch.optim.Adam(deep_model.parameters(), lr=0.01)
    # lambda1 = lambda epoch: 1 - 0.9 ** (epoch / 200)
    # trainer = Trainer(deep_model, **{'loss': nn.BCELoss(), 'optimizer': optimizer,
    #                                  'epochs': 1500, "show_plots_every_training": True,
    #                                  "early_stoping_condition": (early_stoping_condition, 'metric', 'accuracy'),
    #                                  "lr_scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)})
    # deep_model.set_trainer(trainer)
    # deep_model.fit(all_data[0], all_data[1])
    # evaluator = TestPerformances(
    #     [max_model, max_diff_model, max_model_scipy, max_diff_model_scipy, optimal_thresh_on_diff,
    #      optimal_thresh_on_max, deep_model], all_data[0], all_data[1])
    #
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
    # accuracy = evaluator.evaluate_single_model('CNNClassifier2D')
    # print(f'Accuracy: {accuracy}')
    # p_value_evaluator = PValueComparisonOfModelPerformancesDistribution([max_model,
    #                                                                      max_diff_model,
    #                                                                      max_model_scipy,
    #                                                                      max_diff_model_scipy, optimal_thresh_on_diff,
    #                                                                      optimal_thresh_on_max, deep_model],
    #                                                                     all_data[0], all_data[1])
    # p_value_evaluator.evaluate_for_different_training_and_prediction_set(10)
    # p_value_evaluator.plot_statistics_for_single_model('Predefined Threshold on Max')
    # p_value_evaluator.plot_statistics_for_single_model('Predefined Threshold on Max Diff')
    # p_value_evaluator.plot_statistics_for_single_model('Sigmoid Curve Fit on Max')
    # p_value_evaluator.plot_statistics_for_single_model('Sigmoid Curve Fit on Max Diff')
    # p_value_evaluator.plot_statistics_for_single_model('Optimal Threshold on Max Diff')
    # p_value_evaluator.plot_statistics_for_single_model('Optimal Threshold on Max')
    # p_value_evaluator.plot_statistics_for_single_model('CNNClassifier2D')

    # auc_comparison = AUCComparisonOfModelPerformances([max_model,
    #                                                    max_diff_model,
    #                                                    max_model_scipy,
    #                                                    max_diff_model_scipy, optimal_thresh_on_diff,
    #                                                    optimal_thresh_on_max,
    #                                                    deep_model], all_data[0], all_data[1])
    # auc_comparison.evaluate_for_different_training_and_prediction_set(5, 100)
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Predefined Threshold on Max')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Predefined Threshold on Max Diff')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Sigmoid Curve Fit on Max')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Sigmoid Curve Fit on Max Diff')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Optimal Threshold on Max Diff')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('Optimal Threshold on Max')
    # auc_comparison.plot_auc_for_single_model_on_multiple_splits('CNNClassifier2D')
    # auc_comparison.plot_auc_on_all_models_on_multiple_splits()
