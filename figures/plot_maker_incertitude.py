import matplotlib.pyplot as plt
import numpy as np
import scipy
import json
import matplotlib.patches
from matplotlib.collections import PatchCollection


class PlotMaker:
    def __init__(self, trainer, model, data):
        self.trainer = trainer
        self.model = model
        self.data = data

    def write_all_plots(self, output_dir):
        self.loss_plot(output_dir)
        predictions_in, predictions_center, predictions_upper, predictions_lower = self.model.get_predictions(
            self.trainer.sess, self.data.data_train, self.data.scaler)
        self.plot_model(output_dir, predictions_in, predictions_center, predictions_upper, predictions_lower)

    def loss_plot(self, output_dir):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch', fontsize='xx-large')
        ax1.set_ylabel('Maximum Absolute Error (Minibatch)', color=color, fontsize='xx-large')
        ax1.semilogy(self.trainer.loss_graph, color=color, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color, labelsize='x-large')
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Learning Rate', color=color, fontsize='xx-large')
        ax2.plot(self.trainer.lr_graph, color=color, label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor=color, labelsize='x-large')
        fig.tight_layout()
        handles, labels = [], []
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        plt.legend(handles, labels)
        plt.savefig("/".join([output_dir, "loss.png"]), format='png')
        plt.savefig("/".join([output_dir, "loss.pdf"]), format='pdf')
        plt.close()

    def plot_model(self, output_dir, predictions_in, predictions_center, predictions_upper, predictions_lower):
        train_data_original = self.data.scaler.inverse_transform(self.data.data_train)
        test_data_original = self.data.scaler.inverse_transform(self.data.data_test)
        predictions_in, predictions_center, predictions_upper, predictions_lower = zip(
            *sorted(zip(predictions_in, predictions_center, predictions_upper, predictions_lower))
        )
        fig, ax = plt.subplots(1)
        plot_rect(ax, train_data_original, fc='r')
        plt.errorbar(test_data_original[:, 0], test_data_original[:, 2], xerr=test_data_original[:, 1],
                     yerr=test_data_original[:, 3], color='y', ls='none', label='Testing Data')
        plt.plot(predictions_in, predictions_center, color='b', label='Interval Network')
        plt.plot(predictions_in, predictions_upper, color='b')
        plt.plot(predictions_in, predictions_lower, color='b')
        plt.ylabel('y', fontsize='xx-large')
        plt.xlabel('x', fontsize='xx-large')
        plt.tick_params(labelsize='x-large')
        red_patch = matplotlib.patches.Patch(color='red')
        handles, labels = [], []
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        handles.append(red_patch)
        labels.append('Training Data')
        plt.legend(handles, labels)
        plt.savefig("/".join([output_dir, "trained_model.pdf"]), format='pdf')
        plt.savefig("/".join([output_dir, "trained_model.png"]), format='png')
        plt.close()

    def write_results_json(self, output_dir):
        v_lower, v_upper, v_estimator, v_estimator_uncertainty, v_train_set, n_violate_bound_test = self.produce_results()

        model_half_width = self.model.model_half_width(self.data.scaler)
        results = {"v_lower": v_lower,
                   "v_upper": v_upper,
                   "n_v": str(n_violate_bound_test),
                   "v_estimator": str(v_estimator),
                   "v_estimator_std": str(v_estimator_uncertainty),
                   "h": str(model_half_width),
                   "v_train_set": str(v_train_set)}
        with open("/".join([output_dir, "results.txt"]), 'w') as outfile:
            json.dump(results, outfile)

    def produce_results(self):
        corners_1 = self.data.data_test
        corners_1[:, 0] -= self.data.data_test[:, 1]

        (_,
         _,
         predictions_upper,
         predictions_lower) = self.model.get_predictions(self.trainer.sess, corners_1, self.data.scaler,
                                                         rescale=False)

        corners_2 = self.data.data_test
        corners_2[:, 0] += self.data.data_test[:, 1]

        (_,
         _,
         predictions_upper_,
         predictions_lower_) = self.model.get_predictions(self.trainer.sess, corners_2, self.data.scaler,
                                                         rescale=False)

        n_test_points = self.data.data_test.shape[0]
        n_violate_bound_test = sum(np.logical_or(
            np.logical_or(predictions_lower > self.data.data_test[:, 2] - self.data.data_test[:, 3],
                          self.data.data_test[:, 2] + self.data.data_test[:, 3] > predictions_upper),
            np.logical_or(predictions_lower_ > self.data.data_test[:, 2] - self.data.data_test[:, 3],
                          self.data.data_test[:, 2] + self.data.data_test[:, 3] > predictions_upper_))
        )

        corners_1 = self.data.data_train
        corners_1[:, 0] -= self.data.data_train[:, 1]

        (_,
         _,
         predictions_upper,
         predictions_lower) = self.model.get_predictions(self.trainer.sess, corners_1, self.data.scaler,
                                                         rescale=False)

        corners_2 = self.data.data_train
        corners_2[:, 0] += self.data.data_train[:, 1]

        (_,
         _,
         predictions_upper_,
         predictions_lower_) = self.model.get_predictions(self.trainer.sess, corners_2, self.data.scaler,
                                                         rescale=False)

        n_train_points = self.data.data_train.shape[0]
        n_violate_bound_train = sum(np.logical_or(
            np.logical_or(predictions_lower > self.data.data_train[:, 2] - self.data.data_train[:, 3],
                          self.data.data_train[:, 2] + self.data.data_train[:, 3] > predictions_upper),
            np.logical_or(predictions_lower_ > self.data.data_train[:, 2] - self.data.data_train[:, 3],
                          self.data.data_train[:, 2] + self.data.data_train[:, 3] > predictions_upper_))
        )
        v_train_set = n_violate_bound_train / n_train_points

        v_estimator = n_violate_bound_test / n_test_points
        v_estimator_uncertainty = np.sqrt(v_estimator * (1 - v_estimator) / n_test_points)

        def v_lower_bound_function(p):
            return scipy.stats.binom.cdf(n_test_points - n_violate_bound_test,
                                         n_test_points,
                                         1 - p) - self.model.config.beta

        def v_upper_bound_function(p):
            return (scipy.stats.binom.cdf(n_test_points, n_test_points, 1 - p)
                    - scipy.stats.binom.cdf(n_test_points - n_violate_bound_test - 1,
                                            n_test_points,
                                            1 - p)
                    - self.model.config.beta)

        try:
            if n_violate_bound_test > 0:
                v_lower = scipy.optimize.bisect( lambda p: v_lower_bound_function(p), 0, 1)
            else:
                v_lower = 0
            v_upper = scipy.optimize.bisect(lambda p: v_upper_bound_function(p), 0, 1)
        except:
            v_lower = 0
            v_upper = 0

        return v_lower, v_upper, v_estimator, v_estimator_uncertainty, v_train_set, n_violate_bound_test


def plot_rect(ax, data, fc='r'):
    errorboxes = []
    for i in range(data.shape[0]):
        rect = matplotlib.patches.Rectangle((data[i, 0] - data[i, 1],
                                             data[i, 2] - data[i, 3]),
                                            2 * data[i, 1],
                                            2 * data[i, 3])
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=fc, alpha=0.2, edgecolors='b')

    # Add collection to axes
    ax.add_collection(pc)
