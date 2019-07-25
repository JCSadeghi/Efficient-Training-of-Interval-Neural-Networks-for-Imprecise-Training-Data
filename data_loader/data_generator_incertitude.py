import numpy as np
import sklearn
import sklearn.model_selection


class DataGenerator:
    def __init__(self, config, model):
        self.config = config

        np.random.seed(0)

        input_samples = np.random.random(size=(1, self.config.n_samples))
        input_samples_incertitude = (np.abs(input_samples - 0.5) + 0.1) ** -1 / 160
        output_samples, output_samples_uncertainty = model.eval(input_samples)

        dataset = np.zeros((self.config.n_samples, 4))

        dataset[:, 0] = input_samples
        dataset[:, 1] = input_samples_incertitude
        dataset[:, 2] = output_samples
        dataset[:, 3] = output_samples_uncertainty

        dataset_train, dataset_test = sklearn.model_selection.train_test_split(
            dataset, test_size=0.20, random_state=42
        )

        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False, with_std=False)
        self.scaler.fit(dataset_train)
        self.data_train = self.scaler.transform(dataset_train)
        self.data_test = self.scaler.transform(dataset_test)

    def next_batch(self, batch_size):
        idx = np.random.permutation(self.data_train.shape[0])[:batch_size]
        yield (self.data_train[idx, 0].reshape((batch_size, 1)), self.data_train[idx, 1].reshape((batch_size, 1)),
               self.data_train[idx, 2].reshape((batch_size, 1)), self.data_train[idx, 3].reshape((batch_size, 1)))


class IncreasingWidth:
    def __init__(self):
        pass

    def eval(self, x):
        return (0.3 * (15 * x * np.exp(-3 * x) + x * np.random.normal(scale=2.5 * 10 ** -2, size=x.shape)),
                x * np.random.normal(scale=2.5 * 10 ** -2, size=x.shape) / 5)


class ConstWidth:
    def __init__(self):
        pass

    def eval(self, x):
        return (0.3 * (15 * x * np.exp(-3 * x) + np.random.normal(scale=2.5 * 10 ** -2, size=x.shape)),
                (np.abs(x - 0.5) + 0.1) ** -1 / 160)
