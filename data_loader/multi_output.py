import numpy as np
import sklearn
import sklearn.model_selection


class DataGenerator:
    def __init__(self, config):
        self.config = config

        np.random.seed(0)

        input_samples = np.random.rand(self.config.n_samples, 3)
        output_samples = np.zeros([self.config.n_samples, 2])
        for i in range(self.config.n_samples):
            x = input_samples[i, :]
            output_samples[i, 0] = 3.0 * x[0] ** 3 + np.exp(np.cos(10.0 * x[1]) * np.cos(5.0 * x[0]) ** 2) \
                                   + np.exp(np.sin(7.5 * x[2]))
            output_samples[i, 1] = 2.0 * x[0] ** 2 + np.exp(np.cos(10.0 * x[0]) * np.cos(5.0 * x[1]) ** 2) \
                                   + np.exp(np.sin(7.5 * x[2] * x[2]))
        w1 = 1
        w2 = 1.5
        output_samples[:, 0] = output_samples[:, 0] + w1 * np.random.rand(self.config.n_samples)
        output_samples[:, 1] = output_samples[:, 1] + w2 * np.random.rand(self.config.n_samples)

        dataset = np.zeros((self.config.n_samples, 5))
        dataset[:, :3] = [[sample[i] for i in range(3)] for sample in input_samples]
        dataset[:, 3:] = [[sample[i] for i in range(2)] for sample in output_samples]

        dataset_train, dataset_test = sklearn.model_selection.train_test_split(
            dataset, test_size=0.20, random_state=42
        )

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(dataset_train)
        self.data_train = self.scaler.transform(dataset_train)
        self.data_test = self.scaler.transform(dataset_test)

    def next_batch(self, batch_size):
        idx = np.random.permutation(self.data_train.shape[0])[:batch_size]
        yield self.data_train[idx, :3].reshape((batch_size, 3)), self.data_train[idx, 3:].reshape((batch_size, 2))

    def next_batch_valid(self, batch_size):
        idx = np.random.permutation(self.data_test.shape[0])[:batch_size]
        yield self.data_test[idx, :3].reshape((batch_size, 3)), self.data_test[idx, 3:].reshape((batch_size, 2))