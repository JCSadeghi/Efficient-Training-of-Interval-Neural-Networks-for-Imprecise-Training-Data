from base.base_model import BaseModel
import tensorflow as tf
import numpy as np


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_uncertainty = tf.placeholder(tf.float32, shape=[None, 1])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_uncertainty = tf.placeholder(tf.float32, shape=[None, 1])

        # network architecture
        d1 = tf.layers.dense(self.x, 10, activation=tf.nn.tanh, name="dense1")

        self.predictions = tf.layers.dense(d1, 1, name="dense2")
        sigma = tf.layers.dense(d1, 1, activation=tf.nn.softplus, name="dense2_uncertainty")
        self.sigma_hat = tf.divide(sigma, tf.reduce_mean(sigma))

        self.h_variable = tf.get_variable(name='model_width', initializer=tf.initializers.zeros, shape=[1])

        with tf.name_scope("loss"):
            a = tf.div(tf.losses.absolute_difference(tf.add(self.y, self.y_uncertainty),
                                                     self.predictions,
                                                     reduction=tf.losses.Reduction.NONE), self.sigma_hat)
            b = tf.div(tf.losses.absolute_difference(tf.subtract(self.y, self.y_uncertainty),
                                                     self.predictions,
                                                     reduction=tf.losses.Reduction.NONE), self.sigma_hat)
            self.loss = tf.math.maximum(
                tf.reduce_max(a + tf.math.abs(tf.math.multiply(self.x_uncertainty, tf.gradients(a, self.x)))),
                tf.reduce_max(b + tf.math.abs(tf.math.multiply(self.x_uncertainty, tf.gradients(b, self.x)))))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.learning_rate = tf.train.exponential_decay(self.config.learning_rate_init, self.global_step_tensor,
                                                           self.config.decay_steps, self.config.learning_rate_decay,
                                                           staircase=True)
                optimiser = tf.train.AdamOptimizer(self.learning_rate)

                h_loss = tf.losses.mean_squared_error(self.loss, self.h_variable[0])

                self.loss += h_loss

                self.train_step = optimiser.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_predictions(self, sess, data, scaler, rescale=True):
        n_datapoints = data.shape[0]
        train_feed_dict = {
            self.x: data[:, 0].reshape((n_datapoints, 1)),
            self.is_training: False
        }
        predictions, sigma_hat = sess.run(
            [self.predictions, self.sigma_hat], feed_dict=train_feed_dict)
        data_predictions = np.zeros(shape=(n_datapoints, 4))
        data_predictions[:, :] = data[:, :]
        data_predictions[:, 2] = np.array(predictions).reshape((n_datapoints, 1))[:, 0]
        if rescale:
            data_predictions = scaler.inverse_transform(data_predictions)

        model_half_width = self.model_half_width(scaler, rescale)
        uncertainty_prediction = np.array(sigma_hat).reshape((n_datapoints, 1))[:, 0] * model_half_width

        return (data_predictions[:, 0], data_predictions[:, 2], data_predictions[:, 2] + uncertainty_prediction,
                data_predictions[:, 2] - uncertainty_prediction)

    def model_half_width(self, scaler, rescale=True):
        return self.h


class ExampleModel_deeper(BaseModel):
    def __init__(self, config):
        super(ExampleModel_deeper, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_uncertainty = tf.placeholder(tf.float32, shape=[None, 1])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_uncertainty = tf.placeholder(tf.float32, shape=[None, 1])

        # network architecture
        d1 = tf.layers.dense(self.x, 20, activation=tf.nn.tanh, name="dense1")
        d2 = tf.layers.dense(d1, 10, activation=tf.nn.tanh, name="dense3")
        self.predictions = tf.layers.dense(d2, 1, name="dense2")
        sigma = tf.layers.dense(d2, 1, activation=tf.nn.softplus, name="dense2_uncertainty")
        self.sigma_hat = tf.divide(sigma, tf.reduce_mean(sigma))

        self.h_variable = tf.get_variable(name='model_width', initializer=tf.initializers.zeros, shape=[1])

        with tf.name_scope("loss"):
            a = tf.div(tf.losses.absolute_difference(tf.add(self.y, self.y_uncertainty),
                                                     self.predictions,
                                                     reduction=tf.losses.Reduction.NONE), self.sigma_hat)
            b = tf.div(tf.losses.absolute_difference(tf.subtract(self.y, self.y_uncertainty),
                                                     self.predictions,
                                                     reduction=tf.losses.Reduction.NONE), self.sigma_hat)
            self.loss = tf.math.maximum(
                tf.reduce_max(a + tf.math.abs(tf.math.multiply(self.x_uncertainty, tf.gradients(a, self.x)))),
                tf.reduce_max(b + tf.math.abs(tf.math.multiply(self.x_uncertainty, tf.gradients(b, self.x)))))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.learning_rate = tf.train.exponential_decay(self.config.learning_rate_init, self.global_step_tensor,
                                                           self.config.decay_steps, self.config.learning_rate_decay,
                                                           staircase=True)
                optimiser = tf.train.AdamOptimizer(self.learning_rate)

                h_loss = tf.losses.mean_squared_error(self.loss, self.h_variable[0])

                self.loss += h_loss

                self.train_step = optimiser.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_predictions(self, sess, data, scaler, rescale=True):
        n_datapoints = data.shape[0]
        train_feed_dict = {
            self.x: data[:, 0].reshape((n_datapoints, 1)),
            self.is_training: False
        }
        predictions, sigma_hat = sess.run(
            [self.predictions, self.sigma_hat], feed_dict=train_feed_dict)
        data_predictions = np.zeros(shape=(n_datapoints, 4))
        data_predictions[:, :] = data[:, :]
        data_predictions[:, 2] = np.array(predictions).reshape((n_datapoints, 1))[:, 0]
        if rescale:
            data_predictions = scaler.inverse_transform(data_predictions)

        model_half_width = self.model_half_width(scaler, rescale)
        uncertainty_prediction = np.array(sigma_hat).reshape((n_datapoints, 1))[:, 0] * model_half_width

        return (data_predictions[:, 0], data_predictions[:, 2], data_predictions[:, 2] + uncertainty_prediction,
                data_predictions[:, 2] - uncertainty_prediction)

    def model_half_width(self, scaler, rescale=True):
        return self.h
