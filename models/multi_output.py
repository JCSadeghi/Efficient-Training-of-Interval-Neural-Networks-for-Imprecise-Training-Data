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

        self.x = tf.placeholder(tf.float32, shape=[None, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

        # network architecture
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0015)
        d1 = tf.layers.dense(self.x, 100, activation=tf.nn.relu, name="dense1", kernel_regularizer=regularizer, bias_regularizer=regularizer)
        d3 = tf.layers.dense(d1, 2, name="dense3", kernel_regularizer=regularizer, bias_regularizer=regularizer)

        self.predictions = d3

        self.h_variable = tf.get_variable(name='model_width', initializer=tf.initializers.zeros, shape=[1])
        sig = tf.get_variable(name='sigma', initializer=tf.initializers.ones, shape=[1, 2])
        self.sig_hat = tf.div(sig, tf.sqrt(tf.reduce_sum(tf.square(sig), axis=1)))

        l2_loss = tf.losses.get_regularization_loss()

        with tf.name_scope("loss"):
            loss_ae = tf.losses.absolute_difference(self.predictions, self.y, reduction=tf.losses.Reduction.NONE)
            loss_ae_norm = tf.divide(loss_ae, tf.tile(self.sig_hat, [tf.shape(self.y)[0], 1]))
            loss_max = tf.reduce_max(loss_ae_norm, axis=[0, 1])
            loss_mse = tf.losses.mean_squared_error(self.predictions, self.y)
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(loss_ae), axis=0))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.learning_rate = tf.train.exponential_decay(self.config.learning_rate_init, self.global_step_tensor,
                                                                self.config.decay_steps,
                                                                self.config.learning_rate_decay, staircase=True)
                optimiser = tf.train.AdamOptimizer(self.learning_rate)

                if self.config.loss == "mse":
                    self.loss = tf.reduce_sum(loss_mse)
                elif self.config.loss == "max":
                    self.loss = loss_max
                    self.loss += l2_loss
                    h_loss = tf.losses.mean_squared_error(self.loss, tf.squeeze(self.h_variable))
                    self.loss += h_loss

                else:
                    print("unknown loss")
                    exit(0)

                self.train_step = optimiser.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_predictions(self, sess, data, scaler, rescale=True):
        n_datapoints = data.shape[0]
        train_feed_dict = {
            self.x: data[:, :3].reshape((n_datapoints, 3)),
            self.is_training: False
        }
        predictions, sig_hat = sess.run([self.predictions, self.sig_hat], feed_dict=train_feed_dict)
        data_predictions = np.zeros(shape=(n_datapoints, 5))
        data_predictions[:, :3] = data[:, :3]
        data_predictions[:, 3:] = np.array(predictions).reshape((n_datapoints, 2))[:, :]
        if rescale:
            data_predictions = scaler.inverse_transform(data_predictions)

        model_half_width = self.model_half_width_actual(scaler, sess, rescale)
        model_uncertainty = np.tile(model_half_width, (data.shape[0], 1))

        return (data_predictions[:, :3], data_predictions[:, 3:],
                data_predictions[:, 3:] + model_uncertainty,
                data_predictions[:, 3:] - model_uncertainty)

    def model_half_width(self, scaler, rescale=True):
        if rescale:
            return self.h * scaler.scale_[3:]
        else:
            return self.h

    def model_half_width_actual(self, scaler, sess, rescale=True):
        train_feed_dict = {
            self.is_training: False
        }
        sig_hat = sess.run(self.sig_hat, feed_dict=train_feed_dict)
        model_half_width = self.model_half_width(scaler, rescale)
        return np.array(sig_hat[0]) * model_half_width