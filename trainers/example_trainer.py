from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config, logger)
        n_training_data = data.data_train.shape[0]
        self.num_iter_per_epoch = int(n_training_data / self.config.batch_size)
        self.loss_graph = []
        self.lr_graph = []
        self.valid_graph = []

    def train_epoch(self):
        loop = (range(self.num_iter_per_epoch))
        losses = []
        for _ in loop:
            loss, learning_rate = self.train_step()
            losses.append(loss)

        loss = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        #self.model.save(self.sess)

        self.loss_graph.append(loss)
        self.lr_graph.append(learning_rate)
        try:
            batch_x, batch_y = next(self.data.next_batch_valid(self.config.batch_size))
            feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
            loss_valid = self.sess.run([self.model.loss], feed_dict=feed_dict)
            self.valid_graph.append(loss_valid)
        except:
            pass

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, learning_rate, self.model.h = self.sess.run(
            [self.model.train_step, self.model.loss, self.model.learning_rate, self.model.h_variable],
            feed_dict=feed_dict)
        return loss, learning_rate
