import tensorflow as tf

from data_loader.data_generator_incertitude import DataGenerator, ConstWidth
from models.model_incertitude import ExampleModel_deeper
from trainers.example_trainer_incertitude import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from figures.plot_maker_incertitude import PlotMaker
import time


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    config = process_config("configs/interval_neural_net.json")

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data = DataGenerator(config, ConstWidth())
    
    # create an instance of the model you want
    model = ExampleModel_deeper(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    # load model if exists
    model.load(sess)
    # here you train your model
    t = time.time()
    trainer.train()
    elapsed = time.time() - t
    print("".join(["Elapsed time: ", str(elapsed)]))

    plotmaker = PlotMaker(trainer, model, data)
    plotmaker.write_all_plots("figures/experiment1_incertitude_deeper")
    plotmaker.write_results_json("figures/experiment1_incertitude_deeper")


if __name__ == '__main__':
    main()
