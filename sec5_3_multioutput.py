import tensorflow as tf


from data_loader.multi_output import DataGenerator
from models.multi_output import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from figures.plot_maker_multioutput import PlotMaker
import time


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    config = process_config("configs/multi_output.json")

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data = DataGenerator(config)
    
    # create an instance of the model you want
    model = ExampleModel(config)
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
    plotmaker.write_all_plots("figures/multi_output")
    plotmaker.write_results_json("figures/multi_output")


if __name__ == '__main__':
    main()
