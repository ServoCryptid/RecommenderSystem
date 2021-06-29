import logging

import neat.population as pop
import neat.experiments.reco_sys.config as c
from neat.visualize import draw_net
from tqdm import tqdm
import time
from multiprocessing import Pool


def run_neat(index_population_size):
    c.RecoSysConfig.set_global_var(index_population_size)
    neat = pop.Population(c.RecoSysConfig)
    # print(f"Indexes used: {index_population_size}")
    # print("Generated the population!")
    # logger.info(f"Indexes used: {index_population_size}")
    # logger.info("Generated the population!")
    solution, generation, mae, r2 = neat.run()


if __name__ == '__main__':
    start_time = time.time()
    number_processes = 8

    # Create and configure logger
    # logging.basicConfig(filename=r"C:\Users\laris\PycharmProjects\KaggleRS\experiments\logs\recosys_run.log",
    #                     format='%(asctime)s %(message)s',
    #                     filemode='w',
    #                     level=logging.DEBUG,
    #                     force=True)
    # logger = logging.getLogger()
    #
    # logger.setLevel(level=logging.DEBUG)

    with Pool(5) as pool:
        pool.map(run_neat, range(5))

    # run all the desired lengths from sizes_to_try list



