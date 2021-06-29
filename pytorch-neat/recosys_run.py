import neat.population as pop
import neat.experiments.reco_sys.config as c
from neat.visualize import draw_net
import time
from multiprocessing import Pool


def run_neat(index_population_size):
    c.RecoSysConfig.set_global_var(index_population_size)
    neat = pop.Population(c.RecoSysConfig)
    solution, generation, mae, r2 = neat.run()


if __name__ == '__main__':
    start_time = time.time()
    number_processes = 8

    with Pool(5) as pool:  # run all the desired lengths from sizes_to_try list
        pool.map(run_neat, range(5))




