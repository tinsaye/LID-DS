import os
import csv
from pprint import pprint

from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.path_evilness import PathEvilness
from algorithms.features.impl.syscalls_in_time_window import SyscallsInTimeWindow
from algorithms.features.impl.w2v_embedding import W2VEmbedding

from algorithms.decision_engines.som import Som
from algorithms.ids import IDS
from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction

if __name__ == '__main__':
    # data loader for scenario
    base_path = '/home/felix/repos/LID-DS/LID-DS-2019'
    scenario = 'CVE-2017-7529'
    dataloader = dataloader_factory(os.path.join(base_path, scenario), direction=Direction.OPEN)

    N_GRAM_SIZES = [3, 5, 7, 9, 15]
    DIMENSIONS = [3, 5, 7, 9, 15]
    EPOCHS = [10, 50, 100, 1000]

    header_exists = False

    for n_gram_size in N_GRAM_SIZES:
        for dimensions in DIMENSIONS:
            for epochs in EPOCHS:
                print('Running SOM algorithm with config:')
                print(f'   Scenario: {scenario}')
                print(f'   n_gram Size: {n_gram_size}')
                print(f'   w2v dimensions: {dimensions}')
                print(f'   epochs: {epochs}')

                w2v = W2VEmbedding(vector_size=dimensions,
                                   epochs=100,
                                   path='Models',
                                   force_train=True,
                                   distinct=True,
                                   window_size=n_gram_size,
                                   thread_aware=True,
                                   scenario_path=dataloader.scenario_path)


                ngram = Ngram(
                    feature_list=[w2v],
                    thread_aware=True,
                    ngram_length=n_gram_size
                )

                pe = PathEvilness(scenario_path=dataloader.scenario_path)

                sit = SyscallsInTimeWindow(
                    window_length_in_s=5
                )

                DE = Som(
                    epochs=epochs
                )

                # define the used features
                ids = IDS(data_loader=dataloader,
                          feature_list=[ngram, pe, sit],
                          decision_engine=DE,
                          plot_switch=False)

                ids.train_decision_engine()
                ids.determine_threshold()
                ids.do_detection()

                performance = ids.performance.get_performance()
                pprint(performance)
                stats = {}
                stats['scenario'] = scenario
                stats['ngram'] = n_gram_size
                stats['dimensions'] = dimensions
                stats['epochs'] = epochs
                stats['alarm_count'] = performance['alarm_count']
                stats['cfp_exp'] = performance['consecutive_false_positives_exploits']
                stats['cfp_norm'] = performance['consecutive_false_positives_normal']
                stats['detection_rate'] = performance['detection_rate']
                stats['fp'] = performance['false_positives']

                csv_file = "stats.csv"
                csv_columns = ['scenario',
                               'ngram',
                               'dimensions',
                               'epochs',
                               'alarm_count',
                               'cfp_exp',
                               'cfp_norm',
                               'detection_rate',
                               'fp']
                try:
                    with open(csv_file, 'a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
                        if header_exists is False:
                            writer.writeheader()
                            header_exists = True
                        writer.writerow(stats)
                except IOError:
                    print("I/O error")
