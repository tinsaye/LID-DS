import os

from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.path_evilness import PathEvilness
from algorithms.features.impl.syscalls_in_time_window import SyscallsInTimeWindow
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.features.impl.unknown_flags import UnknownFlags
from itertools import combinations
from algorithms.persistance import save_to_json, print_as_table

from algorithms.decision_engines.som import Som
from algorithms.ids import IDS
from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction


def get_configs():
    return [
        {
            'scenario': "Bruteforce_CWE-307",
            'dim': 9,
            'n_gram_size': 7,
            'epochs': 10
        },
        {
            'scenario': "CVE-2012-2122",
            'dim': 2,
            'n_gram_size': 3,
            'epochs': 1000
        },
        {
            'scenario': "CVE-2014-0160",
            'dim': 11,
            'n_gram_size': 5,
            'epochs': 50
        },
        {
            'scenario': "CVE-2017-7529",
            'dim': 5,
            'n_gram_size': 9,
            'epochs': 10
        },
        {
            'scenario': "CVE-2018-3760",
            'dim': 3,
            'n_gram_size': 5,
            'epochs': 1000
        },
        {
            'scenario': "CVE-2019-5418",
            'dim': 11,
            'n_gram_size': 9,
            'epochs': 1000
        },
        {
            'scenario': "EPS_CWE-434",
            'dim': 2,
            'n_gram_size': 7,
            'epochs': 50
        },
        {
            'scenario': "PHP_CWE-434",
            'dim': 2,
            'n_gram_size': 4,
            'epochs': 1000
        },
        {
            'scenario': "SQL_Injection_CWE-89",
            'dim': 2,
            'n_gram_size': 3,
            'epochs': 10
        },
        {
            'scenario': "ZipSlip",
            'dim': 4,
            'n_gram_size': 4,
            'epochs': 10
        }
    ]


if __name__ == '__main__':
    base_path = '/home/felix/repos/LID-DS/LID-DS-2019'

    feature_combinations = [['pd'], ['sit'], ['uf'], ['pd', 'sit'], ['pd', 'uf'], ['sit', 'uf'], ['pd', 'sit', 'uf']]

    for config in get_configs():
        for feature_combo in feature_combinations:

            scenario = config['scenario']
            dimensions = config['dim']
            n_gram_size = config['n_gram_size']
            epochs = config['epochs']

            dataloader = dataloader_factory(os.path.join(base_path, scenario), direction=Direction.OPEN)

            # feature initialization
            w2v = W2VEmbedding(vector_size=dimensions,
                               epochs=100,
                               path='Models',
                               force_train=False,
                               distinct=True,
                               window_size=n_gram_size,
                               thread_aware=True,
                               scenario_path=dataloader.scenario_path)

            pe = PathEvilness(scenario_path=dataloader.scenario_path)
            sit = SyscallsInTimeWindow(5)
            uf = UnknownFlags()
            features = []
            if 'pd' in feature_combo:
                features.append(pe)

            if 'sit' in feature_combo:
                features.append(sit)

            if 'uf' in feature_combo:
                features.append(uf)

            print('Running SOM algorithm with config:')
            print(f'   Scenario: {scenario}')
            print(f'   n_gram Size: {n_gram_size}')
            print(f'   w2v dimensions: {dimensions}')
            print(f'   epochs: {epochs}')
            print(f'   features: {feature_combo}')

            ngram = Ngram(
                feature_list=[w2v],
                thread_aware=True,
                ngram_length=n_gram_size
            )

            DE = Som(
                epochs=epochs,
                max_size=75
            )

            # define the used features
            ids = IDS(data_loader=dataloader,
                      feature_list=[ngram] + features,
                      decision_engine=DE,
                      plot_switch=False)

            ids.train_decision_engine()
            ids.determine_threshold()
            ids.do_detection()
            DE.calculate_errors()

            stats = ids.performance.get_performance()

            stats['scenario'] = scenario
            stats['ngram'] = n_gram_size
            stats['dimensions'] = dimensions
            stats['epochs'] = epochs
            stats['features'] = feature_combo
            stats['quantization_error'] = DE.custom_fields['training_quantization_error']
            stats['topographic_error'] = DE.custom_fields['training_topographic_error']

            result_path = 'persistent_data/best_scenario_runs.json'
            save_to_json(stats, result_path)
            print_as_table(path=result_path)
