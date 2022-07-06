import os
import sys
import math
import argparse

from pprint import pprint

from algorithms.ids import IDS
from dataloader.direction import Direction
from algorithms.decision_engines.ae import AE
from algorithms.features.impl.mode import Mode
from algorithms.decision_engines.mlp import MLP
from algorithms.decision_engines.som import Som
from algorithms.persistance import save_to_json
from algorithms.features.impl.flags import Flags
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.concat import Concat
from algorithms.features.impl.stream_sum import StreamSum
from dataloader.dataloader_factory import dataloader_factory
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.features.impl.concat_strings import ConcatStrings
from algorithms.features.impl.ngram_minus_one import NgramMinusOne
from algorithms.features.impl.one_hot_encoding import OneHotEncoding

if __name__ == '__main__':
    # argument handling
    parser = argparse.ArgumentParser(description='LID-DS experiment')
    parser.add_argument('-v', dest='vector_size', type=int)
    parser.add_argument('-e', dest='w2v_epochs', type=int)
    parser.add_argument('-n', dest='n_gram_size', type=int)
    parser.add_argument('-w', dest='stream_window', type=int)
    parser.add_argument('-s', dest='scenario', type=int)
    parser.add_argument('-a', dest='algorithm', type=str)
    args = parser.parse_args()



    # static bb features
    name = SyscallName()
    flags = Flags()
    mode = Mode()
    concat = Concat([name, flags, mode])
    concat_strings = ConcatStrings(concat)

    # dynamic bb features
    w2v = W2VEmbedding(word=concat_strings,
                       epochs=args.w2v_epochs,
                       vector_size=args.vector_size,
                       window_size=10)
    ngram = Ngram([w2v], False, args.n_gram_size)

    # algorithms
    if args.algorithm == 'som':
        algorithm = Som(input_vector=ngram, epochs=100)
        config = {
            'epochs': 100
        }
    elif args.algorithm == 'ae':
        input_size = args.vector_size * args.n_gram_size
        algorithm = AE(input_vector=ngram, hidden_size=int(math.sqrt(input_size)))
        config = {
            'hidden_size': int(math.sqrt(input_size))
        }
    elif args.algorithm == 'mlp':
        inte = IntEmbedding(name)
        ohe = OneHotEncoding(inte)
        ngram_minus_one = NgramMinusOne(ngram=ngram,
                                        element_size=args.vector_size)

        algorithm = MLP(
            input_vector=ngram_minus_one,
            output_label=ohe,
            hidden_size=64,
            hidden_layers=3,
            batch_size=512,
            learning_rate=0.003
        )
        config = {
            'hidden_size': 64,
            'hidden_layers': 3,
            'batch_size': 512,
            'learning_rate': 0.003
        }
    else:
        sys.exit()

    # final bb
    stream_sum = StreamSum(algorithm, False, args.stream_window)

    # data loading
    lid_ds_base_path = os.environ['LID_DS_BASE']
    scenarios = sorted(os.listdir(os.path.join(lid_ds_base_path, 'LID-DS-2019')))
    scenario_path = os.path.join(lid_ds_base_path,
                                 'LID-DS-2019',
                                 scenarios[args.scenario])
    dataloader = dataloader_factory(scenario_path, direction=Direction.OPEN)

    # intrusion detection
    ids = IDS(data_loader=dataloader,
              resulting_building_block=stream_sum,
              plot_switch=False)

    ids.determine_threshold()
    ids.detect()

    # result handling and persisting
    results = ids.performance.get_results()
    pprint(results)

    results['algorithm'] = type(algorithm).__name__
    results['ngram_length'] = args.n_gram_size
    results['w2v_size'] = args.vector_size
    results['w2v_epochs'] = args.w2v_epochs
    results['stream_sum_size'] = args.stream_window
    results['config'] = config
    results['scenario'] = scenarios[args.scenario]
    result_path = '../results/results_name_mode_flags.json'
    save_to_json(results, result_path)
