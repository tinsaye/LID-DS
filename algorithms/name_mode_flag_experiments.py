import math
import os
import sys
from pprint import pprint

from algorithms.decision_engines.ae import AE
from algorithms.decision_engines.mlp import MLP
from algorithms.decision_engines.som import Som
from algorithms.features.impl.concat import Concat
from algorithms.features.impl.concat_strings import ConcatStrings
from algorithms.features.impl.flags import Flags
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.mode import Mode
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.ngram_minus_one import NgramMinusOne
from algorithms.features.impl.one_hot_encoding import OneHotEncoding
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.ids import IDS
from algorithms.persistance import save_to_json
from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction

if __name__ == '__main__':
    # getting the LID-DS base path from argument or environment variable
    if len(sys.argv) > 1:
        lid_ds_base_path = sys.argv[1]
    else:
        try:
            lid_ds_base_path = os.environ['LID_DS_BASE']
        except KeyError:
            raise ValueError(
                "No LID-DS Base Path given. Please specify as argument or set Environment Variable "
                "$LID_DS_BASE")
    scenarios = sorted(os.listdir(os.path.join(lid_ds_base_path, 'LID-DS-2019')))

    current_input_size = 0.0

    # parameter lists
    enc_sizes = range(2, 18, 2)  # [2, 4, ..., 16]
    w2v_epochs = [50, 500, 5000]
    n_gram_sizes = [3, 5, 7, 9]
    stream_sum_sizes = [1] + list(range(20, 1200, 100))  # [1, 200, ..., 1000]


    # static bb features
    name = SyscallName()
    flags = Flags()
    mode = Mode()
    concat = Concat([name, flags, mode])
    concat_strings = ConcatStrings(concat)


    for enc_size in enc_sizes:
        for epochs in w2v_epochs:
            for n in n_gram_sizes:
                for stream_size in stream_sum_sizes:
                    for scenario in scenarios:
                        current_input_size = enc_size * n
                        w2v = W2VEmbedding(word=concat_strings,
                                           epochs=epochs,
                                           vector_size=enc_size,
                                           window_size=10)

                        ngram = Ngram([w2v], False, n)
                        som = Som(input_vector=ngram, epochs=100)
                        ae = AE(input_vector=ngram, hidden_size=int(math.sqrt(current_input_size)))

                        inte = IntEmbedding(name)
                        ohe = OneHotEncoding(inte)
                        ngram_minus_one = NgramMinusOne(ngram=ngram,
                                                        element_size=enc_size)

                        mlp = MLP(
                            input_vector=ngram_minus_one,
                            output_label=ohe,
                            hidden_size=64,
                            hidden_layers=3,
                            batch_size=512,
                            learning_rate=0.003
                        )

                        algorithms = [som, ae, mlp]

                        for algorithm in algorithms:
                            stream_sum = StreamSum(algorithm, False, stream_size)
                            scenario_path = os.path.join(lid_ds_base_path,
                                                         'LID-DS-2019',
                                                         scenario)
                            dataloader = dataloader_factory(scenario_path, direction=Direction.OPEN)

                            ids = IDS(data_loader=dataloader,
                                      resulting_building_block=stream_sum,
                                      plot_switch=False)

                            ids.determine_threshold()
                            ids.detect()
                            results = ids.performance.get_results()
                            pprint(results)

                            results['algorithm'] = type(algorithm).__name__
                            results['ngram_length'] = n
                            results['w2v_size'] = enc_size
                            results['w2v_epochs'] = epochs
                            results['stream_sum_size'] = stream_size
                            results['config'] = ids.get_config()
                            results['scenario'] = scenario
                            result_path = 'results/results_name_mode_flags.json'
                            save_to_json(results, result_path)
