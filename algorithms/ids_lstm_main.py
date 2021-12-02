from algorithms.features.impl.thread_change_flag import ThreadChangeFlag
from algorithms.features.impl.ngram_minus_one import NgramMinusOne
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.features.impl.return_value import ReturnValue
from algorithms.features.impl.time_delta import TimeDelta
from algorithms.features.impl.threadID import ThreadID
from algorithms.features.impl.ngram import Ngram

from dataloader.data_loader_2019 import DataLoader

from algorithms.decision_engines.lstm import LSTM

from algorithms.ids import IDS

from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction

from pprint import pprint

if __name__ == '__main__':
    """
    this is an example script to show the usage uf our classes
    """
    ngram_length = 4
    embedding_size = 5
    thread_aware = True
    use_return_value = True
    use_thread_change_flag = True
    use_time_delta = False
    scenario = "CVE-2017-7529"
    scenario_path = f'../../Dataset/{scenario}/'

    # data loader for scenario
    dataloader = dataloader_factory(scenario_path, direction=Direction.CLOSE)

    element_size = embedding_size
    # embedding
    w2v = W2VEmbedding(
        vector_size=embedding_size,
        window_size=10,
        epochs=5000,
        scenario_path=scenario_path,
        path=f'Models/{scenario}/W2V/',
        force_train=True,
        distinct=True,
        thread_aware=True
    )
    feature_list = [w2v]
    if use_return_value:
        element_size += 1
        rv = ReturnValue()
        feature_list.append(rv)
    if use_time_delta:
        td = TimeDelta()
        element_size += 1
        feature_list.append(td)
    ngram = Ngram(
        feature_list=feature_list,
        thread_aware=thread_aware,
        ngram_length=ngram_length + 1
    )
    ngram_minus_one = NgramMinusOne(
        ngram=ngram,
        element_size=element_size
    )
    int_embedding = IntEmbedding()
    if use_thread_change_flag:
        tcf = ThreadChangeFlag(ngram_minus_one)
    feature_list = [int_embedding,
                    ngram_minus_one]
    if use_thread_change_flag:
        feature_list.append(tcf)

    # decision engine (DE)
    distinct_syscalls = dataloader.distinct_syscalls_training_data()
    lstm = LSTM(element_size=element_size,
                use_thread_change_flag=use_thread_change_flag,
                ngram_length=ngram_length,
                distinct_syscalls=distinct_syscalls,
                epochs=20,
                batch_size=256,
                force_train=True,
                model_path=f'Models/{scenario}/LSTM/')

    # define the used features
    ids = IDS(data_loader=dataloader,
              feature_list=[int_embedding,
                            ngram_minus_one],
              decision_engine=lstm,
              plot_switch=True)

    # training
    ids.train_decision_engine()
    # threshold
    ids.determine_threshold()
    # detection
    ids.do_detection()
    # print(results)
    pprint(ids.performance.get_performance())
