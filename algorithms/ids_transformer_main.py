from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.decision_engines.transformer import TransformerDE
from dataloader.dataloader_factory import dataloader_factory
from algorithms.features.impl.ngram import Ngram
from algorithms.ids import IDS

if __name__ == '__main__':
    """
    this is an example script to show the usage uf our classes
    """
    ngram_length = 6
    thread_aware = True
    scenario = "CVE-2017-7529"
    scenario_path = f'../../Dataset/{scenario}/'

    # data loader for scenario
    dataloader = dataloader_factory(scenario_path)

    # embedding
    int_embedding = IntEmbedding()

    # ngram
    ngram = Ngram(feature_list=[int_embedding],
                  thread_aware=True,
                  ngram_length=ngram_length)

    distinct_syscalls = dataloader.distinct_syscalls_training_data()
    # decision engine (DE)
    transformer = TransformerDE(input_vector=ngram,
                                distinct_syscalls=distinct_syscalls,
                                epochs=20,
                                batch_size=2,
                                force_train=True,
                                model_path=f'Models/Transformer/{scenario}/')

    # define the used features and train
    ids = IDS(data_loader=dataloader,
              resulting_building_block=transformer,
              plot_switch=True)

    # threshold
    ids.determine_threshold()
    # detection
    # ids.do_detection()
    # pprint(ids.performance.get_performance())

    # ids.plot.feed_figure()
    # ids.plot.show_plot()
