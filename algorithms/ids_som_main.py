from pprint import pprint

from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.path_evilness import PathEvilness
from algorithms.features.impl.w2v_embedding import W2VEmbedding

from algorithms.decision_engines.som import Som
from algorithms.ids import IDS
from dataloader.data_preprocessor import DataPreprocessor
from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction

if __name__ == '__main__':
    # data loader for scenario
    dataloader = dataloader_factory('/home/felix/repos/LID-DS/LID-DS-2019/CVE-2017-7529', direction=Direction.OPEN)

    w2v = W2VEmbedding(vector_size=5,
                       epochs=200,
                       path='Models',
                       force_train=False,
                       distinct=True,
                       window_size=7,
                       thread_aware=True,
                       scenario_path=dataloader.scenario_path)

    # inte = IntEmbedding()

    ngram = Ngram(
        feature_list=[w2v],
        thread_aware=True,
        ngram_length=7
    )


    # pe = PathEvilness(scenario_path=dataloader.scenario_path)


    DE = Som(
        epochs=100
    )

    # define the used features

    ids = IDS(data_loader=dataloader,
              feature_list=[ngram],
              decision_engine=DE,
              plot_switch=True)

    ids.train_decision_engine()
    ids.determine_threshold()
    ids.do_detection()

    pprint(ids.performance.get_performance())

    ids.draw_plot()
