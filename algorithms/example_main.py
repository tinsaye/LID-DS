"""
Example execution of LIDS Framework
"""
import os
import sys
from pprint import pprint

from dataloader.dataloader_factory import dataloader_factory

from dataloader.direction import Direction

from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.decision_engines.stide import Stide
from algorithms.features.impl.ngram import Ngram
from algorithms.ids import IDS


if __name__ == '__main__':

    # getting the LID-DS base path from argument or environment variable
    if len(sys.argv) > 1:
        LID_DS_BASE_PATH = sys.argv[1]
    else:
        try:
            LID_DS_BASE_PATH = os.environ['LID_DS_BASE']
        except KeyError as exc:
            raise ValueError("No LID-DS Base Path given."
                             "Please specify as argument or set Environment Variable "
                             "$LID_DS_BASE") from exc

    LID_DS_VERSION = "LID-DS-2019"
    SCENARIO_NAME = "CVE-2017-7529"
    #scenario_name = "CVE-2014-0160"
    #scenario_name = "Bruteforce_CWE-307"
    #scenario_name = "CVE-2012-2122"

    scenario_path = f"{LID_DS_BASE_PATH}/{LID_DS_VERSION}/{SCENARIO_NAME}"
    # just load < closing system calls for this example
    dataloader = dataloader_factory(scenario_path,direction=Direction.BOTH)

    ### features (for more information see Paper:
    # https://dbs.uni-leipzig.de/file/EISA2021_Improving%20Host-based%20Intrusion%20Detection%20Using%20Thread%20Information.pdf
    THREAD_AWARE = True
    WINDOW_LENGTH = 1000
    NGRAM_LENGTH = 5

    ### building blocks
    # first: map each systemcall to an integer
    int_embedding = IntEmbedding()
    # now build ngrams from these integers
    ngram = Ngram([int_embedding], THREAD_AWARE, NGRAM_LENGTH)
    # finally calculate the STIDE algorithm using these ngrams
    stide = Stide(ngram)
    # build stream sum of stide results
    stream_sum = StreamSum(stide, False, WINDOW_LENGTH, False)
    ### the IDS
    ids = IDS(data_loader=dataloader,
              resulting_building_block=stream_sum,
              create_alarms=True,
              plot_switch=False)

    print("at evaluation:")
    # threshold
    ids.determine_threshold()

    # detection
    # normal / seriell
    # results = ids.detect().get_results()
    # parallel / map-reduce
    results = ids.detect_parallel().get_results()

    # to get alarms:
    # print(performance.alarms.alarm_list)

    ### print results
    pprint(results)
