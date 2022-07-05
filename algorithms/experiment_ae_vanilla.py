import math
from pprint import pprint
from algorithms.decision_engines.ae import AE
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.position_in_file import PositionInFile
from algorithms.features.impl.sinusoidal_encoding import SinusoidalEncoding
from algorithms.features.impl.random_value import RandomValue
from algorithms.features.impl.return_value import ReturnValue
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.sum import Sum

from algorithms.ids import IDS
from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction

if __name__ == '__main__':

    # todo: change this to your base path
    lid_ds_base_path = "/home/grimmer/data"
    lid_ds_version = "LID-DS-2021"
    scenario_name = "CVE-2017-7529"
    # scenario_name = "CVE-2014-0160"
    # scenario_name = "Bruteforce_CWE-307"
    scenario_path = f"{lid_ds_base_path}/{lid_ds_version}/{scenario_name}"        
    dataloader = dataloader_factory(scenario_path,direction=Direction.CLOSE)

    ### features
    thread_aware = True
    ngram_length = 7
    enc_size = 10
    ae_hidden_size = int(math.sqrt(ngram_length * enc_size))

    ### building blocks  
    name = SyscallName()
    inte = IntEmbedding(name)
    w2v = W2VEmbedding(word=inte,vector_size=enc_size,window_size=10,epochs=1000)
    ngram = Ngram(feature_list = [w2v],thread_aware = thread_aware,ngram_length = ngram_length)
    ae = AE(ngram,ae_hidden_size,batch_size=256,max_training_time=120)
    stream_window = StreamSum(ae,True,600)

    ### the IDS    
    ids = IDS(data_loader=dataloader,
            resulting_building_block=stream_window,
            create_alarms=False,
            plot_switch=False)

    print("at evaluation:")
    # threshold
    ids.determine_threshold()
    # detection
    results = ids.detect().get_results()

    pprint(results)
