import sys
import math
from pprint import pprint
from algorithms.decision_engines.ae import AE
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.random_value import RandomValue
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.sum import Sum

from algorithms.ids import IDS
from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction

if __name__ == '__main__':

    # getting the LID-DS base path, version and scenario from argument
    try:        
        lid_ds_base_path = sys.argv[1]
        lid_ds_version = sys.argv[2]
        scenario_name = sys.argv[3]
    except:
        print(f"Error, call with:\n> python3 {sys.argv[0]} lid_ds_base_path lid_ds_version scenario_name")
        exit()        

    scenario_path = f"{lid_ds_base_path}/{lid_ds_version}/{scenario_name}"        
    dataloader = dataloader_factory(scenario_path,direction=Direction.CLOSE)

    ### features
    thread_aware = True
    ngram_length = 7
    enc_size = 10    

    ### building blocks  
    name = SyscallName()
    inte = IntEmbedding(name)
    random_numbers = RandomValue(size=enc_size,scale=0.01)
    w2v = W2VEmbedding(word=inte,vector_size=enc_size,window_size=20,epochs=5000)
    sum = Sum([w2v, random_numbers])
    ngram = Ngram(feature_list = [sum],thread_aware = thread_aware,ngram_length = ngram_length)
    ae = AE(ngram, batch_size=256,max_training_time=120,early_stopping_epochs=10000)
    #stream_window = StreamSum(ae,False,100)

    ### the IDS    
    ids = IDS(data_loader=dataloader,
            resulting_building_block=ae,
            create_alarms=False,
            plot_switch=False)

    # after training - dont add random numbers
    sum._dependency_list = [w2v]

    print("at evaluation:")
    # threshold
    ids.determine_threshold()    
    # detection
    # results = ids.detect().get_results()
    results = ids.detect_parallel().get_results()

    pprint(results)
