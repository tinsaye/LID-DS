#!/bin/bash
#SBATCH --time=200:00:00
#SBATCH --partition=clara-job
#SBATCH --gres=gpu:rtx2080ti:2
#SBATCH --mem=150G
#SBATCH --nodes=1

# source /scratch/ws/1/tikl664d-master/master/bin/activate
# module load Python
# module load Python/3.7.4-GCCcore-8.3.0
module load PyTorch/1.9.0-fosscuda-2020b
# module load PyTorch/1.7.1-fosscuda-2019b-Python-3.7.4
# module load CUDA/11.3.1
# module load CUDA/10.1.243
# module load CUDA/10.1.243-GCC-8.3.0
# module load CUDA/10.1.243-iccifort-2019.5.281
# module load Python/3.9.5-GCCcore-10.3.0
# module load PyTorch/1.8.1-fosscuda-2020b
# module load PyTorch/1.7.1-fosscuda-2019b-Python-3.7.4             
# module load PyTorch/1.8.1-foss-2020b
# module load PyTorch/1.8.1-fosscuda-2019b-Python-3.7.4             
# PyTorch/1.9.0-fosscuda-2020b

module load matplotlib
pip install --upgrade pip
pip install --user -e ../
pip install --user nest-asyncio
pip install --user pypcapkit
pip install --user networkx
pip install --user pydot
pip install --user gensim==4.1.2
pip install --user matplotlib


# parameters:
# 1: base_path
# 2: scenario_name
# 3: batch_size
# 4: epochs
# 5: embedding_size
# 6: ngram_length
# 7: time_delta
# 8: thread_change_flag
# 9: return_value

time_delta_flag=" -td"
thread_change_flag=" -tcf"
return_value_flag=" -rv"
flags=""
if [[ $7 == "True" ]]; then
    flags="$flags$time_delta_flag"
fi
if [[ $8 == "True" ]]; then
    flags="$flags$thread_change_flag"
fi
if [[ $9 == "True" ]]; then
    flags="$flags$return_value_flag"
fi
echo $flags
python lstm_cluster_main.py -d $1 -s $2 -b $3 -ep $4 -e $5 -n $6 $flags
# srun python test.py
