import os

if __name__ == '__main__':
    # parameter lists
    # enc_sizes = range(2, 18, 2)  # [2, 4, ..., 16]
    enc_sizes = [2, 6, 10, 14]
    w2v_epochs = [50, 500, 5000]
    n_gram_sizes = [3, 5, 7, 9]
    stream_sum_sizes = [1] + list(range(200, 1200, 200))  # [1, 200, ..., 1000]
    scenarios = range(10)  # [0, 2, ..., 9]
    algorithms = ['som', 'ae', 'mlp']

    i = 1
    for enc_size in enc_sizes:
        for epochs in w2v_epochs:
            for n in n_gram_sizes:
                for stream_size in stream_sum_sizes:
                    for scenario in scenarios:
                        for algorithm in algorithms:
                            os.system(f'sbatch batch_experiment.job {enc_size} {epochs} {n} {stream_size} {scenario} {algorithm}')
