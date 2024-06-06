import os
import numpy

# pair_task_lst = []
# for i in range(8):
#     pair_task = numpy.random.choice(range(0, 8), size=2, replace=False, p=None).tolist()
#     pair_task_lst.append(pair_task)
# print("pair_task_lst:", pair_task_lst)
t_lst = [[3, 5], [2, 3], [5, 4], [1, 6], [6, 3], [0, 2], [4, 0], [7, 6]]
model = "roberta-base"
method = "fisher"   # "simple", "fisher", "regmean", "evolver", "evolver_fisher", "evolver_reg"

os.system(f" \
    CUDA_VISIBLE_DEVICES=0, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/glue.yaml \
                                    configs/exps/{model}/glue/{model}-{method}.yaml \
                    --filter_model model{t_lst[0][0]} model{t_lst[0][1]} --templates seed={1} & \
    CUDA_VISIBLE_DEVICES=1, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/glue.yaml \
                                    configs/exps/{model}/glue/{model}-{method}.yaml \
                    --filter_model model{t_lst[1][0]} model{t_lst[1][1]} --templates seed={2} & \
    CUDA_VISIBLE_DEVICES=2, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/glue.yaml \
                                    configs/exps/{model}/glue/{model}-{method}.yaml \
                    --filter_model model{t_lst[2][0]} model{t_lst[2][1]} --templates seed={3} & \
    CUDA_VISIBLE_DEVICES=3, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/glue.yaml \
                                    configs/exps/{model}/glue/{model}-{method}.yaml \
                    --filter_model model{t_lst[3][0]} model{t_lst[3][1]} --templates seed={4} & \
    CUDA_VISIBLE_DEVICES=4, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/glue.yaml \
                                    configs/exps/{model}/glue/{model}-{method}.yaml \
                    --filter_model model{t_lst[4][0]} model{t_lst[4][1]} --templates seed={5} & \
    CUDA_VISIBLE_DEVICES=5, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/glue.yaml \
                                    configs/exps/{model}/glue/{model}-{method}.yaml \
                    --filter_model model{t_lst[5][0]} model{t_lst[5][1]} --templates seed={6} & \
    CUDA_VISIBLE_DEVICES=6, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/glue.yaml \
                                    configs/exps/{model}/glue/{model}-{method}.yaml \
                    --filter_model model{t_lst[6][0]} model{t_lst[6][1]} --templates seed={7} & \
    CUDA_VISIBLE_DEVICES=7, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/glue.yaml \
                                    configs/exps/{model}/glue/{model}-{method}.yaml \
                    --filter_model model{t_lst[7][0]} model{t_lst[7][1]} --templates seed={8} & \
        ")

# # ************************** Emotion ********************************
