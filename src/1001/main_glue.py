import os

# # ************************** Emotion ********************************
# orders = ["model2", "model5", "model1", "model3", "model0"]

# for seed in [1, ]:   #1, 2, 3, 4, 5
#     for idx in range(5, 6): #2, len(orders) + 1
#         # simple
#         to_merge = " ".join(orders[:idx])
#         os.system(
#             f"CUDA_VISIBLE_DEVICES=1, \
#             python -m run_experiments --config_file configs/defaults.yaml \
#                                         configs/datasets/emotion.yaml \
#                                         configs/exps/roberta-base/emotion/roberta-base-emotion-fisher.yaml \
#                                     --filter_model {to_merge} --templates seed={seed}"
#         )

# ************************** GLUE ********************************
# for seed in [1, 2, 3, 4, 5, 6, 7, 8]:  #1, 2, 3, 4, 5, 6, 7, 8
model = "roberta-base"
method = "evolver"   # "simple", "fisher", "regmean", "evolver", "evolver_fisher", "evolver_reg"
for task in ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte"]:  #"cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte" 
    os.system(f" \
    CUDA_VISIBLE_DEVICES=0, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/subsets/glue_partition_1k_niid.yaml \
                                configs/exps/{model}/subset/rb-1k-{method}-whead.yaml \
                                --templates seed={1} dataset_name={task} & \
    CUDA_VISIBLE_DEVICES=1, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/subsets/glue_partition_1k_niid.yaml \
                                configs/exps/{model}/subset/rb-1k-{method}-whead.yaml \
                                --templates seed={2} dataset_name={task} & \
    CUDA_VISIBLE_DEVICES=2, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/subsets/glue_partition_1k_niid.yaml \
                                configs/exps/{model}/subset/rb-1k-{method}-whead.yaml \
                                --templates seed={3} dataset_name={task} & \
    CUDA_VISIBLE_DEVICES=3, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/subsets/glue_partition_1k_niid.yaml \
                                configs/exps/{model}/subset/rb-1k-{method}-whead.yaml \
                                --templates seed={4} dataset_name={task} & \
    CUDA_VISIBLE_DEVICES=4, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/subsets/glue_partition_1k_niid.yaml \
                                configs/exps/{model}/subset/rb-1k-{method}-whead.yaml \
                                --templates seed={5} dataset_name={task} & \
    CUDA_VISIBLE_DEVICES=5, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/subsets/glue_partition_1k_niid.yaml \
                                configs/exps/{model}/subset/rb-1k-{method}-whead.yaml \
                                --templates seed={6} dataset_name={task} & \
    CUDA_VISIBLE_DEVICES=6, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/subsets/glue_partition_1k_niid.yaml \
                                configs/exps/{model}/subset/rb-1k-{method}-whead.yaml \
                                --templates seed={7} dataset_name={task} & \
    CUDA_VISIBLE_DEVICES=7, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/subsets/glue_partition_1k_niid.yaml \
                                configs/exps/{model}/subset/rb-1k-{method}-whead.yaml \
                                --templates seed={8} dataset_name={task}  \
    ")

