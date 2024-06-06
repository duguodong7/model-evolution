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
for seed in [1, 2, 3, 4, 5]:  #1, 2, 3, 4, 5
    for task in ["mrpc", "stsb", "mnli", "qnli", "qqp", "rte"]:  #"cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte" 
        os.system(
            f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-whead.yaml --templates seed={seed} dataset_name={task}"
        )
        # # fisher
        os.system(
            f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-fisher-whead.yaml --templates seed={seed} dataset_name={task}"
        )
        # regmean
        os.system(
            f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-regmean-whead.yaml --templates seed={seed} dataset_name={task}"
        )
        # evolver
        os.system(
            f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-evolver-whead.yaml --templates seed={seed} dataset_name={task}"
        )

