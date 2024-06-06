import os

# ************************** Emotion Pretrain ********************************
# orders = ["model2", "model5", "model1", "model3", "model0"]
# to_merge = " ".join(orders[:5])
# seed = 1
# os.system(
#     f"CUDA_VISIBLE_DEVICES=3, \
#         python -m run_experiments --config_file configs/defaults.yaml \
#         configs/datasets/emotion.yaml  \
#         configs/exps/roberta-base/emotion/roberta-base-emotion_train_model.yaml \
#             --filter_model {to_merge} --templates seed={seed}"
# )

# ************************** Emotion Evolver ********************************

# ************************** Emotion ********************************
orders = ["model2", "model5", "model1", "model3", "model0"]

for seed in [1, 2, 3, 4, 5]:   #1, 2, 3, 4, 5
    for idx in range(5, 6): #2, len(orders) + 1
        # simple
        to_merge = " ".join(orders[:idx])
        os.system(
            f"CUDA_VISIBLE_DEVICES=2, \
            python -m run_experiments --config_file configs/defaults.yaml \
                                        configs/datasets/emotion.yaml \
                                        configs/exps/roberta-base/emotion/roberta-base-emotion.yaml \
                                    --filter_model {to_merge} --templates seed={seed}"
        )
# orders = ["model2", "model5", "model1", "model3", "model0"]
# # 1, 2, 3, 4, 5
# for idx in range(5, 6): #2, len(orders) + 1
#     # simple
#     to_merge = " ".join(orders[:idx])
#     os.system(f" \
#         CUDA_VISIBLE_DEVICES=4, \
#         python -m run_experiments --config_file configs/defaults.yaml \
#                                     configs/datasets/emotion.yaml \
#                                     configs/exps/roberta-base/emotion/roberta-base-emotion-evolver_reg.yaml \
#                                 --filter_model {to_merge} --templates seed={1}  \
#         & CUDA_VISIBLE_DEVICES=5, \
#         python -m run_experiments --config_file configs/defaults.yaml \
#                                     configs/datasets/emotion.yaml \
#                                     configs/exps/roberta-base/emotion/roberta-base-emotion-evolver_reg.yaml \
#                                 --filter_model {to_merge} --templates seed={2}  \
#         & CUDA_VISIBLE_DEVICES=6, \
#         python -m run_experiments --config_file configs/defaults.yaml \
#                                     configs/datasets/emotion.yaml \
#                                     configs/exps/roberta-base/emotion/roberta-base-emotion-evolver_reg.yaml \
#                                 --filter_model {to_merge} --templates seed={3}  \
#         & CUDA_VISIBLE_DEVICES=7, \
#         python -m run_experiments --config_file configs/defaults.yaml \
#                                     configs/datasets/emotion.yaml \
#                                     configs/exps/roberta-base/emotion/roberta-base-emotion-evolver_reg.yaml \
#                                 --filter_model {to_merge} --templates seed={4} & ")


        # # fisher
        # os.system(
        #     f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/emotion.yaml configs/exps/roberta-base/emotion/roberta-base-emotion-fisher.yaml --filter_model {to_merge} --templates seed={seed}"
        # )
        # # regmean
        # os.system(
        #     f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/emotion.yaml configs/exps/roberta-base/emotion/roberta-base-emotion-regmean.yaml --filter_model {to_merge} --templates seed={seed}"
        # )


# ************************** GLUE ********************************
# for seed in [1, 2, 3, 4, 5]:  #1, 2, 3, 4, 5
#     for task in ["mrpc", "stsb", "mnli", "qnli", "qqp", "rte"]:  #"cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte" 
#         os.system(
#             f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-whead.yaml --templates seed={seed} dataset_name={task}"
#         )
#         # # fisher
#         os.system(
#             f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-fisher-whead.yaml --templates seed={seed} dataset_name={task}"
#         )
#         # regmean
#         os.system(
#             f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-regmean-whead.yaml --templates seed={seed} dataset_name={task}"
#         )
#         # evolver
#         os.system(
#             f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-evolver-whead.yaml --templates seed={seed} dataset_name={task}"
#         )


# ************************** Emotion ********************************
# same classification head init

# for seed in [1, 2, 3, 4, 5]:
#     for idx1 in range(0, 6):
#         for idx2 in range(idx1 + 1, 6):
#             if idx1 != 4 and idx2 != 4:  # this dataset uses a different label space
#                 # simple
#                 os.system(
#                     f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/roberta-base/roberta-base-emotion.yaml --filter_model model{idx1} model{idx2} --templates seed={seed}"
#                 )
#                 # fisher
#                 os.system(
#                     f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/roberta-base/roberta-base-emotion-fisher.yaml --filter_model model{idx1} model{idx2} --templates seed={seed}"
#                 )
#                 # regmean
#                 os.system(
#                     f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/roberta-base/roberta-base-emotion-regmean.yaml --filter_model model{idx1} model{idx2} --templates seed={seed}"
#                 )

# different classification head init

# for seed in [1, 2, 3, 4, 5]:
#     for idx1 in range(0, 6):
#         for idx2 in range(idx1 + 1, 6):
#             if idx1 != 4 and idx2 != 4:  # this dataset uses a different label space
#                 # simple
#                 os.system(
#                     f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_diffseed.yaml src/configs/exps/roberta-base/roberta-base-emotion.yaml --filter_model model{idx1} model{idx2} --templates dseed_generator={seed} seed={seed}"
#                 )
#                 # fisher
#                 os.system(
#                     f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_diffseed.yaml src/configs/exps/roberta-base/roberta-base-emotion-fisher.yaml --filter_model model{idx1} model{idx2} --templates dseed_generator={seed} seed={seed}"
#                 )
#                 # regmean
#                 os.system(
#                     f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_diffseed.yaml src/configs/exps/roberta-base/roberta-base-emotion-regmean.yaml --filter_model model{idx1} model{idx2} --templates dseed_generator={seed} seed={seed}"
#                 )

  
