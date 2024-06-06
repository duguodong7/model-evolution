import os
# import sys 
# print(sys.path) #查看 
# os.system(f"cd /home/guodong/nlp/dataless-model-merging_new/src")
# sys.path.append('/home/guodong/nlp/dataless-model-merging_new/src')
# ************************** GLUE debug ********************************
# for seed in [3]:  #, 2, 3, 4, 5
#     for task in ["cola"]:  #"cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte" 
#         # fisher
#         os.system(
#             f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-fisher-whead.yaml --templates seed={seed} dataset_name={task}"
#         )
#         # # regmean
#         # os.system(
#         #     f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-regmean-whead.yaml --templates seed={seed} dataset_name={task}"
#         # )

# ************************** GLUE ********************************
# for seed in [2]:  #, 2, 3, 4, 5
#     for task in ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte"]:  #"cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte" 
#         os.system(
#             f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-whead.yaml --templates seed={seed} dataset_name={task}"
#         )
#         # fisher
#         os.system(
#             f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-fisher-whead.yaml --templates seed={seed} dataset_name={task}"
#         )
#         # regmean
#         os.system(
#             f"python -m run_experiments --config_file configs/defaults.yaml configs/datasets/subsets/glue_partition_1k_niid.yaml configs/exps/roberta-base/subset/rb-1k-regmean-whead.yaml --templates seed={seed} dataset_name={task}"
#         )


# ************************** emotion2 ********************************
orders = ["model2", "model5", "model1", "model3", "model0"]
seed = 1 
idx = 2
# simple
to_merge = " ".join(orders[:idx])
os.system(
    f"python -m run_experiments --config_file configs/defaults.yaml \
                                                  configs/datasets/emotion.yaml \
                                                  configs/exps/roberta-base/ood/roberta-base-emotion-ood.yaml \
                                    --filter_model {to_merge} --templates seed={seed}"
)
# ************************** ner ********************************
# os.system(f"python run_experiments.py --config_file configs/defaults.yaml configs/datasets/ner.yaml \
#     configs/exps/roberta-base/ner/ood/roberta-base-ner.yaml \
#           --filter_model model0 model5 --templates seed=2")

# ************************** emotion1 ********************************
# same classification head init, model merging

# orders = ["model2", "model5", "model1", "model3", "model0"]
# to_merge = " ".join(orders[:2])
# os.system(f"python -m run_experiments --config_file configs/defaults.yaml \
#           configs/datasets/emotion.yaml configs/exps/roberta-base/ood/roberta-base-emotion-ood.yaml \
#             --filter_model {to_merge} --templates seed={1}")
