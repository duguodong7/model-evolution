import os

# evolver-reg, evolver-fisher, evolver, fisher, regmean, simple
# deberta-large, roberta-base, distilbert
model = "distilbert"
method = "regmean"
# ************************** Emotion ********************************
orders = ["model2", "model5", "model1", "model3", "model0"]
# 1, 2, 3, 4, 5
for idx in range(5, 6): #2, len(orders) + 1
    to_merge = " ".join(orders[:idx])
    os.system(f" \
        CUDA_VISIBLE_DEVICES=3, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                --filter_model {to_merge} --templates seed={1} & \
        CUDA_VISIBLE_DEVICES=4, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                --filter_model {to_merge} --templates seed={2} & \
        CUDA_VISIBLE_DEVICES=5, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                --filter_model {to_merge} --templates seed={3} & \
        CUDA_VISIBLE_DEVICES=6, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                --filter_model {to_merge} --templates seed={4} & \
        CUDA_VISIBLE_DEVICES=7, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                --filter_model {to_merge} --templates seed={5} & \
                ")
    # os.system(f" \
    #     CUDA_VISIBLE_DEVICES=0, \
    #     python -m run_experiments --config_file configs/defaults.yaml \
    #                                 configs/datasets/emotion.yaml \
    #                                 configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
    #                             --filter_model {to_merge} --templates seed={4} & \
    #     CUDA_VISIBLE_DEVICES=1, \
    #     python -m run_experiments --config_file configs/defaults.yaml \
    #                                 configs/datasets/emotion.yaml \
    #                                 configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
    #                             --filter_model {to_merge} --templates seed={5} & \
    #             ")
        # CUDA_VISIBLE_DEVICES=2, \
        # python -m run_experiments --config_file configs/defaults.yaml \
        #                             configs/datasets/emotion.yaml \
        #                             configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
        #                         --filter_model {to_merge} --templates seed={3} & \
