import os

# evolver-reg, evolver-fisher, evolver, fisher, regmean, simple
# deberta-large, roberta-base, distilbert
model = "distilbert"
method = "fisher"
# ************************** Emotion ********************************
# orders = ["model2", "model5", "model1", "model3", "model0"] 
# model4:  # this dataset uses a different label space
lst = [[0,1],[0,2],[0,3],[0,5],[1,2],[1,3],[1,5],[2,3],[2,5],[3,5]]
for idx1, idx2 in lst[:11]:
    os.system(f" \
        CUDA_VISIBLE_DEVICES=3, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                    --filter_model model{idx1} model{idx2} \
                                    --templates seed={1} exp_name={idx1}+{idx2} & \
        CUDA_VISIBLE_DEVICES=4, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                    --filter_model model{idx1} model{idx2} \
                                    --templates seed={2} exp_name={idx1}+{idx2} & \
        CUDA_VISIBLE_DEVICES=5, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                    --filter_model model{idx1} model{idx2} \
                                    --templates seed={3}  exp_name={idx1}+{idx2} & \
        CUDA_VISIBLE_DEVICES=6, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                    --filter_model model{idx1} model{idx2} \
                                    --templates seed={4}  exp_name={idx1}+{idx2} & \
        CUDA_VISIBLE_DEVICES=7, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                    configs/datasets/emotion.yaml \
                                    configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                    --filter_model model{idx1} model{idx2} \
                                    --templates seed={5}  exp_name={idx1}+{idx2} \
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
