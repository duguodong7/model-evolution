import os

# evolver-reg, evolver-fisher, evolver, fisher, regmean, simple
# deberta-large, roberta-base, distilbert
model = "roberta-base"
method = "fisher"
# ************************** Emotion ********************************
orders = ["model2", "model5", "model1", "model3", "model0"] 
to_merge = " ".join(orders[:5])
exp_name = "debug1"
# model4:  # this dataset uses a different label space
os.system(f" \
    CUDA_VISIBLE_DEVICES=3, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/emotion.yaml \
                                configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                --filter_model {to_merge} \
                                --templates seed={1} exp_name={exp_name} & \
    CUDA_VISIBLE_DEVICES=4, \
    python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/emotion.yaml \
                                configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                --filter_model {to_merge} \
                                --templates seed={2} exp_name={exp_name} & \
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
