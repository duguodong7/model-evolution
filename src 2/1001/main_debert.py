import os

# ************************** Emotion Pretrain ********************************
# orders = ["model2", "model5", "model1", "model3", "model0"]
orders = ["model3", "model5", ]
to_merge = " ".join(orders[:5])
os.system(f" \
            CUDA_VISIBLE_DEVICES=0, \
            python -m run_experiments --config_file configs/defaults.yaml \
                configs/datasets/emotion.yaml \
                configs/exps/deberta/deberta-large-emotion-regmean.yaml \
                    --filter_model {to_merge} --templates seed={5} & \       ")
            # CUDA_VISIBLE_DEVICES=5, \
            # python -m run_experiments --config_file configs/defaults.yaml \
            #     configs/datasets/emotion.yaml \
            #     configs/exps/deberta/deberta-large-emotion-regmean.yaml \
            #         --filter_model {to_merge} --templates seed={2} & \
            # CUDA_VISIBLE_DEVICES=6, \
            # python -m run_experiments --config_file configs/defaults.yaml \
            #     configs/datasets/emotion.yaml \
            #     configs/exps/deberta/deberta-large-emotion-regmean.yaml \
            #         --filter_model {to_merge} --templates seed={3} & \
            # CUDA_VISIBLE_DEVICES=7, \
            #     python -m run_experiments --config_file configs/defaults.yaml \
            #     configs/datasets/emotion.yaml \
            #     configs/exps/deberta/deberta-large-emotion-regmean.yaml \
            #         --filter_model {to_merge} --templates seed={4} & \
                 

# ************************** Emotion Evolver ********************************

# ************************** Emotion ********************************
