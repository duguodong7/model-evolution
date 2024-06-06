import os

# ************************** Emotion ********************************
orders = ["model2", "model5", "model1", "model3", "model0"]
# 1, 2, 3, 4, 5
for idx in range(5, 6): #2, len(orders) + 1
    to_merge = " ".join(orders[:idx])
    os.system(f" \
        CUDA_VISIBLE_DEVICES=4, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/emotion_diffseed.yaml \
                                configs/exps/roberta-base/emotion/roberta-base-emotion-regmean-woclc.yaml \
                            --filter_model {to_merge} --templates dseed_generator={1} seed={1} & \
        CUDA_VISIBLE_DEVICES=5, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/emotion_diffseed.yaml \
                                configs/exps/roberta-base/emotion/roberta-base-emotion-regmean-woclc.yaml \
                            --filter_model {to_merge} --templates dseed_generator={2} seed={2} & \
        CUDA_VISIBLE_DEVICES=6, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/emotion_diffseed.yaml \
                                configs/exps/roberta-base/emotion/roberta-base-emotion-regmean-woclc.yaml \
                            --filter_model {to_merge} --templates dseed_generator={3} seed={3} & \
        CUDA_VISIBLE_DEVICES=7, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/emotion_diffseed.yaml \
                                configs/exps/roberta-base/emotion/roberta-base-emotion-regmean-woclc.yaml \
                            --filter_model {to_merge} --templates dseed_generator={4} seed={4} & \
        CUDA_VISIBLE_DEVICES=3, \
        python -m run_experiments --config_file configs/defaults.yaml \
                                configs/datasets/emotion_diffseed.yaml \
                                configs/exps/roberta-base/emotion/roberta-base-emotion-regmean-woclc.yaml \
                            --filter_model {to_merge} --templates dseed_generator={5} seed={5} & \
                ")

# evolver-reg-woclc,
# evolver-reg, evolver-fisher, evolver, fisher, regmean, simple

# deberta-large, roberta-base