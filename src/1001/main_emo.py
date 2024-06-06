import os

# evolver-reg, evolver-fisher, evolver, fisher, regmean, simple
# deberta-large, roberta-base, distilbert
model = "roberta-base"
method, method2 = "evolver", "evolver-reg"
# exp_lst = ['f_0.01_cr_0.5', 'f_0.1_cr_0.5', 'f_0.2_cr_0.5', 'f_0.3_cr_0.5', 'f_0.4_cr_0.5', 
#            'f_0.5_cr_0.5', 'f_0.6_cr_0.5', 'f_0.7_cr_0.5', 'f_0.8_cr_0.5', 'f_0.9_cr_0.5']     
exp_lst = ['f_0.05_cr_0.5', 'f_0.15_cr_0.5', 'f_0.5_cr_0.01', 'f_0.5_cr_0.05', 'f_0.5_cr_0.1', 
           'f_0.5_cr_0.2', 'f_0.5_cr_0.3', 'f_0.5_cr_0.4', 'f_0.5_cr_0.6', 'f_0.5_cr_0.7', 'f_0.5_cr_0.8',
           'f_0.5_cr_0.9', 'f_0.5_cr_0.15',]     
# ************************** Emotion ********************************
orders = ["model2", "model5", "model1", "model3", "model0"]
# 1, 2, 3, 4, 5
for exp in exp_lst:
    for idx in range(5, 6): #2, len(orders) + 1
        to_merge = " ".join(orders[:idx])
        os.system(f" \
            CUDA_VISIBLE_DEVICES=0, \
            python -m run_experiments --config_file configs/defaults.yaml \
                                        configs/datasets/emotion.yaml \
                                        configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                    --filter_model {to_merge} --templates seed={1} exp_name={exp} & \
            CUDA_VISIBLE_DEVICES=1, \
            python -m run_experiments --config_file configs/defaults.yaml \
                                        configs/datasets/emotion.yaml \
                                        configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                    --filter_model {to_merge} --templates seed={2} exp_name={exp} & \
            CUDA_VISIBLE_DEVICES=2, \
            python -m run_experiments --config_file configs/defaults.yaml \
                                        configs/datasets/emotion.yaml \
                                        configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                    --filter_model {to_merge} --templates seed={3} exp_name={exp} & \
            CUDA_VISIBLE_DEVICES=3, \
            python -m run_experiments --config_file configs/defaults.yaml \
                                        configs/datasets/emotion.yaml \
                                        configs/exps/{model}/emotion/{model}-emotion-{method}.yaml \
                                    --filter_model {to_merge} --templates seed={4} exp_name={exp} & \
            CUDA_VISIBLE_DEVICES=4, \
            python -m run_experiments --config_file configs/defaults.yaml \
                                        configs/datasets/emotion.yaml \
                                        configs/exps/{model}/emotion/{model}-emotion-{method2}.yaml \
                                    --filter_model {to_merge} --templates seed={1} exp_name={exp} & \
            CUDA_VISIBLE_DEVICES=5, \
            python -m run_experiments --config_file configs/defaults.yaml \
                                        configs/datasets/emotion.yaml \
                                        configs/exps/{model}/emotion/{model}-emotion-{method2}.yaml \
                                    --filter_model {to_merge} --templates seed={2} exp_name={exp} & \
            CUDA_VISIBLE_DEVICES=6, \
            python -m run_experiments --config_file configs/defaults.yaml \
                                        configs/datasets/emotion.yaml \
                                        configs/exps/{model}/emotion/{model}-emotion-{method2}.yaml \
                                    --filter_model {to_merge} --templates seed={3} exp_name={exp} & \
            CUDA_VISIBLE_DEVICES=7, \
            python -m run_experiments --config_file configs/defaults.yaml \
                                        configs/datasets/emotion.yaml \
                                        configs/exps/{model}/emotion/{model}-emotion-{method2}.yaml \
                                    --filter_model {to_merge} --templates seed={4} exp_name={exp} \
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
