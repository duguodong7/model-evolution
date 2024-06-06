import os

# same classification head init, model merging

orders = ["model2", "model5", "model1", "model3", "model0"]
to_merge = " ".join(orders[:2])
os.system(f"python -m src.run_experiments --config_file src/configs/defaults.yaml \
          src/configs/datasets/emotion.yaml src/configs/exps/roberta-base/ood/roberta-regmean-emotion-ood.yaml \
            --filter_model {to_merge} --templates seed={1}")
