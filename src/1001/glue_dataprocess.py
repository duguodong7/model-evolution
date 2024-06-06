# hhh
from datasets import load_dataset
from torch.utils.data import Subset
import os


# Define a list of GLUE task names you want to download.
glue_tasks = [
    'cola',
    'sst2',
    'mrpc',
    'qqp',
    'stsb',
    'mnli',
    'mnli_mismatched',
    'mnli_matched',
    'qnli',
    'rte',
    'wnli',
    'ax'
]
# default_cache_dir = '~/.cache/huggingface/datasets' 
# my_cache_dir = '/home/guodong/nlp/evonlp/src/resources/huggingface/datasets'
my_cache_dir = '/home/guodong/nlp/evonlp/src/resources/huggingface/datasets'

# task_name = 'sst2'
# task_dataset = load_dataset("/home/guodong/nlp/evonlp/src/resources/hf/glue", task_name, cache_dir=my_cache_dir)
for task_name in glue_tasks[:100]:
    # task_name = 'qqp'
    task_dataset = load_dataset("/home/guodong/nlp/evonlp/src/resources/huggingface/datasets/glue", task_name, 
                                cache_dir=my_cache_dir,
                                ignore_verifications=False)
    # task_dataset = load_dataset("glue", task_name)
    print('task-name:', task_name, len(task_dataset))
# # Replace 'local_glue_dataset' with the desired path to your local GLUE dataset directory.
# local_glue_path = 'local_glue_dataset'

# # Loop through the GLUE tasks and save the data to TSV files.
# for task_name in glue_tasks:
#     # Download the specific GLUE task dataset.
#     # task_dataset = load_dataset("glue", task_name, cache_dir=os.path.join('huggingface', "datasets"))
#     task_dataset = load_dataset("glue", task_name)
    
#     # Create a folder for the task if it doesn't exist.
#     task_folder = os.path.join(local_glue_path, task_name)
#     os.makedirs(task_folder, exist_ok=True)
    
#     # Save train, validation, and test data to TSV files.
#     for split_name, split_data in task_dataset.items():
#         tsv_file = os.path.join(task_folder, f"{split_name}.tsv")
#         with open(tsv_file, "w", encoding="utf-8") as file:
#             for example in split_data:
#                 sentence = example.get("sentence", "")
#                 label = example.get("label", "")
#                 file.write(f"{sentence}\t{label}\n")


# glue_path = '/home/guodong/nlp/evonlp/src/myglue'
# task_name = 'qqp'
# # raw_datasets = load_dataset(glue_path, "glue", split=f'{task_name}/train')
# raw_datasets = load_dataset('/home/guodong/nlp/evonlp/src/myglue/qqp/')

# train_dataset = raw_datasets["train"]
# eval_dataset = raw_datasets[
#     "validation_matched"
#     if task_name == "mnli"
#     else "validation"
# ]
# print(len(raw_datasets))
# print(len(eval_dataset))
