# Copyright 2023 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def filter_params_to_merge(param_names, exclude_param_regex):
    params_to_merge = []
    for name in param_names:
        valid = not any([re.match(patt, name) for patt in exclude_param_regex])
        if valid:
            params_to_merge.append(name)
    return params_to_merge


def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any(
            [re.match(patt, name) for patt in include_patterns]
        )
        valid_type = not include_type or any(
            [isinstance(module, md_cls) for md_cls in include_type]
        )
        if valid_type and valid_name:
            modules[name] = module
    return modules


import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# plot_evolver_data_with_markers(df_filtered)
def plot_evolver_data_with_markers(df):
    df = pd.read_csv('C:/Users/25435/Desktop/IJCAI/1/evolver_nlp.csv')
    df_filtered = df[df['generation'] != 'generation']
    df_filtered['score'] = df_filtered['score'].apply(ast.literal_eval)
    df_filtered['test'] = df_filtered['test'].apply(ast.literal_eval)

    fig, ax = plt.subplots(figsize=(15, 8))
    for idx, row in df.iterrows():
        generation = row['generation']
        scores = row['score']
        tests = row['test']
        ax.scatter([generation] * len(scores), scores, c='red', marker='o', label='Score' if idx == 0 else "")
        ax.scatter([generation] * len(tests), tests, c='blue', marker='x', label='Test' if idx == 0 else "")

    custom_lines = [Line2D([0], [0], marker='o', color='red', lw=0),
                    Line2D([0], [0], marker='x', color='blue', lw=0)]
    ax.legend(custom_lines, ['Score', 'Test'])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Value')
    ax.set_title('Evolver Data with Different Markers')

    # plt.show()
    plt.savefig()
