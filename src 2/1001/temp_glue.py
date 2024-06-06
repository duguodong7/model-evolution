import os
import json
import numpy as np
# path_lst = os.listdir(path)

def r5(x):
    return round(x, 5)

def obtain_after_merge_met_in_domain(method = 'fisher', seeds = [1,2,3,4,5]):
    # print('after_merge:', os.path.join(path, method))
    seed_lst = []
    for seed in seeds: #,2,3,4,5
        exp = method + '/' + str(seed)
        file = os.path.join(path, exp, 'metrics.json')
        if not os.path.exists(file): continue
        with open(file, encoding="utf-8") as f: met = json.load(f)
        # after_merge_lst = []
        # for modelname, metric in met["after_merge_locals"]['in_domain'].items():
        #     after_merge_lst.append(round(metric['eval_key_score'], 4))
        # after_merge_mean = r5(np.mean(after_merge_lst[:5]))
        after_merge_mean = r5(met["after_merge_locals"]['in_domain']['model0']['eval_key_score'])
        seed_lst.append(after_merge_mean)
    # print('mean: {} of seed_lst: {}'.format(r5(np.mean(seed_lst)), seed_lst))
    return r5(np.mean(seed_lst))

def obtain_before_merge_met_in_domain(method = 'fisher', seeds = [1,2,3,4,5]):
    # print('before_merge:', os.path.join(path, method))
    seed_lst = []
    for seed in seeds: #,2,3,4,5
        exp = method + '/' + str(seed)
        file = os.path.join(path, exp, 'metrics.json')
        if not os.path.exists(file): continue
        with open(file, encoding="utf-8") as f: met = json.load(f)
        before_merge_lst = []
        for modelname, metric in met["before_merge_locals"]['in_domain'].items():
            before_merge_lst.append(round(metric['eval_key_score'], 4))
        before_merge_mean = r5(np.mean(before_merge_lst[:5]))
        seed_lst.append(before_merge_mean)
    # print('mean: {} of seed_lst: {}'.format(r5(np.mean(seed_lst)), seed_lst))
    return r5(np.mean(seed_lst))

if __name__ == '__main__':
    print('tasks:', ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte" ])
    path = '/data/guodong/nlp/runs/glue-roberta-base'
    seeds = [1,2,3,4,5,6,7,8]
    # method_lst = ["simple", "fisher", "regmean0.5"] 
    # method_lst = ["simple", "fisher", "regmean0.5", "regmean0.2", "regmean0.1", "regmean"] 
    # method_lst = ["simple", "fisher", "regmean0.05", "regmean0.1", "regmean0.2", "regmean0.3", "regmean0.5", "regmean", ]
    method_lst = ["simple", "evolver", "fisher", "evolver_fisher", "regmean0.5", "evolver_reg0.5"]
    for method in method_lst: 
        after_task_lst = []
        for task in ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte"]:
            after_merge_mean = obtain_after_merge_met_in_domain(method = f'niid_1k_{method}/{task}', seeds = seeds)
            after_task_lst.append(after_merge_mean)
        print('method:', method)
        print('after merge mean: {} of task_lst: {}'.format(r5(np.mean(after_task_lst)), after_task_lst))

    method = "simple"
    after_task_lst, before_task_lst = [], []
    for task in ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte"]:
        after_merge_mean = obtain_after_merge_met_in_domain(method = f'niid_1k_{method}/{task}', seeds = seeds)
        before_merge_mean = obtain_before_merge_met_in_domain(method = f'niid_1k_{method}/{task}', seeds = seeds)
        before_task_lst.append(before_merge_mean)
        after_task_lst.append(after_merge_mean)
    print('method:', "AVG")
    # print('after merge mean: {} of task_lst: {}'.format(r5(np.mean(after_task_lst)), after_task_lst))
    print('before merge mean: {} of task_lst: {}'.format(r5(np.mean(before_task_lst)), before_task_lst))
