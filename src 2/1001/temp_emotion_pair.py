import os
import json
import numpy as np

# path = '/data/guodong/nlp/runs/emotion-9.12'
# path_lst = os.listdir(path)

def r5(x):
    return round(x, 5)

def obtain_after_merge_met_in_domain(method = 'simple_avg', exp_name='haha', diffseed=False):
    # print('obtain_after_merge_met_in_domain of exp_path:', os.path.join(path, method))
    seed_lst = []
    for seed in [1,2,3,4,5]: #,2,3,4,5
        if diffseed:
            exp = method + '/'+exp_name + '/seed' + str(seed) + '/diffseed_' + str(seed)
        else:
            exp = method + '/'+exp_name + '/seed' + str(seed)
        file = os.path.join(path, exp, 'metrics.json')
        if not os.path.exists(file): continue
        with open(file, encoding="utf-8") as f: met = json.load(f)
        before_merge_lst = []
        for modelname, metric in met["after_merge_locals"]['in_domain'].items():
            before_merge_lst.append(round(metric['eval_key_score'], 4))
        before_merge_mean = r5(np.mean(before_merge_lst[:5]))
        seed_lst.append(before_merge_mean)
    # print('in_domain mean: {} of seed_lst_before_merge: {}'.format(r5(np.mean(seed_lst)), seed_lst))
    return r5(np.mean(seed_lst))

# def obtain_after_merge_met_ood(method = 'simple_avg', exp_name='haha', diffseed=False):
#     print('obtain_after_merge_met_ood of exp_path:', os.path.join(path, method))
#     seed_lst = []
#     for seed in [1,2,3,4,5]: #,2,3,4,5
#         if diffseed:
#             exp = method + '/'+exp_name + '/seed' + str(seed) + '/diffseed_' + str(seed)
#         else:
#             exp = method + '/'+exp_name + '/seed' + str(seed)
#         file = os.path.join(path, exp, 'metrics.json')
#         if not os.path.exists(file): continue
#         with open(file, encoding="utf-8") as f: met = json.load(f)
#         before_merge_lst = []
#         for modelname, metric in met["after_merge_locals"]['ood']['model0'].items():
#             before_merge_lst.append(round(metric['eval_key_score'], 4))
#         before_merge_mean = r5(np.mean(before_merge_lst[:5]))
#         seed_lst.append(before_merge_mean)
#     print('ood mean: {} of seed_lst_before_merge: {}'.format(r5(np.mean(seed_lst)), seed_lst))

def obtain_before_merge_met(method = 'simple_avg', exp_name='haha', diffseed=False):
    print('obtain_before_merge_met of exp_path:', os.path.join(path, method))
    seed_lst = []
    for seed in [1,2,3,4,5]: #,2,3,4,5
        if diffseed:
            exp = method + '/'+exp_name + '/seed' + str(seed) + '/diffseed_' + str(seed)
        else:
            exp = method + '/'+exp_name + '/seed' + str(seed)
        file = os.path.join(path, exp, 'metrics.json')
        if not os.path.exists(file): continue
        with open(file, encoding="utf-8") as f: met = json.load(f)
        before_merge_lst = []
        for modelname, metric in met['before_merge_locals']['in_domain'].items():
            before_merge_lst.append(round(metric['eval_key_score'], 4))
        before_merge_mean = r5(np.mean(before_merge_lst[:5]))
        seed_lst.append(before_merge_mean)
    # print('in_domain mean: {} of seed_lst_before_merge: {}'.format(r5(np.mean(seed_lst)), seed_lst))
    return r5(np.mean(seed_lst))
              
def obtain_ensemble_met(method='simple_avg', exp_name='haha', diffseed=False):
    print('obtain ensemble met of exp_path:', os.path.join(path, method))
    domain = "ensemble"
    seed_lst = []
    for seed in [1,2,3,4,5]: #,2,3,4,5
        if diffseed:
            exp = method + '/'+exp_name + '/seed' + str(seed) + '/diffseed_' + str(seed)
        else:
            exp = method + '/'+exp_name + '/seed' + str(seed)
        file = os.path.join(path, exp, 'metrics.json')
        if not os.path.exists(file): continue
        with open(file, encoding="utf-8") as f: met = json.load(f)
        # print(met.keys())
        in_domain_ensemble_lst = []
        for modelname, metric in met["before_merge_ensemble"][domain].items():
            in_domain_ensemble_lst.append(round(metric['key_score'], 4))
        in_domain_ensemble_mean = np.mean(in_domain_ensemble_lst)
        seed_lst.append(r5(in_domain_ensemble_mean))
    return r5(np.mean(seed_lst))

def obtain_individuals_met(method='simple_avg', exp_name='haha', diffseed=False):
    for domain in ["in_domain"]:
        seed_lst_avg, seed_lst_best = [], []
        for seed in [1,2,3,4,5]: #,2,3,4,5
            if diffseed:
                exp = method + '/'+exp_name + '/seed' + str(seed) + '/diffseed_' + str(seed)
            else:
                exp = method + '/'+exp_name + '/seed' + str(seed)
            file = os.path.join(path, exp, 'metrics.json')
            if not os.path.exists(file): continue
            # print('obtain individuals met of exp_path:', exp)
            with open(file, encoding="utf-8") as f: met = json.load(f)

            individuals_lst = []
            for individual, indi_dict in met["before_merge_individuals"][domain].items():
                indi_lst = []
                for id, metric in indi_dict.items():
                    indi_lst.append(round(metric['eval_key_score'], 5))
                individuals_lst.append(round(np.mean(indi_lst), 5))
            individuals_mean = np.mean(individuals_lst)
            individuals_best = np.max(individuals_lst)
            seed_lst_avg.append(individuals_mean)
            seed_lst_best.append(individuals_best)
            print('AVG:', round(np.mean(individuals_mean), 5), \
                  'BEST:', round(np.mean(individuals_best), 5), \
                  'lst:', individuals_lst)
        # print(domain, round(np.mean(seed_lst_avg), 5), round(np.mean(seed_lst_best), 5))
    return round(np.mean(seed_lst_avg), 5), round(np.mean(seed_lst_best), 5)

if __name__ == '__main__':
    # path = '/data/guodong/nlp/runs/emotion-roberta_base'
    # path = '/data/guodong/nlp/runs/emotion-deberta-large'
    path = "/data/guodong/nlp/runs/emotion-distilbert-base-uncased"
    # sorted()
    # m_lst1, m_lst2, exp_name_lst = [], [], []
    # for exp_name in os.listdir(os.path.join(path, 'simple')):
    #     exp_name_lst.append(exp_name)
    #     avg, best = obtain_individuals_met('simple', exp_name)
    #     m_lst1.append(avg)
    #     m_lst2.append(best)
    # print(exp_name_lst)
    # print(np.mean(m_lst1), m_lst1)
    # print(np.mean(m_lst2), m_lst2)

    # m_lst, exp_name_lst = [], []
    # for exp_name in os.listdir(os.path.join(path, 'simple')):
    #     exp_name_lst.append(exp_name)
    #     out = obtain_ensemble_met('simple', exp_name)
    #     # out = obtain_before_merge_met('simple', exp_name)
    #     # out = obtain_after_merge_met_in_domain('simple', exp_name)
    #     m_lst.append(out)
    # print(exp_name_lst)
    # print(np.mean(m_lst), m_lst)

    # obtain_ensemble_met('simple')
    # obtain_before_merge_met('simple')  
    # obtain_after_merge_met_ood('simple')
    # obtain_after_merge_met_in_domain('simple')

    # method_lst = ["evolver-reg_afa0.5_cr0.5_f0.5_g15"]
    method_lst=["evolver-fisher_cr0.5_f0.5_g15", 'fisher'] # 'fisher'
    # method_lst=['simple', "evolver-simple_cr0.5_f0.5_g15"] 
    # method_lst=['evolver-reg0.3_cr0.5_f0.5_g15', 'regmean0.3'] # 'regmean0.3'
    for method in method_lst:
        m_lst, exp_name_lst = [], []
        for exp_name in os.listdir(os.path.join(path, method)):
            exp_name_lst.append(exp_name)
            a = obtain_after_merge_met_in_domain(method, exp_name, diffseed=False)
            m_lst.append(a)
            # obtain_after_merge_met_ood(method, exp_name, diffseed=False)
        print(exp_name_lst)
        print(np.mean(m_lst), m_lst)
