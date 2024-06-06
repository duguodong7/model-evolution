import logging
import json, os, csv
import re
import torch
import numpy as np
import math
from .misc import filter_params_to_merge
from collections import OrderedDict
from .avg_merger import FedAvgMerger

def fmt(name, met_dict):
    return "{}\t{}".format(name, json.dumps(met_dict))

def MaxMinNorm_Sigmoid(score, MIN=-5, MAX=0):  #-5, -2
    scaled_score = []
    for x in score:
        x1 = (x - min(score)) / (max(score) - min(score)) 
        scaled_score.append(x1 * (MAX - MIN) + MIN)
    out_score = [1/(1 + math.exp(-num))+0.05 for num in scaled_score]
    return out_score

class EvolverBase:
    def __init__(
        self, config, local_models, global_model, evo_ds=None, test_dss=None,
    ):
        self.local_models = local_models
        self.global_model = global_model
        self.config = config
        self.evo_cfg = self.config.evolver
        self.evo_ds = evo_ds
        self.test_dss = test_dss

class Evolver(EvolverBase):
    def __init__(
        self, config, local_models, global_model, evo_ds=None, test_dss=None):
        super().__init__(config, local_models, global_model, evo_ds, test_dss)
        self.csv_file = os.path.join(self.config.main_output_dir, 'evolver.csv')
        self.rowd = OrderedDict()
        self.merger_options = None
        self.with_merger = self.config.evolver.with_merger
        self.pop_size = len(self.local_models)
        # self.test_dss = [loc_m.test_dataset for loc_m in self.loca_models] 
        self.global_output_dir = config.global_model.output_dir_format.format(
                                        main_output_dir=config.main_output_dir)
        # model_coeffs=None, fisher_weights=None, grams=None, **kwargs

    def evolve_to_global(self, merger_options):
        logging.info('Start to evolve ...')
        self.fw = merger_options["fisher_weights"]
        # self.grams = merger_options["grams"]
        self.merger_options = merger_options
        if self.config.evolver.resume_global:
            self.config.evolver.save_global = False
            params = torch.load(self.global_output_dir+'/evolved_global.pt')
            for n, p in self.global_model.base.named_parameters():
                if n in params: p.data.copy_(params[n])
            logging.info("Evolver global model is resumed from {}".format(self.global_output_dir))
        self.fevolver()
        if self.config.evolver.save_global:        
            params = {k: v for k, v in self.global_model.base.named_parameters()}
            torch.save(params, self.global_output_dir+'/evolved_global.pt')
            logging.info("Evolvered global model is saved")

    def fevolve(self):
        self.fishers_normed = [x/torch.stack(self.fishers).mean(0) for x in self.fishers]
        self.fishers = self.fishers_normed
        # sum_p = params * fisher * coeff
        # params = {}
        # for local_model in self.local_models:
        #     n2p = {k: v for k, v in local_model.base.named_parameters()}
        #     merge_param_names = filter_params_to_merge(
        #         [n for n in n2p], self.merger_config.exclude_param_regex
        #     )
        #     for n in merge_param_names:
        #         if n not in params:
        #             params[n] = []
        #         params[n].append(n2p[n])

        # all_params = params
        # for n, params in all_params.items():
        #     hh = 0
        model_pop = self.models_to_pop()
        fisher_pop = self.fishers_to_pop()
        score = self.score_func()
        test_score = self.test_score_func()
        # init_model_pop, init_score = list(torch.stack(model_pop).clone()), score
        # best_model_pop, best_test_score = list(torch.stack(model_pop).clone()), test_score
        best_model_pop, best_test_score = None, 0
        # [i.clone().detach() for i in model_pop]
        self.rowd.update([('generation', 'Init'), ('idx', 'null'), ('score', score), ('test', test_score)])
        self.update_summary(write_header=True)
        logging.info(">>>>>>>>current iteration: {} idx: {} score: {} test_score: {}"
                             .format('Init', 'null', score, test_score))

        for iter in range(0, self.config.evolver.max_iters):
            lst = range(0, self.pop_size)
            # for idx in np.random.permutation(lst):   # for idx in lst:
            rand_idx = np.random.permutation(lst)[0]
            fisher_pop_idx = self.mutate_and_crossover(iter, rand_idx, fisher_pop, score) #, *_ 
            fisher_pop[rand_idx] = fisher_pop_idx  # 变异用的种群和更新用的种群可以分离，直到变异种群被榨干了再被替换；
            # model_pop_idx = self.update_model(fisher_pop, model_pop)
            model_pop_global = self.update_model_pop(fisher_pop, model_pop)
            self.update_global_model(model_pop_global)
            # self.update_local_model(idx, model_pop_idx)
            score_idx = self.score_func()
            score_worst_idx = 0
            score # socre 的更新有两种方案：1）超过最好的更新，这样只用存一个score; 2) 超过原来最差的就更新，这样要存一个score_lst;
            # 这两种方案感觉差不多，第二种搜索范围更大，效果更稳定；
            # 原本只更新模型参数，现在是模型参数和matrics一起更新；
            # 
            worst_idx = 0
            model_pop, score, test_score = self.update_pop_and_score2(idx, model_pop, \
                                                    model_pop_global, score, score_idx, test_score)
            update_sucessfully = True
            def compute_fisher(model, dataloader, step):
                return None
            if update_sucessfully: 
                fisher_new = compute_fisher(model_pop_global)
                self.fishers[worst_idx] = fisher_new
            self.update_local_model(idx, model_pop[idx])
            logging.info(">>>>>>>>current iteration: {} idx: {} score: {} test_score: {}"
                            .format(iter, idx, score, test_score))
            self.rowd.update([('generation', iter), ('idx', idx), ('score', score), ('test', test_score)])
            self.update_summary()

            if test_score > best_test_score:
                best_model_pop, best_test_score = self.models_to_pop(), test_score
                # best_model_pop, best_test_score = list(torch.stack(model_pop)), test_score
                # for idx in range(self.pop_size): 
                #     self.update_local_model(idx, best_model_pop[idx])
                # merger = FedAvgMerger(self.config, self.config.merger,
                #     self.local_models, self.global_model, merger_ds=None,)
                # merger.merge_to_global(**self.merger_options)
                # actual_best_test_score = self.test_score_func2()
                logging.info(">>>>>>>>score: {}, best_test_score: {}".format(score, test_score))

            # if iter < -10:
            #     score = init_score
            #     for idx in range(self.pop_size): 
            #         model_pop[idx] = init_model_pop[idx].clone()   # !!!!!!!
            #         self.update_local_model(idx, model_pop[idx])

        # push the best model_pop to global
        for idx in range(self.pop_size): 
            self.update_local_model(idx, best_model_pop[idx])
        merger = FedAvgMerger(self.config, self.config.merger,
            self.local_models, self.global_model, merger_ds=None,)
        merger.merge_to_global(**self.merger_options)
        best_test_score = self.test_score_func2()
        logging.info(">>>>>>>>final best_test_score: {}".format(best_test_score))
        self.rowd.update([('generation', 'best'), ('idx', 'null'), ('score', score), ('test', best_test_score)])
        self.update_summary()

    def update_model_pop(self, fisher_pop, model_pop):
        avg_params = 0
        # coeff = torch.FloatTensor(1/self.pop_size)
        for i in range(self.pop_size):
            avg_params += fisher_pop[i] * model_pop[i] / self.pop_size
        return avg_params
        # return sum
    
    # def score_func(self, idx):
    #     local_model = self.local_models[idx]
    #     metrics = local_model.trainer.evaluate(eval_dataset=self.evo_ds, \
    #                 ignore_keys=None, metric_key_prefix='evo')
    #     out = round(metrics['evo_key_score'], 5)
    #     return out

    def score_func(self):
        # merger = FedAvgMerger(self.config, self.config.merger,
        #     self.local_models, self.global_model, merger_ds=None,)
        # merger.merge_to_global(**self.merger_options)
        # avg_params = 
        # for n, p in self.global_model.base.named_parameters():
        #     if n in avg_params:
        #         p.data.copy_(avg_params[n])

        # merger_options['gram'] = sefl.local_models.compute_regmean
        # merger = FedAvgMerger(self.config, self.config.merger,
        #         self.local_models, self.global_model, merger_ds=None,)
        # merger.merge_to_global(**self.merger_options)
        # metrics = self.global_model.evaluate(self.evo_ds)
        metrics = self.global_model.trainer.evaluate(eval_dataset=self.evo_ds, \
                                        ignore_keys=None, metric_key_prefix='evo')
        return round(metrics['evo_key_score'], 5)

    def test_score_func(self, idx): 
        local_model = self.local_models[idx]
        local_test_score = []
        for local_eval_set in self.test_dss:
            metrics = local_model.evaluate(local_eval_set)
            local_test_score.append(metrics['eval_key_score'])
        return round(np.mean(local_test_score), 5)

    def test_score_func2(self): 
        # since test_score is called after score or None, there is no need to merge to global
        test_score_global = []
        for local_eval_set in self.test_dss:
            metrics = self.global_model.evaluate(local_eval_set)
            test_score_global.append(metrics['eval_key_score'])
        return np.mean(test_score_global)

    def update_pop_and_score(self, idx, model_pop, model_pop_idx, score, score_idx, test_score):
        t= score_idx
        t2 = t + 0 # regul[idx]
        if t2 > (score[idx]):
            model_pop[idx] = model_pop_idx.clone().detach()
            score[idx] = t
            # update the test result for every success
            test_score_idx_lst = []
            for local_eval_set in self.test_dss:
                metrics = self.local_models[idx].evaluate(local_eval_set)
                test_score_idx_lst.append(metrics['eval_key_score'])
            test_score_idx = np.mean(test_score_idx_lst)
            test_score[idx] = test_score_idx
        return model_pop, score, test_score

    def update_pop_and_score2(self, idx, model_pop, model_pop_idx, score, score_idx, test_score):
        t= score_idx
        t2 = t + 0 # regul[idx]
        if t2 > score:
            model_pop[idx] = model_pop_idx.clone().detach()
            # 如果没有clone, 则更新成功的结果没有能被保存下来，反而被下一个mutate and crossover得到的model_pop_idx代替了；
            # 这样效果相当于，更新成功则变成下一次的变异结果，不成功则不变，但是下一次不一定是成功的，相当于是更新成了随机的变异
            score = t
            # update the test result for every success
            test_score_idx_lst = []
            for local_eval_set in self.test_dss:
                metrics = self.global_model.evaluate(local_eval_set)
                test_score_idx_lst.append(metrics['eval_key_score'])
            test_score = np.mean(test_score_idx_lst)

        return model_pop, score, test_score

    def mutate_and_crossover(self, iter, idx, fisher_pop, score):
        popsize = self.pop_size
        dim = len(fisher_pop[0])
        device = fisher_pop[0].device
        # f = self.config.evolver.f
        max_iters = self.config.evolver.max_iters
        # cr_max, cr_min = self.config.evolver.cr_max, self.config.evolver.cr_min 
        exp = self.config.templates.exp_name
        f = float(exp.split('_')[1])
        cr = float(exp.split('_')[3])
        p = 0.5 / (max_iters - iter) + 0.5
        # p = 0.95
        # cr = max(cr_max * (1 - iter/max_iters), 0.2)
        # cr = cr_max-(cr_max-cr_min)*iter/max_iters
        # cr = cr_max-(cr_max-cr_min)*1/max_iters
        # cr = cr_max
        # np.random.seed(0)
        # score = MaxMinNorm_Sigmoid(score)
        if not self.with_merger:
            bestidx = self.find_max_index_after_removing_idx(score, idx)
            if np.random.rand() < p:
                id_ref = np.random.choice(list(set(range(0, popsize)) - {idx}), 1, replace=False)[0]
            else:
                id_ref = bestidx
        else:
            id_ref = np.random.choice(list(set(range(0, popsize)) - {idx}), 1, replace=False)[0]
        logging.info('id_ref: {}'.format(id_ref))
        # x_new = (score[idx] * model_pop[idx] + score[id_ref] * model_pop[id_ref]) / (score[idx]+score[id_ref])
        fisher_new = (1-f) * fisher_pop[idx] + f * fisher_pop[id_ref]
        # x_new = (gram[idx] * model_pop[idx] + gram[id_ref] * model_pop[id_ref])/(gram[idx]+gram[id_ref])
        # x_new = merger.merge(model_pop[idx], model_pop[id_ref])
        fisher_new_cr = torch.where(torch.rand(dim).to(device) < torch.ones(dim).to(device)*cr, \
                               fisher_new, fisher_pop[idx]) # no clone is ok, "where" create new memory;
        return fisher_new_cr

    def models_to_pop(self):
        def model_dict_to_vector(local_model, merger_config):
            n2p = {k: v for k, v in local_model.base.named_parameters()}
            merge_param_names = filter_params_to_merge(
                [n for n in n2p], merger_config.exclude_param_regex
            )
            weights_vector = []
            for k, v in n2p.items():
                if k in merge_param_names:
                    vector = v.flatten().clone().detach()
                    weights_vector.append(vector)
            return torch.cat(weights_vector, 0)   # torch.cat create a new memory for new tensor lst
        model_pop = []
        for local_model in self.local_models:
            solution = model_dict_to_vector(local_model, self.config.merger)
            model_pop.append(solution)
        return model_pop
    
    def fishers_to_pop(self):
        def fisher_to_vector(fisher, merger_config):
            n2p = {k: v for k, v in fisher}
            merge_param_names = filter_params_to_merge(
                [n for n in n2p], merger_config.exclude_param_regex
            )
            weights_vector = []
            for k, v in n2p.items():
                if k in merge_param_names:
                    vector = v.flatten().clone().detach()
                    weights_vector.append(vector)
            return torch.cat(weights_vector, 0)   # torch.cat create a new memory for new tensor lst
        model_pop = []
        for fisher in self.fishers:
            solution = fisher_to_vector(fisher, self.config.merger)
            model_pop.append(solution)
        return model_pop

    def update_local_model(self, idx, model_pop_idx):
        local_model = self.local_models[idx]
        weights_vector = model_pop_idx
        n2p = {k: v for k, v in local_model.base.named_parameters()}
        merge_param_names = filter_params_to_merge(
            [n for n in n2p], self.config.merger.exclude_param_regex
        )
        start = 0
        for k, v in n2p.items():
            if k in merge_param_names:
                layer_weights_shape = v.shape
                layer_weights_size = v.numel()
                layer_weights_vector = weights_vector[start: start+layer_weights_size]
                restored_v = layer_weights_vector.view(layer_weights_shape).contiguous()
                v.data.copy_(restored_v.data)   # data.copy_ if safer than weights_dict[key]
                start = start + layer_weights_size

    def update_global_model(self, weights_vector):
        # local_model = self.local_models[idx]
        # weights_vector = model_pop_idx
        n2p = {k: v for k, v in self.global_model.base.named_parameters()}
        merge_param_names = filter_params_to_merge(
            [n for n in n2p], self.config.merger.exclude_param_regex
        )
        start = 0
        for k, v in n2p.items():
            if k in merge_param_names:
                layer_weights_shape = v.shape
                layer_weights_size = v.numel()
                layer_weights_vector = weights_vector[start: start+layer_weights_size]
                restored_v = layer_weights_vector.view(layer_weights_shape).contiguous()
                v.data.copy_(restored_v.data)   # data.copy_ if safer than weights_dict[key]
                start = start + layer_weights_size

    def deliver_to_local(self):
        n2p = {k: v for k, v in self.global_model.named_parameters()}
        merge_param_names = filter_params_to_merge(
            [n for n in n2p], self.config.merger.exclude_param_regex
        )
        for local_model in self.local_models:
            for n, p in local_model.named_parameters():
                if n in merge_param_names:
                    p.data.copy_(n2p[n].data)

    def find_max_index_after_removing_idx(self, lst, idx):
        lst_without_idx = lst[:idx] + lst[idx+1:]
        max_idx = lst_without_idx.index(max(lst_without_idx))
        # 考虑到索引的变化，如果最大值索引在移除的索引之前，需要加上1
        if max_idx >= idx: max_idx += 1
        return max_idx

    def update_summary(self, write_header=True):
        filename = self.csv_file
        rowd = self.rowd
        with open(filename, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=rowd.keys())
            if write_header: dw.writeheader()
            dw.writerow(rowd)

