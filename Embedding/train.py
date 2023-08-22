# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import math
import json
import time
import torch
import shutil
import numpy as np
import torch.nn as nn
from pathlib import Path
from loguru import logger
from copy import deepcopy
from models import build_model
from parse_args import args_parse
from torch.utils.data import DataLoader
from similarity import cos_sim, compute_spearmanr, compute_pearsonr
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from dataset import CosentTrainDataset, load_cosent_train_data, TextMatchingTestDataset, load_text_matching_test_data


def calc_similarity_scores(model, sents1, sents2, labels):
    t1 = time.time()
    e1 = model.encode(sents1)
    e2 = model.encode(sents2)
    spend_time = time.time() - t1
    s = cos_sim(e1, e2)
    sims = []
    for i in range(len(sents1)):
        sims.append(s[i][i])
    sims = np.array(sims)
    labels = np.array(labels)
    spearman = compute_spearmanr(labels, sims)
    logger.debug(f'labels: {labels[:10]}')
    logger.debug(f'preds:  {sims[:10]}')
    logger.debug(f'Spearman: {spearman}')
    logger.debug(
        f'spend time: {spend_time:.4f}, count:{len(sents1 + sents2)}, qps: {len(sents1 + sents2) / spend_time}')
    return spearman


class Trainer:

    def __init__(self, configs):
        logger.info(json.dumps(configs, ensure_ascii=False, indent=2))
        self.configs = configs

        self.train_loader = None
        self.train_data_total = 0
        self.train_loader_len = 0
        self.eval_loader = None
        self.eval_data_total = 0
        self.eval_loader_len = 0

        self.output_dir = self.configs['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.global_step = 0
        self.current_epoch = 0
        self.start_epoch = 0
        self.num_epochs = self.configs['num_epochs']
        self.data_steps = 0

        self.device = torch.device(self.configs['device'])
        logger.info('train with device {} and pytorch {}'.format(self.device, torch.__version__))

        self._initialize()

        self.metrics = {
            "best_model_step": 0,
            "best_model_epoch": 0,
            "eval_spearman": 0,
            "eval_pearson": 0,
        }

    def calc_loss(self, y_true, y_pred):
        """
        矩阵计算batch内的cos loss
        """
        # 1. 取出真实的标签
        y_true = y_true[::2]  # tensor([1, 0, 1]) 真实的标签
        # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
        norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
        y_pred = y_pred / norms
        # 3. 奇偶向量相乘, 相似度矩阵除以温度系数0.05(等于*20)
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
        # 4. 取出负例-正例的差值
        y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
        y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
        y_true = y_true.float()
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        # 这里加0是因为e^0 = 1相当于在log中加了1
        y_pred = torch.cat((torch.tensor([0]).float().to(self.device), y_pred), dim=0)
        return torch.logsumexp(y_pred, dim=0)

    def train(self):
        """
        Full training logic
        """
        training_start = time.time()
        self.embedding_model.zero_grad()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch
            start = time.time()
            self._train_epoch()
            epoch_cost = time.time() - start
            self._on_epoch_finish(epoch_cost)

        self._on_train_finish(time.time() - training_start)

    def _train_epoch(self):
        self.embedding_model.train()

        batch_start = time.time()
        for data_steps, (inputs, labels) in enumerate(self.train_loader):
            if data_steps + 1 < self.data_steps:
                continue

            lr = self.optimizer.param_groups[0]['lr']

            for k, v in inputs.items():
                v = v.squeeze(1)  # [batch, 1, seq_len] -> [batch, seq_len]
                inputs[k] = v.to(self.device)
            labels = labels.to(self.device)

            cur_batch_size = labels.size()[0]

            output_embeddings = self.model_cls.get_embeddings(inputs)
            loss = self.calc_loss(labels, output_embeddings)

            current_loss = loss.item()
            if self.configs['gradient_accumulation_steps'] > 1:
                loss = loss / self.configs['gradient_accumulation_steps']
            loss.backward()
            if (data_steps + 1) % self.configs['gradient_accumulation_steps'] == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self.data_steps = data_steps + 1

                batch_cost = time.time() - batch_start

                if self.global_step % self.configs['logging_steps'] == 0:
                    eta = (batch_cost * (
                            self.num_epochs - self.current_epoch - 1) * self.train_loader_len + batch_cost * (
                                   self.train_loader_len - data_steps)) / 60 / 60
                    logger.info(
                        f" [epoch:{self.current_epoch}/{self.num_epochs}]"
                        f" [step:{self.global_step}/{self.total_steps}]"
                        f" loss:{current_loss:.6f}"
                        f" lr:{lr:.9f}"
                        f" batch_cost:{batch_cost:.2f}s"
                        f" speed:{cur_batch_size / batch_cost:.1f}/s"
                        f" [data:{self.data_steps}/{self.train_loader_len} - {(self.data_steps / self.train_loader_len) + self.current_epoch:.2f} epochs]"
                        f" ETA:{eta:.2f}h")

                if self.global_step % self.configs['save_steps'] == 0:
                    self._save_checkpoint()

                if self.global_step % self.configs['eval_steps'] == 0:
                    self.do_eval()

            batch_start = time.time()

    def evaluate(self):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        logger.info(f"****** start eval ******")
        start = time.time()
        results = {}

        self.embedding_model.eval()

        batch_labels = []
        batch_preds = []
        for source, target, labels in self.eval_loader:
            for k, v in source.items():
                v = v.squeeze(1)  # [batch, 1, seq_len] -> [batch, seq_len]
                source[k] = v.to(self.device)

            for k, v in target.items():
                v = v.squeeze(1)
                target[k] = v.to(self.device)

            labels = labels.to(self.device)

            batch_labels.extend(labels.cpu().numpy())

            with torch.no_grad():
                source_embeddings = self.model_cls.get_embeddings(source)
                target_embeddings = self.model_cls.get_embeddings(target)
                preds = torch.cosine_similarity(source_embeddings, target_embeddings)
            batch_preds.extend(preds.cpu().numpy())

        spearman = compute_spearmanr(batch_labels, batch_preds)
        pearson = compute_pearsonr(batch_labels, batch_preds)

        results["eval_spearman"] = spearman
        results["eval_pearson"] = pearson

        logger.info(f"****** eval finished! time_cost: {time.time() - start:.2f}s results: {results}******")
        return results

    def do_eval(self):
        metrics = self.evaluate()

        if metrics['eval_spearman'] > self.metrics['eval_spearman']:
            self.metrics.update(metrics)
            self.metrics['best_model_epoch'] = self.current_epoch
            self.metrics['best_model_step'] = self.global_step

            self._save_checkpoint(best=True)
            logger.info(f" Saving best model. best model: {self.metrics}\n")
        else:
            self._save_checkpoint()

    def _on_epoch_finish(self, epoch_cost):
        logger.info(f" epoch:{self.current_epoch} finished. epoch_cost:{epoch_cost:.2f}s\n")
        self.do_eval()
        self.data_steps = 0

    def _on_train_finish(self, training_cost):
        logger.info(f'****** train finished!!! training_cost: {training_cost/60/60:.2f}h ******')

    def _sorted_checkpoints(self, checkpoint_prefix='epoch'):
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self):
        checkpoints_sorted = self._sorted_checkpoints()
        if len(checkpoints_sorted) <= self.configs['save_total_limit']:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.configs['save_total_limit'] + 1)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                f" Deleting older checkpoint [{checkpoint}] due to save_total_limit:{self.configs['save_total_limit']}")
            shutil.rmtree(checkpoint, ignore_errors=True)

        return

    def _save_checkpoint(self, best=False):
        if best:
            save_dir = os.path.join(self.output_dir, "best")
        else:
            save_dir = os.path.join(self.output_dir, f"epoch-{self.current_epoch}-step-{self.global_step}")
            self._rotate_checkpoints()

        logger.info(f" saving model epoch:{self.current_epoch} step:{self.global_step}")
        os.makedirs(save_dir, exist_ok=True)
        self.embedding_model.save_adapter_model(save_dir)
        states = {
            'batch_size': self.configs['batch_size'],
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'data_steps': self.data_steps,
            'train_data_total': self.train_data_total,
            'train_loader_len': self.train_loader_len,
            'configs': self.configs,
            'metrics': self.metrics
        }

        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))
        torch.save(self.scheduler.state_dict(), os.path.join(save_dir, "scheduler.bin"))
        with open(os.path.join(save_dir, "info.json"), 'w') as f:
            json.dump(states, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self, checkpoint_dir):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        logger.info(f" Loading checkpoint: {checkpoint_dir} ......")

        just_init_weight = self.configs.get("just_init_weight", False)

        self.embedding_model.load_adapter_model(checkpoint_dir)

        if not just_init_weight:
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.bin")))
            self.scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.bin")))

            with open(os.path.join(checkpoint_dir, "info.json"), 'r') as f:
                states = json.load(f)

            self.global_step = states["global_step"]
            self.start_epoch = states['epoch']
            if states['train_data_total'] == self.train_data_total and states[
                'train_loader_len'] == self.train_loader_len and states['batch_size'] == self.configs['batch_size']:
                self.data_steps = states['data_steps']

            self.metrics = states['metrics']

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            logger.info(f" Continuing training from epoch {self.start_epoch}")
            logger.info(f" Continuing training from global step {self.global_step}")
            logger.info(f" Will skip the first {self.data_steps} steps in the current epoch")

            if self.start_epoch >= self.num_epochs:
                self.num_epochs = self.start_epoch + 5

    def _initialize(self):

        self.model_cls = build_model(**self.configs)
        self.tokenizer = self.model_cls.tokenizer
        self.embedding_model = self.model_cls.embedding_model
        if self.configs['data_parallel']:
            self.embedding_model = nn.DataParallel(self.embedding_model)

        t = time.time()
        logger.info(f"****** Dataset Information ******")
        train_dataset = CosentTrainDataset(self.tokenizer,
                                           load_cosent_train_data(self.configs['dataset_dir']),
                                           self.configs['max_seq_len'])
        eval_dataset = TextMatchingTestDataset(self.tokenizer,
                                               load_text_matching_test_data(self.configs['dataset_dir']),
                                               self.configs['max_seq_len'])

        self.train_data_total = len(train_dataset)
        self.eval_data_total = len(eval_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=self.configs.get('batch_size'), shuffle=False,
                                       drop_last=True)
        self.train_loader_len = len(self.train_loader)

        self.eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        self.eval_loader_len = len(self.eval_loader)

        logger.info(f" train data total:{self.train_data_total}")
        logger.info(f" eval data total:{self.eval_data_total}")
        logger.info(f" time cost: {time.time() - t}s")
        logger.info(f"****** Dataset Information ******")

        self.optimizer = AdamW(params=filter(lambda p: p.requires_grad, self.embedding_model.parameters()),
                               lr=self.configs.get('lr'),
                               correct_bias=False)

        self.total_steps = self.train_loader_len * self.configs['num_epochs']
        warmup_steps = math.ceil(self.total_steps * self.configs['warmup_ratio'])
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
        #                                                  num_training_steps=self.total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=self.total_steps)

        resume_checkpoint = self.configs.get("resume_checkpoint")
        if os.path.exists(resume_checkpoint):
            self._load_checkpoint(self.configs["resume_checkpoint"])

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {self.train_loader_len}")
        logger.info(f"  Batch size = {self.configs['batch_size']}")
        logger.info(f"  Num steps = {self.total_steps}")
        logger.info(f"  Warmup-steps: {warmup_steps}")


if __name__ == '__main__':
    trainer = Trainer(vars(args_parse()))
    trainer.train()
