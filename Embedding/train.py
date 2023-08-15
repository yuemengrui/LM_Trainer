# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import math
import json
import time
import torch
import numpy as np
from loguru import logger
from models import build_model
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

        self.output_dir = self.configs.get('output_dir')
        os.makedirs(self.output_dir, exist_ok=True)

        self.global_step = 0
        self.start_epoch = 0
        self.num_epochs = self.configs.get('num_epochs')

        self._initialize()
        self.device = self.model.device
        logger.info('train with device {} and pytorch {}'.format(self.device, torch.__version__))

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
        logger.info('start train')
        for epoch in range(self.start_epoch, self.num_epochs):
            self.epoch_result = self._train_epoch(epoch)
            self._on_epoch_finish()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()

        batch_start = time.time()
        for step, (inputs, labels) in enumerate(self.train_loader):
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            for k, v in inputs.items():
                v = v.squeeze(1)  # [batch, 1, seq_len] -> [batch, seq_len]
                inputs[k] = v.to(self.device)
            labels = labels.to(self.device)

            cur_batch_size = labels.size()[0]

            output_embeddings = self.model.get_embeddings(inputs)
            loss = self.calc_loss(labels, output_embeddings)

            current_loss = loss.item()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            batch_cost = time.time() - batch_start

            logger.info(
                f"[epoch:{epoch}/{self.num_epochs}] [step:{self.global_step}/{self.total_steps}] loss:{current_loss:.6f} lr:{lr:.9f} batch_cost:{batch_cost:.2f}s, speed:{cur_batch_size / batch_cost:.1f}/s")

            batch_start = time.time()

        return {'epoch_cost': time.time() - epoch_start, 'epoch': epoch}

    def evaluate(self):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        start = time.time()
        results = {}

        self.model.eval()

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
                source_embeddings = self.model.get_embeddings(source)
                target_embeddings = self.model.get_embeddings(target)
                preds = torch.cosine_similarity(source_embeddings, target_embeddings)
            batch_preds.extend(preds.cpu().numpy())

        spearman = compute_spearmanr(batch_labels, batch_preds)
        pearson = compute_pearsonr(batch_labels, batch_preds)
        logger.info(f"labels: {batch_labels[:10]}")
        logger.info(f"preds:  {batch_preds[:10]}")
        logger.info(f"pearson: {pearson}, spearman: {spearman}")

        results["eval_spearman"] = spearman
        results["eval_pearson"] = pearson

        return results, time.time() - start

    def _on_epoch_finish(self):
        logger.info(f"epoch:{self.epoch_result['epoch']} finished. epoch_cost:{self.epoch_result['epoch_cost']:.2f}s\n")

        metrics, eval_cost = self.evaluate()

        if metrics['eval_spearman'] > self.metrics['eval_spearman']:
            self.metrics.update(metrics)
            self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            self.metrics['best_model_step'] = self.global_step

            self._save_checkpoint(self.epoch_result['epoch'], self.global_step, best=True)
            logger.info(f"Saving best model. best model: {self.metrics}\n")
        else:
            self._save_checkpoint(self.epoch_result['epoch'], self.global_step)
            logger.info(f"Saving best model")

    def _on_train_finish(self):
        logger.info('train finished!!!')

    def _save_checkpoint(self, epoch, step, best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best_old.pth.tar'
        """
        if best:
            save_dir = os.path.join(self.output_dir, "best")
        else:
            save_dir = os.path.join(self.output_dir, f"epoch-{epoch}-step-{step}")

        os.makedirs(save_dir, exist_ok=True)
        self.model.save_adapter_model(save_dir)
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'configs': self.configs,
            'metrics': self.metrics
        }

        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))
        torch.save(self.scheduler.state_dict(), os.path.join(save_dir, "scheduler.bin"))
        with open(os.path.join(save_dir, "info.json"), 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self, checkpoint_dir):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        logger.info("Loading checkpoint: {} ...".format(checkpoint_dir))

        self.model.load_adapter_model(checkpoint_dir)

        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.bin")))
        self.scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.bin")))

        with open(os.path.join(checkpoint_dir, "info.json"), 'r') as f:
            data = json.load(f)

        self.global_step = data["global_step"]
        self.start_epoch = data['epoch']

        if 'metrics' in data:
            self.metrics = data['metrics']

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        logger.info("resume from checkpoint (epoch {})".format(self.start_epoch))

        if self.start_epoch >= self.num_epochs:
            self.num_epochs = self.start_epoch + 5

    def _initialize(self):

        self.model = build_model(**self.configs)

        t = time.time()
        train_dataset = CosentTrainDataset(self.model.tokenizer,
                                           load_cosent_train_data(self.configs['dataset_dir']),
                                           self.configs['max_seq_len'])
        eval_dataset = TextMatchingTestDataset(self.model.tokenizer,
                                               load_text_matching_test_data(self.configs['dataset_dir']),
                                               self.configs['max_seq_len'])

        self.train_data_total = len(train_dataset)
        self.eval_data_total = len(eval_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=self.configs.get('batch_size'), shuffle=True,
                                       drop_last=True)
        self.train_loader_len = len(self.train_loader)

        self.eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        self.eval_loader_len = len(self.eval_loader)

        dataset_msg = '\n---------------Dataset Information---------------\n'
        dataset_msg += 'train data total:{}'.format(self.train_data_total)
        dataset_msg += '\neval data total:{}'.format(self.eval_data_total)
        dataset_msg += '\ntime_cost:{:.2f}s'.format(time.time() - t)
        dataset_msg += '\ndataset load success'
        dataset_msg += '\n---------------Dataset Information---------------\n'
        logger.info(dataset_msg)

        self.optimizer = AdamW(params=self.model.embedding_layer.parameters(), lr=self.configs.get('lr'),
                               correct_bias=False)

        self.total_steps = self.train_loader_len * self.configs['num_epochs']
        warmup_steps = math.ceil(self.total_steps * self.configs['warmup_ratio'])
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
        #                                                  num_training_steps=self.total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=self.total_steps)

        if self.configs.get("resume_checkpoint", None):
            self._load_checkpoint(self.configs["resume_checkpoint"])

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Batch size = {self.configs['batch_size']}")
        logger.info(f"  Num steps = {self.total_steps}")
        logger.info(f"  Warmup-steps: {warmup_steps}")


if __name__ == '__main__':
    configs = {
        "llm_model_name_or_path": "/Users/yuemengrui/Data/LLM/ChatGLM-6B/V2",
        "device": "mps",
        "dataset_dir": "./data",
        "output_dir": "./output",
        "max_seq_len": 512,
        "num_epochs": 2,
        "batch_size": 2,
        "lr": 2e-5,
        "warmup_ratio": 0.05,
        "resume_checkpoint": None
    }

    trainer = Trainer(configs)
    trainer.train()
