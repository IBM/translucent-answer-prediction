from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import pprint
import time
import random
import numpy as np
import torch
import os
import glob
import socket

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logger = logging.getLogger(__name__)


class Hypers:
    """
    This should be the base hyperparameters class, others should extend this.
    Note that args.num_instances must be set to however many instances this will train over: (train_size * num_epochs)
    """
    def __init__(self, args):
        #if args.num_instances <= 0:
        #    raise ValueError('num_instances must be set to the number of instances this will train over: '
        #                     '(train_size * num_epochs)')
        self.no_apex = args.no_apex

        if "RANK" not in os.environ or args.no_cuda:
            self.global_rank = 0
            self.local_rank = -1
            self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
            self.world_size = 1
        else:
            self.global_rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            env_master_addr = os.environ['MASTER_ADDR']
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            if os.environ['MASTER_ADDR'].startswith('file://'):
                torch.distributed.init_process_group(backend='nccl',
                                                     init_method=os.environ['MASTER_ADDR'],
                                                     world_size=int(os.environ['WORLD_SIZE']),
                                                     rank=self.global_rank)
                logger.info("init-method file: {}".format(env_master_addr))
            else:
                torch.distributed.init_process_group(backend='nccl')
                logger.info("init-method master_addr: {} master_port: {}".format(
                    env_master_addr, os.environ['MASTER_PORT']))

            logger.info(f"world_rank {self.global_rank} cuda_is_available {torch.cuda.is_available()} "
                        f"cuda_device_cnt {torch.cuda.device_count()} on {socket.gethostname()}")
            self.local_rank = int(self.global_rank % torch.cuda.device_count())
            self.n_gpu = 1
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits trainiing: {}".format(
            self.device, self.n_gpu, bool(self.local_rank != -1), args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError(f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, "
                             f"should be >= 1")

        if args.train_batch_size % (args.gradient_accumulation_steps * self.world_size) != 0:
            raise ValueError(f"train_batch_size {args.train_batch_size} must be divisible by "
                             f"gradient_accumulation_steps {args.gradient_accumulation_steps} "
                             f"times world_size {self.world_size}")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

        self.fp16 = args.fp16
        self.exclude_pooler = False  # set this if doing span selection pre-training, apex will complain otherwise
        self.loss_scale = args.loss_scale
        self.preprocessing_threads = 5
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.optimize_on_cpu = args.optimize_on_cpu
        self.train_batch_size = int(args.train_batch_size / (args.gradient_accumulation_steps * self.world_size))
        if args.num_instances is not None:
            self.num_train_steps = int(args.num_instances / args.train_batch_size)
        else:
            self.num_train_steps = None
        if args.seen_instances > 0:
            self.global_step = int(args.seen_instances / args.train_batch_size)
        else:
            self.global_step = 0
        self.warmup_proportion = args.warmup_proportion
        self.learning_rate = args.learning_rate
        self.bert_model = args.bert_model
        self.report_freq = 20 * self.world_size  # report on loss every N batches
        if hasattr(args, 'report_freq'):
            self.report_freq = args.report_freq
        if hasattr(args, 'time_limit'):
            self.time_limit = args.time_limit
        else:
            self.time_limit = -1
        # if we are using distributed training, only log from one process per node
        if self.global_rank > 0:
            try:
                logging.getLogger().setLevel(logging.WARNING)
            except:
                pass


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class BertTrainer:
    def __init__(self, hypers: Hypers, model_name, checkpoint, **extra_model_args):
        """
        initialize the BertOptimizer, with common logic for setting weight_decay_rate, doing gradient accumulation and
        tracking loss
        :param hypers: the core hyperparameters for the bert model
        :param model_name: the fully qualified name of the bert model we will train
            like pytorch_pretrained_bert.modeling.BertForQuestionAnswering
        :param checkpoint: if resuming training,
        this is the checkpoint that contains the optimizer state as checkpoint['optimizer']
        """

        self.init_time = time.time()

        self.model = self.get_model(hypers, model_name, checkpoint, **extra_model_args)

        self.step = 0
        self.hypers = hypers
        self.train_stats = TrainStats(hypers)

        self.model.train()
        logger.info('configured model for training')

        # show parameter names
        # logger.info(str([n for (n, p) in self.model.named_parameters()]))

        # Prepare optimizer
        if hasattr(hypers, 'exclude_pooler') and hypers.exclude_pooler:
            # module.bert.pooler.dense.weight, module.bert.pooler.dense.bias
            # see https://github.com/NVIDIA/apex/issues/131
            self.param_optimizer = [(n, p) for (n, p) in self.model.named_parameters() if '.pooler.' not in n]
        else:
            self.param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.t_total = hypers.num_train_steps
        self.global_step = hypers.global_step

        if hypers.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=hypers.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if hypers.loss_scale == 0:
                self.optimizer = FP16_Optimizer(optimizer,
                                                dynamic_loss_scale=True, verbose=(hypers.global_rank == 0))
            else:
                self.optimizer = FP16_Optimizer(optimizer,
                                                static_loss_scale=hypers.loss_scale, verbose=(hypers.global_rank == 0))
        else:
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=hypers.learning_rate,
                                      warmup=hypers.warmup_proportion,
                                      t_total=self.t_total)
        logger.info('created optimizer')

        if checkpoint and type(checkpoint) is dict and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if hypers.fp16:
                pass
            else:
                # if we load this state, we need to set the t_total to what we passed, not what was saved
                self.optimizer.set_t_total(self.t_total)
                # show state of optimizer
                lrs = self.optimizer.get_lr()
                logger.info('Min and max learn rate:    %s', str([min(lrs), max(lrs)]))
                logger.info('Min and max step in state: %s', str(self.optimizer.get_steps()))
            instances_per_step = hypers.train_batch_size * hypers.gradient_accumulation_steps * hypers.world_size
            if 'seen_instances' in checkpoint:
                self.global_step = int(checkpoint['seen_instances'] / instances_per_step)
                self.train_stats.previous_instances = checkpoint['seen_instances']
                logger.info('got global step from checkpoint = %i', self.global_step)

            logger.info('Loaded optimizer state:')
            logger.info(repr(self.optimizer))

    def reset(self):
        """
        reset any gradient accumulation
        :return:
        """
        self.model.zero_grad()
        self.step = 0

    def should_continue(self):
        """
        :return: True if training should continue
        """
        if self.global_step >= self.t_total:
            logger.info('stopping due to train step %i >= target train steps %i',
                        self.global_step, self.t_total)
            return False
        if 0 < self.hypers.time_limit <= (time.time() - self.init_time):
            logger.info('stopping due to time out %i seconds', self.hypers.time_limit)
            return False
        return True

    def save_simple(self, filename):
        if self.hypers.global_rank != 0:
            logger.info('skipping save in %i', torch.distributed.get_rank())
            return
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model itself
        torch.save(model_to_save.state_dict(), filename)
        logger.info(f'saved model only to {filename}')

    def save(self, filename, **extra_checkpoint_info):
        """
        save a checkpoint with the model parameters, the optimizer state and any additional checkpoint info
        :param filename:
        :param extra_checkpoint_info:
        :return:
        """
        # only local_rank 0, in fact only global rank 0
        if self.hypers.global_rank != 0:
            logger.info('skipping save in %i', torch.distributed.get_rank())
            return
        start_time = time.time()
        checkpoint = extra_checkpoint_info
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model itself
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # also save the optimizer state, since we will likely resume from partial pre-training
        checkpoint['state_dict'] = model_to_save.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        # include world size in instances_per_step calculation
        instances_per_step = self.hypers.train_batch_size * \
                             self.hypers.gradient_accumulation_steps * \
                             self.hypers.world_size
        checkpoint['seen_instances'] = self.global_step * instances_per_step
        checkpoint['num_instances'] = self.t_total * instances_per_step
        # CONSIDER: also save hypers?
        torch.save(checkpoint, filename)
        logger.info(f'saved model to {filename} in {time.time()-start_time} seconds')

    def get_instance_count(self):
        instances_per_step = self.hypers.train_batch_size * \
                             self.hypers.gradient_accumulation_steps * \
                             self.hypers.world_size
        return self.global_step * instances_per_step

    def step_loss(self, loss):
        """
        accumulates the gradient, tracks the loss and applies the gradient to the model
        :param loss: the loss from evaluating the model
        """
        if self.global_step == 0:
            logger.info('first step_loss')
        if self.hypers.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        self.train_stats.note_loss(loss.item())

        if self.hypers.gradient_accumulation_steps > 1:
            loss = loss / self.hypers.gradient_accumulation_steps

        if self.hypers.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        if (self.step + 1) % self.hypers.gradient_accumulation_steps == 0:
            lr_this_step = self.hypers.learning_rate * warmup_linear(self.global_step / self.t_total,
                                                                     self.hypers.warmup_proportion)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_this_step
            self.optimizer.step()
            self.model.zero_grad()
            self.global_step += 1

        self.step += 1

    @classmethod
    def get_files(cls, train_file, completed_files):
        logger.info('completed files = %s, count = %i',
                    str(completed_files[:min(5, len(completed_files))]), len(completed_files))
        # multiple train files
        if not os.path.isdir(train_file):
            train_files = [train_file]
        else:
            if not train_file.endswith('/'):
                train_file = train_file + '/'
            train_files = glob.glob(train_file + '**', recursive=True)
            train_files = [f for f in train_files if not os.path.isdir(f)]

        # exclude completed files
        if not set(train_files) == set(completed_files):
            train_files = [f for f in train_files if f not in completed_files]
        else:
            completed_files = []  # new epoch
        logger.info('train files = %s, count = %i',
                    str(train_files[:min(5, len(train_files))]), len(train_files))

        return train_files, completed_files

    @classmethod
    def get_model(cls, hypers, model_name, checkpoint, **extra_model_args):
        override_state_dict = None
        if checkpoint:
            if type(checkpoint) is dict and 'state_dict' in checkpoint:
                logger.info('loading from multi-part checkpoint')
                override_state_dict = checkpoint['state_dict']
            else:
                logger.info('loading from saved model parameters')
                override_state_dict = checkpoint

        # create the model object by name
        # https://stackoverflow.com/questions/4821104/python-dynamic-instantiation-from-string-name-of-a-class-in-dynamically-imported
        import importlib
        clsdot = model_name.rfind('.')
        class_ = getattr(importlib.import_module(model_name[0:clsdot]), model_name[clsdot + 1:])

        model_args = {'state_dict': override_state_dict,
                      'cache_dir': PYTORCH_PRETRAINED_BERT_CACHE}
        model_args.update(extra_model_args)
        # logger.info(pprint.pformat(extra_model_args, indent=4))
        model = class_.from_pretrained(hypers.bert_model, **model_args)

        logger.info('built model')

        # configure model for fp16, multi-gpu and/or distributed training
        if hypers.fp16:
            model.half()
            logger.info('model halved')
        logger.info('sending model to %s', str(hypers.device))
        model.to(hypers.device)
        logger.info('sent model to %s', str(hypers.device))

        if hypers.local_rank != -1:
            if not hypers.no_apex:
                try:
                    from apex.parallel import DistributedDataParallel as DDP
                    model = DDP(model)
                except ImportError:
                    raise ImportError("Please install apex")
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[hypers.local_rank],
                                                                  output_device=hypers.local_rank)
            logger.info('using DistributedDataParallel for world size %i', hypers.world_size)
        elif hypers.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        return model

    @classmethod
    def get_base_parser(cls):
        parser = argparse.ArgumentParser()

        # Required parameters
        parser.add_argument("--bert_model", default=None, type=str, required=True,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

        # Other parameters
        parser.add_argument("--num_instances", default=-1, type=int,
                            help="Total number of training instances to train over.")
        parser.add_argument("--seen_instances", default=-1, type=int,
                            help="When resuming training, the number of instances we have already trained over.")
        parser.add_argument("--train_batch_size", default=32, type=int,
                            help="Total batch size for training.")
        parser.add_argument("--learning_rate", default=5e-5, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--warmup_proportion", default=0.1, type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10% of training.")
        parser.add_argument("--no_cuda", default=False, action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument("--no_apex", default=False, action='store_true',
                            help="Whether not to use apex when available")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--optimize_on_cpu', default=False, action='store_true',
                            help="Whether to perform optimization and keep the optimizer averages on CPU")
        parser.add_argument('--fp16', default=False, action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--loss_scale', type=float, default=0,
                            help='Loss scaling, positive power of 2 values can improve fp16 convergence. '
                                 'Leave at zero to use dynamic loss scaling')
        return parser


class TrainStats:
    def __init__(self, hypers: Hypers):
        self.sum_loss = 0
        self.total_batches = 0
        self.previous_instances = 0
        self.recency_weighted_loss = 0
        self.start = time.time()
        # last batch counts for 0.5%
        self.recency_weight = 0.005 / hypers.gradient_accumulation_steps
        self.train_batch_size = hypers.train_batch_size
        self.report_freq = hypers.report_freq * hypers.gradient_accumulation_steps
        self.instances_per_step = hypers.train_batch_size * hypers.world_size
        self._timing_paused = True
        self._paused_at = self.start

    def note_loss(self, loss_item):
        self.total_batches += 1
        if self._timing_paused:
            self.start += (time.time() - self._paused_at)  # don't count the time we were paused
            self._timing_paused = False
            if self.total_batches == 1:
                logger.info('finished first batch')
        self.sum_loss += loss_item
        # compute recency weighted loss
        loss_weight = max(self.recency_weight, 1.0 / self.total_batches)
        self.recency_weighted_loss = (1.0 - loss_weight) * self.recency_weighted_loss + loss_weight * loss_item
        if self.total_batches % self.report_freq == 0:
            inst_count = self.total_batches * self.instances_per_step
            logger.info(f"Loss {self.sum_loss / self.total_batches}, "
                        f"Recent Loss {self.recency_weighted_loss}, "
                        f"Instances {inst_count + self.previous_instances}, "
                        f"Instances per second = {inst_count / (time.time() - self.start)}")

    def instances_per_second(self):
        inst_count = self.total_batches * self.instances_per_step
        return inst_count / (time.time() - self.start)

    def pause_timing(self):
        self._timing_paused = True
        self._paused_at = time.time()
