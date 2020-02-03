import argparse
import logging
import random
import numpy as np
import torch
import os
import socket
import json
from datetime import datetime
from util.io_help import to_serializable

logger = logging.getLogger(__name__)


class HypersBase:
    """
    This should be the base hyperparameters class, others should extend this.
    """
    def __init__(self, args):
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
        self.seed = args.seed

        self.fp16 = args.fp16
        self.loss_scale = args.loss_scale
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.optimize_on_cpu = args.optimize_on_cpu
        self.train_batch_size = int(args.train_batch_size / (args.gradient_accumulation_steps * self.world_size))
        if hasattr(args, 'num_instances') and args.num_instances is not None:
            self.num_train_steps = int(args.num_instances / args.train_batch_size)
        else:
            self.num_train_steps = None
        if hasattr(args, 'seen_instances') and args.seen_instances > 0:
            self.global_step = int(args.seen_instances / args.train_batch_size)
        else:
            self.global_step = 0
        self.warmup_proportion = args.warmup_proportion
        self.learning_rate = args.learning_rate
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
        if hasattr(args, 'experiment_name') and args.experiment_name:
            self.experiment_name = args.experiment_name
        else:
            self.experiment_name = f'exp_{str(datetime.now())}'

    def wrap_model(self, model):
        """
        configure model for fp16, multi-gpu and/or distributed training
        :param model:
        :return:
        """
        if self.fp16:
            model.half()
            logger.info('model halved')
        logger.info('sending model to %s', str(self.device))
        model.to(self.device)
        logger.info('sent model to %s', str(self.device))

        if self.local_rank != -1:
            if not self.no_apex:
                try:
                    from apex.parallel import DistributedDataParallel as DDP
                    model = DDP(model)
                except ImportError:
                    raise ImportError("Please install apex")
            else:
                logger.info('using DistributedDataParallel for world size %i', self.world_size)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank],
                                                                      output_device=self.local_rank)
        elif self.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        return model

    def write_results_file(self, results_dir, **kwargs):
        os.makedirs(results_dir, exist_ok=True)
        file = os.path.join(results_dir, self.experiment_name+'.json')
        run_num = 1
        while os.path.exists(file):
            run_num += 1
            file = os.path.join(results_dir, self.experiment_name+f'_{run_num}.json')
        result_dict = dict(self.__dict__)
        del result_dict['device']  # not JSON serializable and not really a hyperparameter
        del result_dict['local_rank']
        del result_dict['global_rank']
        # add git information about the version of the code
        if 'GIT_REPOSITORY' in os.environ:
            result_dict['git_repository'] = os.environ['GIT_REPOSITORY']
        if 'GIT_COMMIT_ID' in os.environ:
            result_dict['git_commit_id'] = os.environ['GIT_COMMIT_ID']
        # add extra info (results, instances-per-second)
        for k, v in kwargs.items():
            result_dict[k] = v
        with open(file, 'w') as fp:
            json.dump(result_dict, fp, indent=4, default=to_serializable)

    @classmethod
    def get_base_parser(cls):
        parser = argparse.ArgumentParser()
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
        # not sure if optimize_on_cpu actually works in the optimizer.py or bert_trainer_apex.py
        parser.add_argument('--optimize_on_cpu', default=False, action='store_true',
                            help="Whether to perform optimization and keep the optimizer averages on CPU")
        parser.add_argument('--fp16', default=False, action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--loss_scale', type=float, default=0,
                            help='Loss scaling, positive power of 2 values can improve fp16 convergence. '
                                 'Leave at zero to use dynamic loss scaling')
        return parser
