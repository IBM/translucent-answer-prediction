import logging
import time
from pytorch_pretrained_bert.optimization import BertAdam

logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class Optimizer:
    def __init__(self, hypers, model, global_step, t_total, no_decay=['bias', 'LayerNorm.bias', 'LayerNorm.weight']):
        self.hypers = hypers
        self.model = model
        self.global_step = global_step
        self.t_total = t_total

        self.init_time = time.time()
        self.step = 0

        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        if hypers.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters, lr=hypers.learning_rate,
                                  bias_correction=False, max_grad_norm=1.0)
            if hypers.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True, verbose=(hypers.global_rank == 0))
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=hypers.loss_scale, verbose=(hypers.global_rank == 0))
        else:
            optimizer = BertAdam(optimizer_grouped_parameters, lr=hypers.learning_rate, warmup=hypers.warmup_proportion, t_total=t_total)
        logger.info('created optimizer')
        self.optimizer = optimizer

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, optimizer_state, seen_instances=None):
        self.optimizer.load_state_dict(optimizer_state)
        if self.hypers.fp16:
            pass
        else:
            # if we load this state, we need to set the t_total to what we passed, not what was saved
            self.optimizer.set_t_total(self.t_total)
            # show state of optimizer
            lrs = self.optimizer.get_lr()
            logger.info('Min and max learn rate:    %s', str([min(lrs), max(lrs)]))
            logger.info('Min and max step in state: %s', str(self.optimizer.get_steps()))
        instances_per_step = self.hypers.train_batch_size * self.hypers.gradient_accumulation_steps * self.hypers.world_size
        if seen_instances is not None:
            self.global_step = int(seen_instances / instances_per_step)
            logger.info('got global step from checkpoint = %i', self.global_step)
        logger.info('Loaded optimizer state:')
        logger.info(repr(self.optimizer))

    def step_loss(self, loss):
        if self.global_step == 0:
            logger.info('first step_loss')
        if self.hypers.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        if self.hypers.gradient_accumulation_steps > 1:
            loss = loss / self.hypers.gradient_accumulation_steps

        if self.hypers.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        if (self.step + 1) % self.hypers.gradient_accumulation_steps == 0:
            lr_this_step = self.hypers.learning_rate * warmup_linear(self.global_step / self.t_total,
                                                                     self.hypers.warmup_proportion)
            if lr_this_step < 0:
                raise ValueError(f'!!! LEARN RATE LESS THAN ZERO: {lr_this_step} !!!')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_this_step
            self.optimizer.step()
            self.model.zero_grad()
            self.global_step += 1

        self.step += 1

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
