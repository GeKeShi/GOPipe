#!/usr/bin/env python
import argparse
import logging
import os
import sys
from ast import literal_eval

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.distributed as dist
from mlperf_compliance import mlperf_log

import seq2seq.data.config as config
import seq2seq.train.trainer3 as trainers
import seq2seq.utils as utils
from seq2seq.data.dataset import LazyParallelDataset
from seq2seq.data.dataset import ParallelDataset
from seq2seq.data.dataset import TextDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.inference.inference import Translator

from seq2seq.models.sequential_model import embed1, embed2, dropout1, dropout2, dropout3
from seq2seq.models.sequential_model import lstm1, lstm2, lstm3, Attention, Classifier

from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.utils import gnmt_print
from collections import OrderedDict

from seq2seq.train.lr_scheduler import WarmupMultiStepLR
from seq2seq.utils import AverageMeter
from torch.nn.utils import clip_grad_norm_
import random
import time


def parse_args():
    """
    Parse commandline arguments.
    """
    def exclusive_group(group, name, default, help):
        destname = name.replace('-', '_')
        subgroup = group.add_mutually_exclusive_group(required=False)
        subgroup.add_argument(f'--{name}', dest=f'{destname}',
                              action='store_true',
                              help=f'{help} (use \'--no-{name}\' to disable)')
        subgroup.add_argument(f'--no-{name}', dest=f'{destname}',
                              action='store_false', help=argparse.SUPPRESS)
        subgroup.set_defaults(**{destname: default})

    parser = argparse.ArgumentParser(
        description='GNMT training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--dataset-dir', default='data/wmt16_de_en',
                         help='path to the directory with training/test data')
    dataset.add_argument('--max-size', default=None, type=int,
                         help='use at most MAX_SIZE elements from training \
                         dataset (useful for benchmarking), by default \
                         uses entire dataset')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--results-dir', default='results',
                         help='path to directory with results, it will be \
                         automatically created if it does not exist')
    results.add_argument('--save', default='gnmt',
                         help='defines subdirectory within RESULTS_DIR for \
                         results from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--hidden-size', default=1024, type=int,
                       help='model hidden size')
    model.add_argument('--num-layers', default=4, type=int,
                       help='number of RNN layers in encoder and in decoder')
    model.add_argument('--dropout', default=0.2, type=float,
                       help='dropout applied to input of RNN cells')

    exclusive_group(group=model, name='share-embedding', default=True,
                    help='use shared embeddings for encoder and decoder')

    model.add_argument('--smoothing', default=0.1, type=float,
                       help='label smoothing, if equal to zero model will use \
                       CrossEntropyLoss, if not zero model will be trained \
                       with label smoothing loss')

    # setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp32', choices=['fp16', 'fp32'],
                         help='arithmetic type')
    general.add_argument('--seed', default=None, type=int,
                         help='master seed for random number generators, if \
                         "seed" is undefined then the master seed will be \
                         sampled from random.SystemRandom()')

    exclusive_group(group=general, name='eval', default=True,
                    help='run validation and test after every epoch')
    exclusive_group(group=general, name='env', default=False,
                    help='print info about execution env')
    exclusive_group(group=general, name='cuda', default=True,
                    help='enables cuda')
    exclusive_group(group=general, name='cudnn', default=True,
                    help='enables cudnn')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--train-batch-size', default=128, type=int,
                          help='training batch size per worker')
    training.add_argument('--train-global-batch-size', default=None, type=int,
                          help='global training batch size, this argument \
                          does not have to be defined, if it is defined it \
                          will be used to automatically \
                          compute train_iter_size \
                          using the equation: train_iter_size = \
                          train_global_batch_size // (train_batch_size * \
                          world_size)')
    training.add_argument('--train-iter-size', metavar='N', default=1,
                          type=int,
                          help='training iter size, training loop will \
                          accumulate gradients over N iterations and execute \
                          optimizer every N steps')
    training.add_argument('--epochs', default=8, type=int,
                          help='max number of training epochs')

    training.add_argument('--grad-clip', default=5.0, type=float,
                          help='enables gradient clipping and sets maximum \
                          norm of gradients')
    training.add_argument('--max-length-train', default=50, type=int,
                          help='maximum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--min-length-train', default=0, type=int,
                          help='minimum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--train-loader-workers', default=2, type=int,
                          help='number of workers for training data loading')
    training.add_argument('--batching', default='bucketing', type=str,
                          choices=['random', 'sharding', 'bucketing'],
                          help='select batching algorithm')
    training.add_argument('--shard-size', default=80, type=int,
                          help='shard size for "sharding" batching algorithm, \
                          in multiples of global batch size')
    training.add_argument('--num-buckets', default=5, type=int,
                          help='number of buckets for "bucketing" batching \
                          algorithm')
    # training.add_argument('--train-num', default=4, type=int,
    #                       help='number of split')

    # optimizer
    optimizer = parser.add_argument_group('optimizer setup')
    optimizer.add_argument('--optimizer', type=str, default='Adam',
                           help='training optimizer')
    optimizer.add_argument('--lr', type=float, default=1.00e-3,
                           help='learning rate')
    optimizer.add_argument('--optimizer-extra', type=str,
                           default="{}",
                           help='extra options for the optimizer')

    # scheduler
    scheduler = parser.add_argument_group('learning rate scheduler setup')
    scheduler.add_argument('--warmup-steps', type=str, default='200',
                           help='number of learning rate warmup iterations')
    scheduler.add_argument('--remain-steps', type=str, default='0.666',
                           help='starting iteration for learning rate decay')
    scheduler.add_argument('--decay-interval', type=str, default='None',
                           help='interval between learning rate decay steps')
    scheduler.add_argument('--decay-steps', type=int, default=4,
                           help='max number of learning rate decay steps')
    scheduler.add_argument('--decay-factor', type=float, default=0.5,
                           help='learning rate decay factor')

    # validation
    val = parser.add_argument_group('validation setup')
    val.add_argument('--val-batch-size', default=64, type=int,
                     help='batch size for validation')
    val.add_argument('--max-length-val', default=125, type=int,
                     help='maximum sequence length for validation \
                     (including special BOS and EOS tokens)')
    val.add_argument('--min-length-val', default=0, type=int,
                     help='minimum sequence length for validation \
                     (including special BOS and EOS tokens)')
    val.add_argument('--val-loader-workers', default=0, type=int,
                     help='number of workers for validation data loading')

    # test
    test = parser.add_argument_group('test setup')
    test.add_argument('--test-batch-size', default=128, type=int,
                      help='batch size for test')
    test.add_argument('--max-length-test', default=150, type=int,
                      help='maximum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--min-length-test', default=0, type=int,
                      help='minimum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--beam-size', default=5, type=int,
                      help='beam size')
    test.add_argument('--len-norm-factor', default=0.6, type=float,
                      help='length normalization factor')
    test.add_argument('--cov-penalty-factor', default=0.1, type=float,
                      help='coverage penalty factor')
    test.add_argument('--len-norm-const', default=5.0, type=float,
                      help='length normalization constant')
    test.add_argument('--intra-epoch-eval', metavar='N', default=0, type=int,
                      help='evaluate within training epoch, this option will \
                      enable extra N equally spaced evaluations executed \
                      during each training epoch')
    test.add_argument('--test-loader-workers', default=0, type=int,
                      help='number of workers for test data loading')

    # checkpointing
    chkpt = parser.add_argument_group('checkpointing setup')
    chkpt.add_argument('--start-epoch', default=0, type=int,
                       help='manually set initial epoch counter')
    chkpt.add_argument('--resume', default=None, type=str, metavar='PATH',
                       help='resumes training from checkpoint from PATH')
    chkpt.add_argument('--save-all', action='store_true', default=False,
                       help='saves checkpoint after every epoch')
    chkpt.add_argument('--save-freq', default=5000, type=int,
                       help='save checkpoint every SAVE_FREQ batches')
    chkpt.add_argument('--keep-checkpoints', default=0, type=int,
                       help='keep only last KEEP_CHECKPOINTS checkpoints, \
                       affects only checkpoints controlled by --save-freq \
                       option')

    # benchmarking
    benchmark = parser.add_argument_group('benchmark setup')
    benchmark.add_argument('--target-bleu', default=24.0, type=float,
                           help='target accuracy, training will be stopped \
                           when the target is achieved')

    # distributed
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='node_rank')
    distributed.add_argument('--local_rank', default=0, type=int,
                             help='local rank of the process, do not set!')
    distributed.add_argument('--num-split', default=4, type=int,
                             help='number of stages use to train single model')
    distributed.add_argument('--train-num', default=4, type=int,
                             help='number of minibatch')
    distributed.add_argument('--Gpipe', default=False, type=bool,
                             help='Gpipe with no checkpoint')

    args = parser.parse_args()

    args.warmup_steps = literal_eval(args.warmup_steps)
    args.remain_steps = literal_eval(args.remain_steps)
    args.decay_interval = literal_eval(args.decay_interval)

    return args

def build_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.:
        logging.info(f'Building CrossEntropyLoss')
        loss_weight = torch.ones(vocab_size)
        loss_weight[padding_idx] = 0
        criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
        gnmt_print(key=mlperf_log.MODEL_HP_LOSS_FN,
                   value='Cross Entropy', sync=False)
    else:
        logging.info(f'Building LabelSmoothingLoss (smoothing: {smoothing})')
        criterion = LabelSmoothing(padding_idx, smoothing)
        gnmt_print(key=mlperf_log.MODEL_HP_LOSS_FN,
                   value='Cross Entropy with label smoothing', sync=False)
        gnmt_print(key=mlperf_log.MODEL_HP_LOSS_SMOOTHING,
                   value=smoothing, sync=False)

    return criterion

class SynchronizedWallClockTimer:
    class Timer:
        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            assert not self.started_
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self, reset=True):
            assert self.started_
            torch.cuda.synchronize()
            if reset:
                self.elapsed_ = (time.time() - self.start_time)
            else:
                self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            started_ = self.started_
            if self.started_:
                self.stop()
            elapsed_ = self.elapsed_
            if reset:
                self.reset()
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

def get_memory(tensors):
    mem = 0.0
    for t in tensors:
        if t.dtype is torch.float32:
            mem += round(t.numel() * 4 / (1024 * 1024), 2)
        elif t.dtype is torch.int64:
            mem += round(t.numel() * 8 / (1024 * 1024), 2)
    return mem

def main():
    mlperf_log.ROOT_DIR_GNMT = os.path.dirname(os.path.abspath(__file__))
    mlperf_log.LOGGER.propagate = False

    args = parse_args()

    # create directory for results
    save_path = os.path.join(args.results_dir, args.save)
    args.save_path = save_path
    os.makedirs(save_path, exist_ok=True)

    # setup logging
    log_filename = f'log_rank_0.log'
    utils.setup_logging(os.path.join(save_path, log_filename))

    # build tokenizer
    pad_vocab = utils.pad_vocabulary(args.math)
    tokenizer = Tokenizer(os.path.join(args.dataset_dir, config.VOCAB_FNAME),
                          pad_vocab)

    # build datasets
    gnmt_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING, sync=False)
    gnmt_print(key=mlperf_log.TRAIN_HP_MAX_SEQ_LEN,
               value=args.max_length_train, sync=False)

    train_data = LazyParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TRAIN_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_TRAIN_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_train,
        max_len=args.max_length_train,
        sort=False,
        max_size=args.max_size)

    gnmt_print(key=mlperf_log.PREPROC_NUM_TRAIN_EXAMPLES,
               value=len(train_data), sync=False)

    val_data = ParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_VAL_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_VAL_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_val,
        max_len=args.max_length_val,
        sort=True)

    gnmt_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL, sync=False)

    test_data = TextDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TEST_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_test,
        max_len=args.max_length_test,
        sort=True)

    gnmt_print(key=mlperf_log.PREPROC_NUM_EVAL_EXAMPLES,
               value=len(test_data), sync=False)

    vocab_size = tokenizer.vocab_size
    gnmt_print(key=mlperf_log.PREPROC_VOCAB_SIZE,
               value=vocab_size, sync=False)

    # get data loaders
    batching_opt = {'shard_size': args.shard_size,
                    'num_buckets': args.num_buckets}
    batch_size = args.train_batch_size

    master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
    seeding_rng = random.Random(master_seed)
    shuffling_seeds = [seeding_rng.randint(0, 2 ** 32 - 1) for _ in range(8)]
    num_worker = 1
    batch_first = False
    worker_rank = 0
    train_loader = train_data.get_loader(batch_size=batch_size,
                                         seeds=shuffling_seeds,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         batching=args.batching,
                                         batching_opt=batching_opt,
                                         num_workers=args.train_loader_workers,
                                         world_size=num_worker,
                                         rank=worker_rank)

    gnmt_print(key=mlperf_log.INPUT_BATCH_SIZE,
               value=batch_size,
               sync=False)
    gnmt_print(key=mlperf_log.INPUT_SIZE,
               value=train_loader.sampler.num_samples, sync=False)

    batch_size = args.val_batch_size
    val_loader = val_data.get_loader(batch_size=batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
                                     num_workers=args.val_loader_workers,
                                     world_size=num_worker,
                                     rank=worker_rank)

    test_loader = test_data.get_loader(batch_size=args.test_batch_size,
                                       batch_first=batch_first,
                                       shuffle=False,
                                       pad=True,
                                       num_workers=args.test_loader_workers,
                                       world_size=num_worker,
                                       rank=worker_rank)

    gnmt_print(key=mlperf_log.EVAL_SIZE,
               value=len(test_loader.dataset), sync=False)

    # build criterion
    criterion = build_criterion(vocab_size, config.PAD, args.smoothing)
    criterion = criterion.cuda()

    # define scheduler and optimizer
    opt_config = {'optimizer': args.optimizer, 'lr': args.lr}
    opt_config.update(literal_eval(args.optimizer_extra))
    logging.info(f'Training optimizer config: {opt_config}')

    scheduler_config = {'warmup_steps': args.warmup_steps,
                        'remain_steps': args.remain_steps,
                        'decay_interval': args.decay_interval,
                        'decay_steps': args.decay_steps,
                        'decay_factor': args.decay_factor}

    logging.info(f'Training LR schedule config: {scheduler_config}')

    # model
    model = nn.Sequential(OrderedDict([
        ('Emb1', embed1()),
        ('E_lstm1', lstm1(bi=True, residual=False)),
        ('Dropout1', dropout1()),
        ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
        ('Dropout2', dropout1()),
        ('E_lstm3', lstm1()),
        ('Dropout3', dropout1()),
        ('E_lstm4', lstm1()),
        ('Dropout4', dropout1()),
        ('E_lstm5', lstm1()),
        ('Dropout5', dropout1()),
        ('E_lstm6', lstm1()),
        ('Dropout6', dropout1()),
        ('E_lstm7', lstm1()),
        ('Dropout7', dropout1()),
        ('E_lstm8', lstm1()),

        ('Emb2', embed2()),
        ('Dropout8', dropout2()),
        ('D_lstm1', lstm2()),
        ('Attention', Attention()),
        ('Dropout9', dropout3()),
        ('D_lstm2', lstm3(residual=False)),
        ('Dropout10', dropout3()),
        ('D_lstm3', lstm3()),
        ('Dropout11', dropout3()),
        ('D_lstm4', lstm3()),
        ('Dropout12', dropout3()),
        ('D_lstm5', lstm3()),
        ('Dropout13', dropout3()),
        ('D_lstm6', lstm3()),
        ('Dropout14', dropout3()),
        ('D_lstm7', lstm3()),
        ('Dropout15', dropout3()),
        ('D_lstm8', lstm3(last=True)),
        ('Classifier', Classifier(1024, 32320))
    ])).cuda()
    logging.info(model)

    opt_name = opt_config.pop('optimizer')
    params = model.parameters()
    optimizer = torch.optim.__dict__[opt_name](params, **opt_config)
    logging.info(f'Using optimizer: {optimizer}')
    scheduler = WarmupMultiStepLR(optimizer, 0, **scheduler_config)

    # timer
    timers = SynchronizedWallClockTimer()
    layers_time = {}
    layers_memory = {}
    names = []
    for name, module in model._modules.items():
        names.append(name)
        timers(name).start()
        timers(name).stop()
        timers(name).reset()
        time_ = {
            'forward': 0.0,
            'backward': 0.0
        }
        memory_ = {
            'model': 0.0,
            'inputs': 0.0,
            'out': 0.0
        }
        layers_time[name] = time_
        layers_memory[name] = memory_
    names.reverse()

    for epoch in range(0, args.epochs):
        losses_per_token = AverageMeter(skip_first=False)
        for i, (src, tgt) in enumerate(train_loader, 0):
            # print
            if i % 10 == 0:
                phase = 'TRAIN'
                log = []
                log += [f'{phase} [{epoch}][{i}/{len(train_loader)}]']
                log += [f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})']
                logging.info(log)
            # if i % 100 == 99:
            #     for name in names:
            #         print(name, ':')
            #         print("forward:", layers_time[name]['forward']/i)
            #         print("backward:", layers_time[name]['backward']/i)
            #         print("model_memory:", layers_memory[name]['model']/i)
            #         print("inputs_memory:", layers_memory[name]['inputs']/i)
            #         print("out_memory:", layers_memory[name]['out']/i)
            #     print(' ')

            out_list = []
            inputs_list = []
            src, src_length = src
            tgt, tgt_length = tgt
            src_length = torch.LongTensor(src_length)
            tgt_length = torch.LongTensor(tgt_length)
            src = src.cuda()
            src_length = src_length.cuda()
            tgt = tgt.cuda()
            tgt_input = tgt[:-1]
            tgt_labels = tgt[1:]
            num_toks = int(sum(tgt_length - 1))

            inputs = (src, src_length, tgt_input)

            # forward
            for name, layer in model._modules.items():
                # model parameters
                # model_num = sum(p.numel() for p in layer.parameters())
                # layers_memory[name]['model'] = round(model_num * 4 / (1024 * 1024), 2)
                layers_memory[name]['model'] += get_memory(layer.parameters())

                timers(name).start()
                inputs_list.append(inputs)
                out = layer(inputs)
                out_list.append(out)
                inp = []

                # data parameters
                layers_memory[name]['inputs'] += get_memory(inputs)
                layers_memory[name]['out'] += get_memory(out)

                for tensor in out:
                    out_ = tensor.detach().clone()
                    out_.requires_grad = out_.is_floating_point()
                    inp.append(out_)
                inputs = tuple(inp)
                timers(name).stop()
                layers_time[name]['forward'] += timers(name).elapsed()
                timers(name).reset()

            # backward
            for j in range(0, len(names)):
                timers(names[j]).start()
                if j == 0:
                    out = out_list.pop()
                    T, B = out.size(0), out.size(1)
                    loss = criterion(out.view(T * B, -1),
                                     tgt_labels.contiguous().view(-1))
                    loss_per_token = loss.item() / num_toks
                    loss /= (B * 1)
                    loss.backward()
                else:
                    inputs = inputs_list.pop()
                    out = out_list.pop()
                    grad_tensors = []
                    out_tensors = []
                    for k in range(len(inputs)):
                        if inputs[k].is_floating_point() and inputs[k].grad is not None:
                            grad_tensors.append(inputs[k].grad.detach())
                            out_tensors.append(out[k])
                    assert len(out_tensors) == len(grad_tensors)
                    torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
                timers(names[j]).stop()
                layers_time[names[j]]['backward'] += timers(names[j]).elapsed()
                timers(names[j]).reset()

            losses_per_token.update(loss_per_token, num_toks)
            # update
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

if __name__ == '__main__':
    main()





