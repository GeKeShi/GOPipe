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
import seq2seq.train.trainer as trainers
import seq2seq.utils as utils
from seq2seq.data.dataset import LazyParallelDataset
from seq2seq.data.dataset import ParallelDataset
from seq2seq.data.dataset import TextDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.inference.inference import Translator
from seq2seq.models.gnmt_train import Stage0, Stage1, Stage2, Stage3
from seq2seq.models.gnmt_train import Stage4, Stage5, Stage6, Stage7
from seq2seq.models.sequential_model import embed1, embed2, dropout1, dropout2, dropout3
from seq2seq.models.sequential_model import lstm1, lstm2, lstm3, Attention, Classifier
from seq2seq.models.gnmt_inference import GNMT_v2, GNMT_v4, GNMT_v8
from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.utils import gnmt_print
from collections import OrderedDict

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
    model.add_argument('--num-layers', default=8, type=int,
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
    distributed.add_argument('--partition', default=2, type=int,
                             help='0:layer-nums 1:time 2:weight 3:HIPPIE 4:manual')

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


def main():
    """
    Launches data-parallel multi-gpu training.
    """
    mlperf_log.ROOT_DIR_GNMT = os.path.dirname(os.path.abspath(__file__))
    mlperf_log.LOGGER.propagate = False

    args = parse_args()
    device = utils.set_device(args.cuda, args.local_rank)
    distributed = utils.init_distributed(args.cuda)
    gnmt_print(key=mlperf_log.RUN_START, sync=True)

    rank = utils.get_rank()
    args.rank = rank
    local_rank = args.local_rank
    num_split = args.num_split
    train_num = args.train_num
    group_local_rank = rank % num_split
    num_layers = args.num_layers

    num_worker = torch.distributed.get_world_size() // num_split
    worker_rank = rank // num_split
    worker_in_group = []
    worker_out_group = []
    for i in range(0, num_worker):
        rank_list = []
        # for j in range(0, num_split):
        #     rank_list.append(i * num_split + j)
        if num_split == 2:
            rank_list.append(i * 2)
            rank_list.append(i * 2 + 1)
            group = torch.distributed.new_group(rank_list)
            worker_in_group.append(group)

            # rank_list = []
            # rank_list.append(i * 2)
            # rank_list.append(i * 2 + 1)
            # group = torch.distributed.new_group(rank_list)
            # worker_in_group.append(group)
        if num_split == 4:
            # 0&1, 1&2, 2&3
            for j in range(0, 3):
                rank_list = []
                rank_list.append(i * 4 + j)
                rank_list.append(i * 4 + j + 1)
                group = torch.distributed.new_group(rank_list)
                worker_in_group.append(group)
            # 0 & 3
            rank_list = []
            rank_list.append(i * 4)
            rank_list.append(i * 4 + 3)
            group = torch.distributed.new_group(rank_list)
            worker_in_group.append(group)
        if num_split == 8:
            # 0&1, 1&2, 2&3, 3&4, 4&5, 5&6, 6&7
            for j in range(0, 7):
                rank_list = []
                rank_list.append(i * 8 + j)
                rank_list.append(i * 8 + j + 1)
                group = torch.distributed.new_group(rank_list)
                worker_in_group.append(group)
            # 0 & 7
            rank_list = []
            rank_list.append(i * 8)
            rank_list.append(i * 8 + 7)
            group = torch.distributed.new_group(rank_list)
            worker_in_group.append(group)
    if num_worker > 1:
        for i in range(0, num_split):
            rank_list = []
            for j in range(0, num_worker):
                rank_list.append(i + j * num_split)
            group = torch.distributed.new_group(rank_list)
            worker_out_group.append(group)

    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    # create directory for results
    save_path = os.path.join(args.results_dir, args.save)
    args.save_path = save_path
    os.makedirs(save_path, exist_ok=True)

    # setup logging
    log_filename = f'log_rank_{utils.get_rank()}.log'
    utils.setup_logging(os.path.join(save_path, log_filename))

    if args.env:
        utils.log_env_info()

    logging.info(f'Saving results to: {save_path}')
    logging.info(f'Run arguments: {args}')

    # automatically set train_iter_size based on train_global_batch_size,
    # world_size and per-worker train_batch_size
    # if args.train_global_batch_size is not None:
    #     global_bs = args.train_global_batch_size
    #     bs = args.train_batch_size
    #     world_size = utils.get_world_size()
    #     assert global_bs % (bs * world_size) == 0
    #     args.train_iter_size = global_bs // (bs * world_size)
    #     logging.info(f'Global batch size was set in the config, '
    #                  f'Setting train_iter_size to {args.train_iter_size}')

    worker_seeds, shuffling_seeds = utils.setup_seeds(args.seed, args.epochs,
                                                      device)
    # worker_seed = worker_seeds[worker_rank]
    worker_seed = worker_seeds[0]
    logging.info(f'Worker {rank} is using worker seed: {worker_seed}')
    torch.manual_seed(worker_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    train_len = torch.tensor([0]).cuda(device)
    val_len = torch.tensor([0]).cuda(device)
    test_len = torch.tensor([0]).cuda(device)
    vocab_size_ = torch.tensor([0]).cuda(device)

    batch_first = False
    # stage0
    if group_local_rank == 0:
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
        batch_size = args.train_batch_size * train_num
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

        batch_size = args.val_batch_size * train_num
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

        train_len[0] = len(train_loader)
        val_len[0] = len(val_loader)
        test_len[0] = len(test_loader)
        vocab_size_[0] = vocab_size

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(train_len, 0)
        torch.distributed.broadcast(val_len, 0)
        torch.distributed.broadcast(test_len, 0)
        torch.distributed.broadcast(vocab_size_, 0)
        train_len = train_len[0].item()
        val_len = val_len[0].item()
        test_len = test_len[0].item()
        vocab_size = vocab_size_[0].item()
    assert train_len > 0

    if num_layers == 8:
        if num_split == 2:
            if group_local_rank == 0:
                model = torch.nn.Sequential(OrderedDict([
                    ('Stage0', Stage0()),
                    ('Stage1', Stage1()),
                    ('Stage2', Stage2()),
                    ('Stage3', Stage3()),
                ]))
            else:
                model = torch.nn.Sequential(OrderedDict([
                    ('Stage4', Stage4(False)),
                    ('Stage5', Stage5()),
                    ('Stage6', Stage6()),
                    ('Stage7', Stage7()),
                ]))
        elif num_split == 4:
            if group_local_rank == 0:
                if args.partition == 0:
                    model = torch.nn.Sequential(OrderedDict([
                        ('Emb1', embed1()),
                        ('E_lstm1', lstm1(bi=True, residual=False)),
                        ('Dropout1', dropout1()),
                        ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
                        ('Dropout2', dropout1()),
                        ('E_lstm3', lstm1()),
                        ('Dropout3', dropout1()),
                        ('E_lstm4', lstm1()),
                        ('Dropout4', dropout1()),
                    ]))
                if args.partition == 2:
                    model = torch.nn.Sequential(OrderedDict([
                        ('Emb1', embed1()),
                        ('E_lstm1', lstm1(bi=True, residual=False)),
                        ('Dropout1', dropout1()),
                        ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
                        ('Dropout2', dropout1()),
                        ('E_lstm3', lstm1()),
                        ('Dropout3', dropout1()),
                        ('E_lstm4', lstm1()),
                        ('Dropout4', dropout1()),
                    ]))
                if args.partition == 3:
                    model = torch.nn.Sequential(OrderedDict([
                        ('Emb1', embed1()),
                        ('E_lstm1', lstm1(bi=True, residual=False)),
                        ('Dropout1', dropout1()),
                        ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
                        ('Dropout2', dropout1()),

                    ]))
                if args.partition == 4:
                    # best
                    model = torch.nn.Sequential(OrderedDict([
                        ('Emb1', embed1()),
                        ('E_lstm1', lstm1(bi=True, residual=False)),
                        ('Dropout1', dropout1()),
                        ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
                        ('Dropout2', dropout1()),
                        ('E_lstm3', lstm1()),
                        ('Dropout3', dropout1()),
                        ('E_lstm4', lstm1()),
                    ]))
            if group_local_rank == 1:
                if args.partition == 0:
                    model = torch.nn.Sequential(OrderedDict([
                        ('E_lstm5', lstm1()),
                        ('Dropout5', dropout1()),
                        ('E_lstm6', lstm1()),
                        ('Dropout6', dropout1()),
                        ('E_lstm7', lstm1()),
                        ('Dropout7', dropout1()),
                        ('E_lstm8', lstm1()),

                        ('Emb2', embed2()),
                        ('Dropout8', dropout2()),
                    ]))
                if args.partition == 2:
                    model = torch.nn.Sequential(OrderedDict([
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
                        ('Attention', Attention(True)),
                    ]))
                if args.partition == 3:
                    model = torch.nn.Sequential(OrderedDict([
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
                    ]))
                if args.partition == 4:
                    model = torch.nn.Sequential(OrderedDict([
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
                    ]))
            if group_local_rank == 2:
                if args.partition == 0:
                    model = torch.nn.Sequential(OrderedDict([
                        ('D_lstm1', lstm2()),
                        ('Attention', Attention()),
                        ('Dropout9', dropout3()),
                        ('D_lstm2', lstm3(residual=False)),
                        ('Dropout10', dropout3()),
                        ('D_lstm3', lstm3()),
                        ('Dropout11', dropout3()),
                        ('D_lstm4', lstm3()),
                        ('Dropout12', dropout3()),
                    ]))
                if args.partition == 2:
                    model = torch.nn.Sequential(OrderedDict([
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
                    ]))
                if args.partition == 3:
                    model = torch.nn.Sequential(OrderedDict([
                        ('Emb2', embed2()),
                        ('Dropout8', dropout2()),

                        ('D_lstm1', lstm2()),
                        ('Attention', Attention(True)),
                        # ('Attention', Attention()),
                        ('Dropout9', dropout3()),
                        ('D_lstm2', lstm3(residual=False)),
                        ('Dropout10', dropout3()),
                        ('D_lstm3', lstm3()),
                        ('Dropout11', dropout3()),
                        ('D_lstm4', lstm3()),
                    ]))
                if args.partition == 4:
                    model = torch.nn.Sequential(OrderedDict([
                        ('Attention', Attention(True)),
                        # ('Attention', Attention()),
                        ('Dropout9', dropout3()),
                        ('D_lstm2', lstm3(residual=False)),
                        ('Dropout10', dropout3()),
                        ('D_lstm3', lstm3()),
                        ('Dropout11', dropout3()),
                        ('D_lstm4', lstm3()),
                        ('Dropout12', dropout3()),
                        ('D_lstm5', lstm3()),
                    ]))
            if group_local_rank == 3:
                if args.partition == 0:
                    model = torch.nn.Sequential(OrderedDict([
                        ('D_lstm5', lstm3()),
                        ('Dropout13', dropout3()),
                        ('D_lstm6', lstm3()),
                        ('Dropout14', dropout3()),
                        ('D_lstm7', lstm3()),
                        ('Dropout15', dropout3()),
                        ('D_lstm8', lstm3(last=True)),
                        ('Classifier', Classifier(1024, 32320))
                    ]))
                if args.partition == 2:
                    model = torch.nn.Sequential(OrderedDict([
                        ('D_lstm7', lstm3()),
                        ('Dropout15', dropout3()),
                        ('D_lstm8', lstm3(last=True)),
                        ('Classifier', Classifier(1024, 32320))
                    ]))
                if args.partition == 3:
                    model = torch.nn.Sequential(OrderedDict([
                        ('Dropout12', dropout3()),
                        ('D_lstm5', lstm3()),
                        ('Dropout13', dropout3()),
                        ('D_lstm6', lstm3()),
                        ('Dropout14', dropout3()),
                        ('D_lstm7', lstm3()),
                        ('Dropout15', dropout3()),
                        ('D_lstm8', lstm3(last=True)),
                        ('Classifier', Classifier(1024, 32320))
                    ]))
                if args.partition == 4:
                    model = torch.nn.Sequential(OrderedDict([
                        ('Dropout13', dropout3()),
                        ('D_lstm6', lstm3()),
                        ('Dropout14', dropout3()),
                        ('D_lstm7', lstm3()),
                        ('Dropout15', dropout3()),
                        ('D_lstm8', lstm3(last=True)),
                        ('Classifier', Classifier(1024, 32320))
                    ]))
        elif num_split == 8:
            model_name = 'Stage' + str(group_local_rank)
            if args.Gpipe and group_local_rank == 4:
                model = eval(model_name)(False)
            else:
                model = eval(model_name)()
    if num_layers == 4:
        # num_split = 4
        if group_local_rank == 0:
            if args.partition == 0:
                model = torch.nn.Sequential(OrderedDict([
                    ('Emb1', embed1()),
                    ('E_lstm1', lstm1(bi=True, residual=False)),
                    ('Dropout1', dropout1()),
                    ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
                    ('Dropout2', dropout1()),
                ]))
            if args.partition == 2:
                model = torch.nn.Sequential(OrderedDict([
                    ('Emb1', embed1()),
                ]))
            if args.partition == 3:
                model = torch.nn.Sequential(OrderedDict([
                    ('Emb1', embed1()),
                    ('E_lstm1', lstm1(bi=True, residual=False)),
                ]))
            if args.partition == 4:
                model = torch.nn.Sequential(OrderedDict([
                    ('Emb1', embed1()),
                    ('E_lstm1', lstm1(bi=True, residual=False)),
                    ('Dropout1', dropout1()),
                    ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
                    ('Dropout2', dropout1()),
                    ('E_lstm3', lstm1()),
                ]))
        if group_local_rank == 1:
            if args.partition == 0:
                model = torch.nn.Sequential(OrderedDict([
                    ('E_lstm3', lstm1()),
                    ('Dropout3', dropout1()),
                    ('E_lstm4', lstm1()),

                    ('Emb2', embed2()),
                    ('Dropout4', dropout2()),
                ]))
            if args.partition == 2:
                model = torch.nn.Sequential(OrderedDict([
                    ('E_lstm1', lstm1(bi=True, residual=False)),
                    ('Dropout1', dropout1()),
                    ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
                    ('Dropout2', dropout1()),
                    ('E_lstm3', lstm1()),
                    ('Dropout3', dropout1()),
                    ('E_lstm4', lstm1()),
                ]))
            if args.partition == 3:
                model = torch.nn.Sequential(OrderedDict([
                    ('Dropout1', dropout1()),
                    ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
                    ('Dropout2', dropout1()),
                    ('E_lstm3', lstm1()),
                    ('Dropout3', dropout1()),
                    ('E_lstm4', lstm1()),
                ]))
            if args.partition == 4:
                model = torch.nn.Sequential(OrderedDict([
                    ('Dropout3', dropout1()),
                    ('E_lstm4', lstm1()),

                    ('Emb2', embed2()),
                    ('Dropout4', dropout2()),
                    ('D_lstm1', lstm2()),
                ]))
        if group_local_rank == 2:
            if args.partition == 0:
                model = torch.nn.Sequential(OrderedDict([
                    ('D_lstm1', lstm2()),
                    ('Attention', Attention()),
                    ('Dropout5', dropout3()),
                    ('D_lstm2', lstm3(residual=False)),
                    ('Dropout6', dropout3()),
                ]))
            if args.partition == 2:
                model = torch.nn.Sequential(OrderedDict([
                    ('Emb2', embed2()),
                    ('Dropout4', dropout2()),
                    ('D_lstm1', lstm2()),
                    ('Attention', Attention()),
                    ('Dropout5', dropout3()),
                    ('D_lstm2', lstm3(residual=False)),
                    ('Dropout6', dropout3()),
                ]))
            if args.partition == 3:
                model = torch.nn.Sequential(OrderedDict([
                    ('Emb2', embed2()),
                    ('Dropout4', dropout2()),
                    ('D_lstm1', lstm2()),
                    ('Attention', Attention(True)),
                    ('Dropout5', dropout3()),
                    ('D_lstm2', lstm3(residual=False)),
                    ('Dropout6', dropout3()),
                ]))
            if args.partition == 4:
                model = torch.nn.Sequential(OrderedDict([
                    ('Attention', Attention(True)),
                    ('Dropout5', dropout3()),
                    ('D_lstm2', lstm3(residual=False)),
                    ('Dropout6', dropout3()),
                    ('D_lstm3', lstm3()),
                ]))
        if group_local_rank == 3:
            if args.partition == 0:
                model = torch.nn.Sequential(OrderedDict([
                    ('D_lstm3', lstm3()),
                    ('Dropout7', dropout3()),
                    ('D_lstm4', lstm3(last=True)),
                    ('Classifier', Classifier(1024, 32320))
                ]))
            if args.partition == 2:
                model = torch.nn.Sequential(OrderedDict([
                    ('D_lstm3', lstm3()),
                    ('Dropout7', dropout3()),
                    ('D_lstm4', lstm3(last=True)),
                    ('Classifier', Classifier(1024, 32320))
                ]))
            if args.partition == 3:
                model = torch.nn.Sequential(OrderedDict([
                    ('D_lstm3', lstm3()),
                    ('Dropout7', dropout3()),
                    ('D_lstm4', lstm3(last=True)),
                    ('Classifier', Classifier(1024, 32320))
                ]))
            if args.partition == 4:
                model = torch.nn.Sequential(OrderedDict([
                    ('Dropout7', dropout3()),
                    ('D_lstm4', lstm3(last=True)),
                    ('Classifier', Classifier(1024, 32320))
                ]))

    if (rank + 1) % num_split == 0:
        criterion = build_criterion(vocab_size, config.PAD, args.smoothing)
        criterion = criterion.cuda()

    logging.info(model)

    # define and optimizer
    opt_config = {'optimizer': args.optimizer, 'lr': args.lr}
    opt_config.update(literal_eval(args.optimizer_extra))
    logging.info(f'Training optimizer config: {opt_config}')

    scheduler_config = {'warmup_steps': args.warmup_steps,
                        'remain_steps': args.remain_steps,
                        'decay_interval': args.decay_interval,
                        'decay_steps': args.decay_steps,
                        'decay_factor': args.decay_factor}

    logging.info(f'Training LR schedule config: {scheduler_config}')

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info(f'Number of stage{local_rank} parameters: {num_parameters}')

    save_info = {}
    if group_local_rank == 0:
        if num_split == 2:
            model_infer = GNMT_v2()
        if num_split == 4:
            model_infer = GNMT_v4()
        if num_split == 8:
            model_infer = GNMT_v8()
        if num_worker > 1:
            infer_group = worker_out_group[0]
        else:
            infer_group = None
        translator = Translator(model=model_infer,
                                tokenizer=tokenizer,
                                loader=test_loader,
                                beam_size=args.beam_size,
                                max_seq_len=args.max_length_test,
                                len_norm_factor=args.len_norm_factor,
                                len_norm_const=args.len_norm_const,
                                cov_penalty_factor=args.cov_penalty_factor,
                                cuda=args.cuda,
                                print_freq=args.print_freq,
                                dataset_dir=args.dataset_dir,
                                target_bleu=args.target_bleu,
                                save_path=args.save_path,
                                num_worker=num_worker,
                                worker_group=infer_group)

    # create stage-x trainer
    total_train_iters = train_len * args.epochs
    # save_info['model_config'] = model_config
    # save_info['config'] = args
    save_info = {}
    trainer_options = dict(
        worker_out_group=None,
        local_rank=local_rank,
        group_local_rank=group_local_rank,
        worker_num=num_worker,
        stage_num=num_split,
        train_num=train_num,
        grad_clip=args.grad_clip,
        iter_size=args.train_iter_size,
        save_path=save_path,
        save_freq=args.save_freq,
        save_info=save_info,
        opt_config=opt_config,
        scheduler_config=scheduler_config,
        train_iterations=total_train_iters,
        batch_first=batch_first,
        keep_checkpoints=args.keep_checkpoints,
        math=args.math,
        print_freq=args.print_freq,
        intra_epoch_eval=args.intra_epoch_eval,
        translator=None)

    worker_group = []
    for i in range(0, num_worker):
        rank_list = []
        for j in range(0, num_split):
            rank_list.append(i * num_split + j)
        worker_group.append(torch.distributed.new_group(rank_list))

    if group_local_rank == 0:
        # trainer_options['group_extra'] = group_extra[worker_rank]
        trainer_options['translator'] = translator
        trainer_options['worker_in_group_up'] = worker_in_group[rank + num_split - 1]
        trainer_options['worker_in_group_down'] = worker_in_group[rank]
    else:
        trainer_options['worker_in_group_up'] = worker_in_group[rank - 1]
        trainer_options['worker_in_group_down'] = worker_in_group[rank]

    # if group_local_rank == 2 or group_local_rank == 3 or group_local_rank == 7:
        # trainer_options['group_extra'] = group_extra[worker_rank]

    if group_local_rank == num_split - 1:
        trainer_options['criterion'] = criterion

    if num_worker > 1:
        trainer_options['worker_out_group'] = worker_out_group[group_local_rank]
    trainer_options['model'] = model
    trainer_options['worker_group'] = worker_group[worker_rank]
    trainer_options['batch_size'] = args.train_batch_size
    trainer = trainers.Pipeline_Trainer(**trainer_options)

    # optionally resume from a checkpoint
    # if args.resume:
    #     checkpoint_file = args.resume
    #     if os.path.isdir(checkpoint_file):
    #         checkpoint_file = os.path.join(
    #             checkpoint_file, 'model_best.pth')
    #     if os.path.isfile(checkpoint_file):
    #         trainer.load(checkpoint_file)
    #     else:
    #         logging.error(f'No checkpoint found at {args.resume}')

    # training loop
    best_loss = float('inf')
    break_training = False
    test_bleu = None
    gnmt_print(key=mlperf_log.TRAIN_LOOP, sync=True)
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f'Starting epoch {epoch}')
        gnmt_print(key=mlperf_log.TRAIN_EPOCH,
                   value=epoch, sync=True)

        trainer.epoch = epoch
        trainer.batch_size = args.train_batch_size
        if group_local_rank == 0:
            train_loader.sampler.set_epoch(epoch)
            # train_loss, train_perf = trainer.optimize(train_loader)
            train_out = trainer.optimize(train_loader)
        else:
            trainer.optimize(train_len)

        # evaluate on validation set
        # if args.eval:
        #     logging.info(f'Running validation on dev set')
        #     trainer.batch_size = args.val_batch_size
        #
        #     if group_local_rank == 0:
        #         val_out = trainer.evaluate(val_loader)
        #     else:
        #         trainer.evaluate(val_len)

        if args.eval:
            gnmt_print(key=mlperf_log.EVAL_START, value=epoch, sync=True)
            if rank < num_split:
                trainer.save_for_infer()
            trainer.barrier()
            if group_local_rank == 0:
                trainer.load_for_infer()
                test_bleu, break_training = translator.run(calc_bleu=True,
                                                           epoch=epoch)
                gnmt_print(key=mlperf_log.EVAL_ACCURACY,
                           value={"epoch": epoch, "value": round(test_bleu, 2)},
                           sync=False)
                gnmt_print(key=mlperf_log.EVAL_TARGET,
                           value=args.target_bleu, sync=False)

                acc_log = []
                acc_log += [f'Summary: Epoch: {epoch}']
                if args.eval:
                    acc_log += [f'Test BLEU: {test_bleu:.2f}']

                perf_log = []
                perf_log += [f'Performance: Epoch: {epoch}']
                perf_log += [f'Training: {train_out:.0f} Tok/s']
                # if args.eval:
                #     perf_log += [f'Validation: {val_out:.0f} Tok/s']

                if rank == 0:
                    logging.info('\t'.join(acc_log))
                    logging.info('\t'.join(perf_log))

                logging.info(f'Finished epoch {epoch}')
            else:
                torch.cuda.empty_cache()
                test_bleu = torch.cuda.FloatTensor([0])
                break_training = torch.cuda.LongTensor([0])
                dist.broadcast(break_training, 0)
                dist.broadcast(test_bleu, 0)
            gnmt_print(key=mlperf_log.EVAL_STOP, sync=True)
        if break_training:
            break

    gnmt_print(key=mlperf_log.RUN_STOP,
               value={"success": bool(break_training)}, sync=True)
    gnmt_print(key=mlperf_log.RUN_FINAL, sync=False)


if __name__ == '__main__':
    main()
