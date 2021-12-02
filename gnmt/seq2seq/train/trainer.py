import logging
import os
import time
from itertools import cycle
import copy

import numpy as np
import torch
import torch.optim
import torch.utils.data
from mlperf_compliance import mlperf_log
import torch.distributed as dist
import threading

from seq2seq.train.fp_optimizers import Fp16Optimizer
from seq2seq.train.fp_optimizers import Fp32Optimizer
from seq2seq.train.lr_scheduler import WarmupMultiStepLR
from seq2seq.utils import AverageMeter
from seq2seq.utils import gnmt_print
from seq2seq.utils import sync_workers
from torch.nn.utils import clip_grad_norm_
from torch._six import inf


class Pipeline_Trainer:
    """
    Seq2SeqTrainer
    """

    def __init__(self,
                 model,
                 opt_config,
                 scheduler_config,
                 print_freq=10,
                 save_freq=1000,
                 grad_clip=float('inf'),
                 batch_first=False,
                 save_info={},
                 save_path='.',
                 train_iterations=0,
                 checkpoint_filename='checkpoint%s.pth',
                 keep_checkpoints=5,
                 math='fp32',
                 intra_epoch_eval=0,
                 iter_size=1,
                 translator=None,
                 criterion=None,
                 group_local_rank=0,
                 local_rank=0,
                 train_num=4,
                 stage_num=4,
                 worker_group=None,
                 worker_in_group_up=None,
                 worker_in_group_down=None,
                 worker_out_group=None,
                 group_extra=None,
                 gpipe=False,
                 batch_size=64,
                 worker_num=1):
        """
        Constructor for the Seq2SeqTrainer.

        :param model: model to train
        :param criterion: criterion (loss function)
        :param opt_config: dictionary with options for the optimizer
        :param scheduler_config: dictionary with options for the learning rate
            scheduler
        :param print_freq: prints short summary every 'print_freq' iterations
        :param save_freq: saves checkpoint every 'save_freq' iterations
        :param grad_clip: coefficient for gradient clipping
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param save_info: dict with additional state stored in each checkpoint
        :param save_path: path to the directiory for checkpoints
        :param train_iterations: total number of training iterations to execute
        :param checkpoint_filename: name of files with checkpoints
        :param keep_checkpoints: max number of checkpoints to keep
        :param math: arithmetic type
        :param cuda: if True use cuda, if False train on cpu
        :param distributed: if True run distributed training
        :param intra_epoch_eval: number of additional eval runs within each
            training epoch
        :param iter_size: number of iterations between weight updates
        :param translator: instance of Translator, runs inference on test set
        :param verbose: enables verbose logging
        """
        super(Pipeline_Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.local_rank = local_rank
        self.group_local_rank = group_local_rank
        self.epoch = 0
        self.loss = None
        self.save_info = save_info
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_counter = 0
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_counter = cycle(range(keep_checkpoints))
        self.opt_config = opt_config
        self.print_freq = print_freq
        self.batch_first = batch_first
        self.translator = translator
        self.intra_epoch_eval = intra_epoch_eval
        self.iter_size = iter_size
        self.grad_clip = grad_clip
        self.batch_size = 1
        self.Grad = []
        self.train_num = train_num
        self.worker_group = worker_group
        self.worker_in_group_up = worker_in_group_up
        self.worker_in_group_down = worker_in_group_down
        self.worker_out_group = worker_out_group
        self.group_extra = group_extra
        self.worker_num = worker_num
        self.stage_num = stage_num
        self.rank = dist.get_rank()
        self.batch_size = batch_size
        self.out_list = []
        self.grad_up = None
        self.gpipe = gpipe

        self.handle = None
        self.shape = []
        self.split_size = []
        self.grad_flatten = None
        self.allreduce_num = 0
        self.allreduce_block = 5

        self.src_split = []
        self.src_length_split = []
        self.tgt_split = []
        self.num_toks = {'tgt': 0, 'src': 0}

        self.model = self.model.cuda()

        if math == 'fp16':
            self.model = self.model.half()

        params = self.model.parameters()

        opt_name = opt_config.pop('optimizer')
        self.optimizer = torch.optim.__dict__[opt_name](params, **opt_config)
        logging.info(f'Using optimizer: {self.optimizer}')
        gnmt_print(key=mlperf_log.OPT_NAME,
                   value=mlperf_log.ADAM, sync=False)
        gnmt_print(key=mlperf_log.OPT_LR,
                   value=opt_config['lr'], sync=False)
        gnmt_print(key=mlperf_log.OPT_HP_ADAM_BETA1,
                   value=self.optimizer.defaults['betas'][0], sync=False)
        gnmt_print(key=mlperf_log.OPT_HP_ADAM_BETA2,
                   value=self.optimizer.defaults['betas'][1], sync=False)
        gnmt_print(key=mlperf_log.OPT_HP_ADAM_EPSILON,
                   value=self.optimizer.defaults['eps'], sync=False)

        self.scheduler = WarmupMultiStepLR(self.optimizer, train_iterations,
                                           **scheduler_config)

        for p in self.model.parameters():
            if p.requires_grad:
                self.get_shape(p)
                if self.group_local_rank < stage_num // 2 and self.worker_num > 1:
                    self.Grad.append(torch.zeros(p.shape).cuda())

    def get_shape(self, p):
        shape_ = []
        dim = len(p.shape)
        num = 1
        for k in range(dim):
            num = num * p.shape[k]
            shape_.append(p.shape[k])
        self.split_size.append(num)
        self.shape.append(shape_)

    def grad_hook(self, grad):
        dist.broadcast(grad.detach(), self.rank,
                       group=self.worker_in_group_up)

    def barrier(self, group=None):
        if group is None:
            torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        else:
            torch.distributed.all_reduce(torch.cuda.FloatTensor(1), group=group)
        torch.cuda.synchronize()

    # grad average2 wait()
    def all_reduce_0(self):
        if self.worker_num > 1:
            for k in range(len(self.Grad)):
                self.Grad[k] = self.Grad[k].flatten()
            self.Grad = torch.cat(self.Grad)
            self.Grad = torch.true_divide(self.Grad, self.train_num * self.worker_num)
            torch.cuda.synchronize()
            self.handle = dist.all_reduce(self.Grad, op=dist.ReduceOp.SUM,
                                          group=self.worker_out_group, async_op=True)

    # grad average2
    def all_reduce_1(self):
        if self.worker_num > 1:
            self.Grad = []
            for k, params in enumerate(self.model.parameters()):
                self.Grad.append(params.grad.detach().clone().flatten())
            self.Grad = torch.cat(self.Grad)
            self.Grad = torch.true_divide(self.Grad, self.train_num * self.worker_num)
            dist.all_reduce(self.Grad, op=dist.ReduceOp.SUM, group=self.worker_out_group)
            torch.cuda.synchronize()
            # XXX
            self.Grad = list(self.Grad.split(self.split_size))
            for k, params in enumerate(self.model.parameters()):
                if params.requires_grad:
                    self.Grad[k] = self.Grad[k].resize_(self.shape[k])
                    params.grad = self.Grad[k]

    # wait()
    def update_0(self):
        if self.worker_num > 1:
            self.handle.wait()
            # XXX
            self.Grad = list(self.Grad.split(self.split_size))
            for k, params in enumerate(self.model.parameters()):
                if params.requires_grad:
                    self.Grad[k] = self.Grad[k].resize_(self.shape[k])
                    grad = params.grad.detach().clone()
                    params.grad = self.Grad[k]
                    self.Grad[k] = grad
        if self.worker_num == 1:
            for params in self.model.parameters():
                if params.grad is not None:
                    params.grad.data = torch.true_divide(params.grad.data, self.train_num)

        if self.grad_clip != float('inf'):
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()

    def update_1(self):
        if self.worker_num == 1:
            for params in self.model.parameters():
                if params.grad is not None:
                    params.grad.data = torch.true_divide(params.grad.data, self.train_num)

        if self.grad_clip != float('inf'):
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()

    def pipeline_send(self, out):
        tensors_shape = torch.zeros((6, 5)).int().cuda()
        tensors_shape[0][0] = len(out)
        for i in range(len(out)):
            tensor = out[i]
            tensors_shape[i+1][0] = len(tensor.shape)
            if tensor.dtype == torch.float32:
                tensors_shape[i + 1][1] = 0
            if tensor.dtype == torch.int64:
                tensors_shape[i + 1][1] = 1
            for j in range(len(tensor.shape)):
                tensors_shape[i+1][j+2] = tensor.shape[j]
        dist.broadcast(tensors_shape, self.rank, group=self.worker_in_group_down)
        torch.cuda.synchronize()
        for i in range(len(out)):
            p = out[i].detach().clone()
            dist.broadcast(p, self.rank, group=self.worker_in_group_down)
            torch.cuda.synchronize()

    # flatten()
    # def pipeline_send(self, out):
    #     tensors_shape = torch.zeros((6, 5)).int().cuda()
    #     tensors_shape[0][0] = len(out)
    #     tensors = []
    #     for i in range(len(out)):
    #         tensor = out[i]
    #         tensors_shape[i+1][0] = len(tensor.shape)
    #         if tensor.dtype == torch.float32:
    #             tensors_shape[i + 1][1] = 0
    #         if tensor.dtype == torch.int64:
    #             tensors_shape[i + 1][1] = 1
    #         for j in range(len(tensor.shape)):
    #             tensors_shape[i+1][j+2] = tensor.shape[j]
    #         tensors.append(out[i].detach().clone().flatten().float())
    #     tensors = torch.cat(tensors)
    #     tensors_shape[0][1] = tensors.shape[0]
    #     dist.broadcast(tensors_shape, self.rank, group=self.worker_in_group_down)
    #     dist.broadcast(tensors, self.rank, group=self.worker_in_group_down)
    #     torch.cuda.synchronize()

    def pipeline_recv(self):
        tensors_shape = torch.zeros((6, 5)).int().cuda()
        dist.broadcast(tensors_shape, self.rank - 1, group=self.worker_in_group_up)
        inp = []
        for i in range(tensors_shape[0][0].item()):
            shape = []
            for j in range(tensors_shape[i+1][0].item()):
                shape.append(tensors_shape[i+1][j+2].item())
            flag = tensors_shape[i+1][1].item()
            if flag == 0:
                dtype = torch.float32
            if flag == 1:
                dtype = torch.int64
            recv = torch.zeros(shape, dtype=dtype).cuda()
            dist.broadcast(recv, self.rank - 1, group=self.worker_in_group_up)
            torch.cuda.synchronize()
            if recv.is_floating_point():
                recv.requires_grad = True
                recv.retain_grad()
            inp.append(recv)
        return tuple(inp)

    # flatten()
    # def pipeline_recv(self):
    #     tensors_shape = torch.zeros((6, 5)).int().cuda()
    #     dist.broadcast(tensors_shape, self.rank - 1, group=self.worker_in_group_up)
    #     inps = []
    #     recv = torch.zeros(tensors_shape[0][1].item()).cuda()
    #     dist.broadcast(recv, self.rank - 1, group=self.worker_in_group_up)
    #     torch.cuda.synchronize()
    #     shapes = []
    #     split_size = []
    #     dtype = []
    #     for i in range(tensors_shape[0][0].item()):
    #         shape_ = []
    #         length = 1
    #         for j in range(tensors_shape[i+1][0].item()):
    #             length = length * tensors_shape[i+1][j+2].item()
    #             shape_.append(tensors_shape[i+1][j+2].item())
    #         shapes.append(shape_)
    #         split_size.append(length)
    #         flag = tensors_shape[i+1][1].item()
    #         if flag == 0:
    #             dtype.append(torch.float32)
    #         if flag == 1:
    #             dtype.append(torch.int64)
    #     recv = list(recv.split(split_size))
    #     for k, inp in enumerate(recv):
    #         inp = inp.resize_(shapes[k]).type(dtype[k])
    #         if inp.is_floating_point():
    #             inp.requires_grad = True
    #             inp.retain_grad()
    #         inps.append(inp)
    #     return tuple(inps)

    def grad_send(self, inp):
        position = torch.zeros(6).int().cuda()
        for k in range(len(inp)):
            if inp[k].is_floating_point() and inp[k].grad is not None:
                position[k] = 1
        dist.broadcast(position, self.rank, group=self.worker_in_group_up)
        for k in range(len(inp)):
            if inp[k].is_floating_point() and inp[k].grad is not None:
                dist.broadcast(inp[k].grad.detach(), self.rank, group=self.worker_in_group_up)

    # flatten()
    # def grad_send(self, inp):
    #     position = torch.zeros(6).int().cuda()
    #     tensors = []
    #     for k in range(len(inp)):
    #         if inp[k].is_floating_point() and inp[k].grad is not None:
    #             position[k+1] = 1
    #             tensors.append(inp[k].flatten())
    #     tensors = torch.cat(tensors)
    #     position[0] = tensors.shape[0]
    #     dist.broadcast(position, self.rank, group=self.worker_in_group_up)
    #     dist.broadcast(tensors, self.rank, group=self.worker_in_group_up)

    def grad_recv(self, out):
        grad_tensors = []
        out_tensors = []
        position = torch.zeros(6).int().cuda()
        dist.broadcast(position, self.rank + 1, group=self.worker_in_group_down)
        for k in range(len(out)):
            if position[k].item() == 1:
                grad = torch.zeros_like(out[k]).cuda()
                dist.broadcast(grad, self.rank + 1, group=self.worker_in_group_down)
                torch.cuda.synchronize()
                grad_tensors.append(grad)
                out_tensors.append(out[k])
        assert len(out_tensors) == len(grad_tensors)
        return out_tensors, grad_tensors

    # flatten()
    # def grad_recv(self, out):
    #     grad_tensors = []
    #     out_tensors = []
    #     position = torch.zeros(6).int().cuda()
    #     dist.broadcast(position, self.rank + 1, group=self.worker_in_group_down)
    #     recv = torch.zeros(position[0]).cuda()
    #     dist.broadcast(recv, self.rank + 1, group=self.worker_in_group_down)
    #     torch.cuda.synchronize()
    #     split_size = []
    #     shapes = []
    #     for k in range(len(out)):
    #         if position[k+1].item() == 1:
    #             split_size.append(out[k].numel())
    #             shapes.append(out[k].shape)
    #             out_tensors.append(out[k])
    #     recv = list(recv.split(split_size))
    #     for k, grad in enumerate(recv):
    #         grad = grad.resize_(shapes[k])
    #         grad_tensors.append(grad)
    #     assert len(out_tensors) == len(grad_tensors)
    #     return out_tensors, grad_tensors

    def iterate_stage0(self, src, tgt, update=True, training=True):
        """
        Performs one iteration of the training/validation.

        :param src: batch of examples from the source language
        :param tgt: batch of examples from the target language
        :param update: if True: optimizer does update of the weights
        :param training: if True: executes optimizer
        """
        src, src_length = src
        tgt, tgt_length = tgt
        src_length = torch.LongTensor(src_length)
        tgt_length = torch.LongTensor(tgt_length)

        num_toks = {}
        num_toks['tgt'] = int(sum(tgt_length - 1))
        num_toks['src'] = int(sum(src_length))

        src = src.cuda()
        src_length = src_length.cuda()
        tgt = tgt.cuda()
        tgt_input = tgt[:-1]
        tgt_labels = tgt[1:]

        src_split = src.chunk(self.train_num, dim=1)
        src_length_split = src_length.chunk(self.train_num, dim=0)
        tgt_input_split = tgt_input.chunk(self.train_num, dim=1)

        # comm
        head = torch.tensor([tgt_labels.shape[0], tgt_labels.shape[1], num_toks['tgt']]).int().cuda()
        dist.broadcast(head, self.rank, group=self.worker_in_group_up)
        dist.broadcast(tgt_labels, self.rank, group=self.worker_in_group_up)
        self.out_list = []

        for k in range(0, self.train_num):
            # forward
            src = src_split[k]
            src_length = src_length_split[k]
            tgt_input = tgt_input_split[k]
            out = self.model((src, src_length, tgt_input))
            torch.cuda.synchronize()
            if training:
                self.out_list.append(out)

            # XXX wait()
            if k == self.stage_num - 1 and training and update:
                if training and update:
                    self.all_reduce_0()

            if k >= self.stage_num - 1 and training:
                # grad_recv
                out_back = self.out_list.pop(0)
                out_tensors, grad_tensors = self.grad_recv(out_back)
                # pipeline_send
                self.pipeline_send(out)
                # backward
                torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
                torch.cuda.synchronize()

            else:
                # pipeline_send
                self.pipeline_send(out)

        # XXX
        # torch.cuda.empty_cache()
        # print("rank:", self.rank, torch.cuda.memory_allocated())

        # backward
        for k in range(0, self.stage_num - 1):
            # grad_recv
            out_back = self.out_list.pop(0)
            out_tensors, grad_tensors = self.grad_recv(out_back)
            # backward
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
            torch.cuda.synchronize()

        # update
        if update:
            self.update_0()
            # self.all_reduce_1()
            # self.update_1()
        return num_toks

    def iterate_rest(self, update=True, training=True):
        self.out_list = []
        self.inp_list = []
        for k in range(0, self.train_num):
            # pipeline_recv
            inp = self.pipeline_recv()
            out = self.model(inp)
            torch.cuda.synchronize()
            if training:
                self.inp_list.append(inp)
                self.out_list.append(out)

            # XXX wait()
            if k == self.stage_num - self.group_local_rank - 1:
                if training and update and self.group_local_rank < self.stage_num // 2 and self.worker_num > 1:
                    self.all_reduce_0()

            if k >= self.stage_num - self.group_local_rank - 1 and training:
                # grad_recv
                out_back = self.out_list.pop(0)
                out_tensors, grad_tensors = self.grad_recv(out_back)
                # pipeline_send
                self.pipeline_send(out)
                # backward
                torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
                torch.cuda.synchronize()
                # grad_send
                inp_back = self.inp_list.pop(0)
                self.grad_send(inp_back)
            else:
                # pipeline_send
                self.pipeline_send(out)

        # XXX
        # torch.cuda.empty_cache()
        # print("rank:", self.rank, torch.cuda.memory_allocated())

        # backward
        for k in range(0, self.stage_num - self.group_local_rank - 1):
            # grad_recv
            out_back = self.out_list.pop(0)
            out_tensors, grad_tensors = self.grad_recv(out_back)
            # backward
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
            torch.cuda.synchronize()
            # grad_send
            inp_back = self.inp_list.pop(0)
            self.grad_send(inp_back)
        # update
        if update:
            if self.group_local_rank < self.stage_num // 2 and self.worker_num > 1:
                self.update_0()
            else:
                self.all_reduce_1()
                self.update_1()

    def iterate_last(self, update=True, training=True):
        head = torch.zeros(3).int().cuda()
        dist.broadcast(head, self.rank - self.group_local_rank, group=self.worker_in_group_down)
        tgt_labels = torch.zeros((head[0].item(), head[1].item())).long().cuda()
        dist.broadcast(tgt_labels, self.rank - self.group_local_rank,
                       group=self.worker_in_group_down)
        tgt_labels_split = tgt_labels.chunk(self.train_num, dim=1)
        loss_per_batch = 0
        num_toks = head[2].item()

        for k in range(0, self.train_num):
            # forward
            tgt_label = tgt_labels_split[k]
            # pipeline_recv
            inp = self.pipeline_recv()
            out = self.model(inp)
            T, B = out.size(0), out.size(1)
            loss = self.criterion(out.view(T * B, -1),
                                  tgt_label.contiguous().view(-1))
            loss_per_batch += loss.item()
            loss /= (B * 1)

            # XXX
            # torch.cuda.empty_cache()
            # print("rank:", self.rank, torch.cuda.memory_allocated())

            # backward
            if training:
                loss.backward()
                # grad_send
                self.grad_send(inp)
                torch.cuda.synchronize()
        if training and update:
            self.all_reduce_1()
            self.update_1()
        loss_per_token = loss_per_batch / num_toks
        loss_per_sentence = loss_per_batch / B
        return loss_per_token, loss_per_sentence, num_toks

    def feed_data(self, data_loader, training=True):
        """
        Runs training or validation on batches from data_loader.

        :param data_loader: data loader
        :param training: if True runs training else runs validation
        """
        if training:
            assert self.optimizer is not None
            eval_fractions = np.linspace(0, 1, self.intra_epoch_eval + 2)[1:-1]
            iters_with_update = len(data_loader) // self.iter_size
            eval_iters = (eval_fractions * iters_with_update).astype(int)
            eval_iters = eval_iters * self.iter_size
            eval_iters = set(eval_iters)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_per_token = AverageMeter(skip_first=False)
        losses_per_sentence = AverageMeter(skip_first=False)

        tot_tok_time = AverageMeter()
        src_tok_time = AverageMeter()
        tgt_tok_time = AverageMeter()

        # XXX
        end = time.time()
        for i, (src, tgt) in enumerate(data_loader, 0):
            # if i >20 and training:
            #     break
            # if not training:
            #     print("shape", src[0].shape)
            # if i > 50 and training:
            #     break
            # print(i)
            self.save_counter += 1
            # measure data loading time
            data_time.update(time.time() - end)

            update = False
            if i % self.iter_size == self.iter_size - 1:
                update = True
            # do a train/evaluate iteration
            num_toks = self.iterate_stage0(src, tgt, update, training=training)
            # loss_per_token, loss_per_sentence, num_toks = stats

            # measure accuracy and record loss
            # losses_per_token.update(loss_per_token, num_toks['tgt'])
            # losses_per_sentence.update(loss_per_sentence, self.batch_size)

            # measure elapsed time
            elapsed = time.time() - end
            batch_time.update(elapsed)
            src_tok_time.update(num_toks['src'] / elapsed)
            tgt_tok_time.update(num_toks['tgt'] / elapsed)
            tot_num_toks = num_toks['tgt'] + num_toks['src']
            tot_tok_time.update(tot_num_toks / elapsed)
            # self.loss = losses_per_token.avg

            if training and i in eval_iters:
                if self.rank < self.stage_num:
                    self.save_for_infer()
                self.barrier()
                self.load_for_infer()
                test_bleu, _ = self.translator.run(calc_bleu=True,
                                                   epoch=self.epoch,
                                                   iteration=i)
                log = []
                log += [f'TRAIN [{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'BLEU: {test_bleu:.2f}']
                log = '\t'.join(log)
                logging.info(log)

                self.barrier()
                self.model.train()
                self.preallocate(True)

            if i % self.print_freq == 0:
                phase = 'TRAIN' if training else 'VALIDATION'
                log = []
                log += [f'{phase} [{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})']
                log += [f'Data {data_time.val:.2e} ({data_time.avg:.2e})']
                log += [f'Tok/s {tot_tok_time.val:.0f} ({tot_tok_time.avg:.0f})']
                # log += [f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})']
                if training:
                    lr = self.optimizer.param_groups[0]['lr']
                    log += [f'LR {lr:.3e}']
                log = '\t'.join(log)
                logging.info(log)

            save_chkpt = (self.save_counter % self.save_freq) == (self.save_freq - 1)
            if training and save_chkpt:
                self.save_counter = 0
                self.save_info['iteration'] = i
                identifier = next(self.checkpoint_counter, -1)
                if identifier != -1:
                    self.save(identifier=identifier)

            end = time.time()
        return tot_tok_time.avg

    def feed_data_rest(self, data_len, training=True):
        if training:
            assert self.optimizer is not None
            eval_fractions = np.linspace(0, 1, self.intra_epoch_eval + 2)[1:-1]
            iters_with_update = data_len // self.iter_size
            eval_iters = (eval_fractions * iters_with_update).astype(int)
            eval_iters = eval_iters * self.iter_size
            eval_iters = set(eval_iters)
        for i in range(0, data_len):
            # if i >20 and training:
            #     break
            # if i > 50 and training:
            #     break
            self.iterate_rest(training=training)
            if training and i in eval_iters:
                test_bleu = torch.cuda.FloatTensor([0])
                break_training = torch.cuda.LongTensor([0])
                if self.rank < self.stage_num:
                    self.save_for_infer()
                self.barrier()
                torch.cuda.empty_cache()
                dist.broadcast(break_training, 0)
                dist.broadcast(test_bleu, 0)
                self.barrier()
                self.model.train()
                self.preallocate_rest(training=True)

    def feed_data_last(self, data_len, training=True):
        if training:
            assert self.optimizer is not None
            eval_fractions = np.linspace(0, 1, self.intra_epoch_eval + 2)[1:-1]
            iters_with_update = data_len // self.iter_size
            eval_iters = (eval_fractions * iters_with_update).astype(int)
            eval_iters = eval_iters * self.iter_size
            eval_iters = set(eval_iters)
        losses_per_token = AverageMeter(skip_first=False)
        losses_per_sentence = AverageMeter(skip_first=False)
        for i in range(0, data_len):
            # if i >20 and training:
            #     break
            # if i > 50 and training:
            #     break
            stats = self.iterate_last(training=training)
            loss_per_token, loss_per_sentence, num_toks = stats
            losses_per_token.update(loss_per_token, num_toks)
            losses_per_sentence.update(loss_per_sentence, self.batch_size)
            if i % self.print_freq == 0:
                phase = 'TRAIN' if training else 'VALIDATION'
                log = []
                log += [f'{phase} [{self.epoch}][{i}/{data_len}]']
                log += [f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})']
                logging.info(log)
            if training and i in eval_iters:
                test_bleu = torch.cuda.FloatTensor([0])
                break_training = torch.cuda.LongTensor([0])
                if self.rank < self.stage_num:
                    self.save_for_infer()
                self.barrier()
                torch.cuda.empty_cache()
                dist.broadcast(break_training, 0)
                dist.broadcast(test_bleu, 0)
                self.barrier()
                self.model.train()
                self.preallocate_last(training=True)

    def save_for_infer(self):
        path = os.getcwd() + '/model_v' + str(self.stage_num) + '/stage' + \
               str(self.group_local_rank) + '.pth'
        self.model.cpu()
        torch.save(self.model.state_dict(), path, _use_new_zipfile_serialization=False)
        self.model.cuda()
        time.sleep(1)

    def load_for_infer(self):
        for i in range(0, self.stage_num):
            path = os.getcwd() + '/model_v' + str(self.stage_num) + '/stage' + str(i) + '.pth'
            func = 'self.translator.model.infer' + str(i)
            eval(func).load_state_dict(torch.load(path, map_location='cpu'))
        self.translator.model.cuda()
        self.translator.model.eval()

    def preallocate(self, training):
        """
        Generates maximum sequence length batch and runs forward and backward
        pass without updating model parameters.

        :param data_loader: data loader
        :param training: if True preallocates memory for backward pass
        """
        batch_size = self.batch_size * self.train_num
        max_len = 50

        src_length = [max_len] * batch_size
        tgt_length = [max_len] * batch_size

        if self.batch_first:
            shape = (batch_size, max_len)
        else:
            shape = (max_len, batch_size)

        src = torch.full(shape, 4, dtype=torch.int64)
        tgt = torch.full(shape, 4, dtype=torch.int64)
        src = src, src_length
        tgt = tgt, tgt_length
        self.iterate_stage0(src, tgt, update=False, training=training)
        self.model.zero_grad()

    def preallocate_rest(self, training):
        self.iterate_rest(update=False, training=training)
        self.model.zero_grad()

    def preallocate_last(self, training):
        self.iterate_last(update=False, training=training)
        self.model.zero_grad()

    def optimize(self, data_loader):
        """
        Sets model in training mode, preallocates memory and runs training on
        data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(True)
        self.model.train()
        torch.cuda.empty_cache()
        if self.group_local_rank == 0:
            self.preallocate(training=True)
            output = self.feed_data(data_loader, training=True)
            self.model.zero_grad()
            torch.cuda.empty_cache()
            return output
        elif self.group_local_rank == self.stage_num - 1:
            self.preallocate_last(training=True)
            self.feed_data_last(data_loader, training=True)
        else:
            self.preallocate_rest(training=True)
            self.feed_data_rest(data_loader, training=True)
        self.model.zero_grad()
        torch.cuda.empty_cache()

    def evaluate(self, data_loader):
        """
        Sets model in eval mode, disables gradients, preallocates memory and
        runs validation on data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        torch.cuda.empty_cache()

        if self.group_local_rank == 0:
            self.preallocate(training=False)
            output = self.feed_data(data_loader, training=False)
            self.model.zero_grad()
            torch.cuda.empty_cache()
            return output
        elif self.group_local_rank == self.stage_num - 1:
            self.preallocate_last(training=False)
            self.feed_data_last(data_loader, training=False)
        else:
            self.preallocate_rest(training=False)
            self.feed_data_rest(data_loader, training=False)
        self.model.zero_grad()
        torch.cuda.empty_cache()

    def load(self, filename):
        """
        Loads checkpoint from filename.

        :param filename: path to the checkpoint file
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location={'cuda:0': 'cpu'})
            if self.distributed:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.fp_optimizer.initialize_model(self.model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            logging.info(f'Loaded checkpoint {filename} (epoch {self.epoch})')
        else:
            logging.error(f'Invalid checkpoint: {filename}')

    def save(self, identifier=None, is_best=False, save_all=False):
        """
        Stores checkpoint to a file.

        :param identifier: identifier for periodic checkpoint
        :param is_best: if True stores checkpoint to 'model_best.pth'
        :param save_all: if True stores checkpoint after completed training
            epoch
        """

        def write_checkpoint(state, filename):
            filename = os.path.join(self.save_path, filename)
            logging.info(f'Saving model to {filename}')
            torch.save(state, filename)

        if self.distributed:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        state = {
            'epoch': self.epoch,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': getattr(self, 'loss', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))

        if identifier is not None:
            filename = self.checkpoint_filename % identifier
            write_checkpoint(state, filename)

        if is_best:
            filename = 'model_best.pth'
            write_checkpoint(state, filename)

        if save_all:
            filename = f'checkpoint_epoch_{self.epoch:03d}.pth'
            write_checkpoint(state, filename)



