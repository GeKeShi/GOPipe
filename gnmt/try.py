import time
import itertools
import torch
import torch.nn as nn
# from seq2seq.models.gnmt import GNMT
from seq2seq.models.sequential_model import embed1, embed2, dropout1, dropout2, dropout3
from seq2seq.models.sequential_model import lstm1, lstm2, lstm3, Attention, Classifier
from seq2seq.models.sequential_model import Conv2d, Pool2d, Linear, Flatten
from seq2seq.train.smoothing import LabelSmoothing
from collections import OrderedDict


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
    if isinstance(tensors, torch.Tensor):
        if tensors.dtype is torch.float32:
            mem += round(tensors.numel() * 4 / (1024 * 1024), 2)
        elif tensors.dtype is torch.int64:
            mem += round(tensors.numel() * 8 / (1024 * 1024), 2)
    else:
        for t in tensors:
            if t.dtype is torch.float32:
                mem += round(t.numel() * 4 / (1024 * 1024), 2)
            elif t.dtype is torch.int64:
                mem += round(t.numel() * 8 / (1024 * 1024), 2)
    return mem


batch_size = 4
num_stage = 2
train_num = 8
# time or memory or efficiency or weight
type = 'efficiency'
# gnmt-8 or gnmt-16 or vgg-16 or amoebanet
model_name = 'vgg-16'

if model_name == 'gnmt-16':
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
if model_name == 'gnmt-8':
    model = nn.Sequential(OrderedDict([
        ('Emb1', embed1()),
        ('E_lstm1', lstm1(bi=True, residual=False)),
        ('Dropout1', dropout1()),
        ('E_lstm2', lstm1(bi=False, residual=False, size=2048)),
        ('Dropout2', dropout1()),
        ('E_lstm3', lstm1()),
        ('Dropout3', dropout1()),
        ('E_lstm4', lstm1()),

        ('Emb2', embed2()),
        ('Dropout4', dropout2()),
        ('D_lstm1', lstm2()),
        ('Attention', Attention()),
        ('Dropout5', dropout3()),
        ('D_lstm2', lstm3(residual=False)),
        ('Dropout6', dropout3()),
        ('D_lstm3', lstm3()),
        ('Dropout7', dropout3()),
        ('D_lstm4', lstm3(last=True)),
        ('Classifier', Classifier(1024, 32320))
    ])).cuda()
if model_name == 'vgg-16':
    model = nn.Sequential(OrderedDict([
        ('Conv1', Conv2d(3, 64)),
        ('Conv2', Conv2d(64, 64)),
        ('Pool1', Pool2d()),

        ('Conv3', Conv2d(64, 128)),
        ('Conv4', Conv2d(128, 128)),
        ('Pool2', Pool2d()),

        ('Conv5', Conv2d(128, 256)),
        ('Conv6', Conv2d(256, 256)),
        ('Conv7', Conv2d(256, 256)),
        ('Pool3', Pool2d()),

        ('Conv8', Conv2d(256, 512)),
        ('Conv9', Conv2d(512, 512)),
        ('Conv10', Conv2d(512, 512)),
        ('Pool4', Pool2d()),

        ('Conv11', Conv2d(512, 512)),
        ('Conv12', Conv2d(512, 512)),
        ('Conv13', Conv2d(512, 512)),
        ('Pool5', nn.AdaptiveAvgPool2d((7, 7))),

        ('Flatten', Flatten()),
        ('Linear1', Linear(25088, 4096)),
        ('Dropout1', nn.Dropout(p=0.5)),
        ('Linear2', Linear(4096, 4096)),
        ('Dropout2', nn.Dropout(p=0.5)),
        ('Linear3', Linear(4096, 1000, False, False)),
    ])).cuda()

timers = SynchronizedWallClockTimer()
layers_time = {}
layers_memory = {}
layers_peak = {}
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
        'out': 0.0,
        'compute': 0.0,
        'peak_compute': 0.0
    }
    layers_time[name] = time_
    layers_memory[name] = memory_
    layers_peak[name] = []
names.reverse()

if model_name == 'gnmt-8' or model_name == 'gnmt-16':
    criterion = LabelSmoothing(0, 0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    max_len = 50
    src_length = [max_len] * batch_size
    src_length = torch.LongTensor(src_length)
    shape = (max_len, batch_size)
    src = torch.full(shape, 4, dtype=torch.int64).cuda()
    tgt = torch.full(shape, 4, dtype=torch.int64).cuda()
    src_length = src_length.cuda()
    tgt_input = tgt[:-1]
    tgt_labels = tgt[1:]
    inputs_origin = (src, src_length, tgt_input)
if model_name == 'vgg-16' or model_name == 'amoebanet-18':
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    inputs_origin = torch.rand(batch_size, 3, 224, 224).cuda()
    tgt_labels = torch.randint(1000, (batch_size,)).cuda()

print("start", torch.cuda.memory_allocated(device=torch.device('cuda:0')) / (1024 * 1024))
compute_label = torch.cuda.memory_allocated(device=torch.device('cuda:0')) / (1024 * 1024)

for i in range(0, 30):
    out_list = []
    inputs_list = []
    inputs = inputs_origin
    if i % 10 == 0:
        print("i", i)
    # print("start", torch.cuda.max_memory_allocated(device=torch.device('cuda:0')) / (1024 * 1024))
    # forward
    for name, layer in model._modules.items():
        # model parameters
        layers_memory[name]['model'] = get_memory(layer.parameters())

        timers(name).start()
        inputs_list.append(inputs)
        out = layer(inputs)
        out_list.append(out)

        # data parameters
        layers_memory[name]['inputs'] = get_memory(inputs)
        layers_memory[name]['out'] = get_memory(out)

        if isinstance(out, tuple):
            inp = []
            for tensor in out:
                out_ = tensor.detach().clone()
                out_.requires_grad = out_.is_floating_point()
                inp.append(out_)
            inputs = tuple(inp)
        if isinstance(out, torch.Tensor):
            inputs = out.detach().clone()
            inputs.requires_grad = inputs.is_floating_point()

        timers(name).stop()
        layers_time[name]['forward'] += timers(name).elapsed()
        timers(name).reset()

        if i == 0:
            com = torch.cuda.memory_allocated(device=torch.device('cuda:0')) / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated(device=torch.device('cuda:0')) / (1024 * 1024)
            if com - compute_label - get_memory(out) > 0:
                layers_memory[name]['compute'] = com - compute_label - get_memory(out)
            compute_label = torch.cuda.memory_allocated(device=torch.device('cuda:0')) / (1024 * 1024)
            # print(name, peak, com, peak - com)
            layers_peak[name].append(peak - com)
            torch.cuda.reset_max_memory_allocated()

    # backward
    for j in range(0, len(names)):
        timers(names[j]).start()
        if j == 0:
            if model_name == 'gnmt-8' or model_name == 'gnmt-16':
                out = out_list.pop()
                T, B = out.size(0), out.size(1)
                loss = criterion(out.view(T * B, -1),
                                 tgt_labels.contiguous().view(-1))
                loss /= (B * 1)
            if model_name == 'vgg-16' or model_name == 'amoebanet-18':
                out = out_list.pop()
                loss = criterion(out, tgt_labels)
            loss.backward()
        else:
            inputs = inputs_list.pop()
            out = out_list.pop()
            if isinstance(inputs, tuple):
                grad_tensors = []
                out_tensors = []
                for k in range(len(inputs)):
                    if inputs[k].is_floating_point() and inputs[k].grad is not None:
                        grad_tensors.append(inputs[k].grad.detach())
                        out_tensors.append(out[k])
                assert len(out_tensors) == len(grad_tensors)
                torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
            if isinstance(inputs, torch.Tensor):
                grad = inputs.grad.detach()
                torch.autograd.backward(tensors=(out, ), grad_tensors=(grad, ))
        if i == 0:
            com = torch.cuda.memory_allocated(device=torch.device('cuda:0')) / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated(device=torch.device('cuda:0')) / (1024 * 1024)
            # print(names[j], peak, com, peak - com)
            layers_peak[names[j]].append(peak - com)
            layers_memory[names[j]]['peak_compute'] = max(layers_peak[names[j]])
            torch.cuda.reset_max_memory_allocated()

        timers(names[j]).stop()
        layers_time[names[j]]['backward'] += timers(names[j]).elapsed()
        timers(names[j]).reset()
    optimizer.step()
    model.zero_grad()

model_memory = 0.0
inputs_memory = 0.0
out_memory = 0.0
compute_memory = 0.0
for_total = 0.0
com_total = 0.0
peak = 0.0
checkpoint = []
names.reverse()
for name in names:
    for_total += layers_time[name]['forward']
    com_total += layers_memory[name]['compute']
    peak = max(layers_memory[name]['peak_compute'], peak)

mem_total = com_total + peak

for name in names:
    model_memory = layers_memory[name]['model']
    inputs_memory = layers_memory[name]['inputs']
    out_memory = layers_memory[name]['out']
    compute_memory = layers_memory[name]['compute']
    forward = layers_time[name]['forward']
    backward = layers_time[name]['backward']
    peak_memory = layers_memory[name]['peak_compute']

    # print(name, ':')
    # print("forward:", forward)
    # print("backward:", backward)
    # print("model:", model_memory)
    # print("compute_memory:", compute_memory)
    # print("peak_memory:", peak_memory)
    # print("inputs:", inputs_memory)
    # print("out:", out_memory)
    # print("checkpoint", forward/for_total, compute_memory/com_total, (compute_memory + peak_memory)/mem_total)
    # print(' ')

    # if compute_memory/com_total > 0.15 and ((compute_memory/com_total)/(forward/for_total) > 2
    #                                         or (compute_memory + peak_memory)/mem_total > 0.4):
    if compute_memory / com_total > 0.15 and (compute_memory / com_total) / (forward / for_total) > 2:
        layers_memory[name]['compute'] = 0
        layers_time[name]['backward'] += layers_time[name]['forward']
        checkpoint.append(name)
names.reverse()

print("checkpoint!!", checkpoint)

stage_time = {}
stage_memory = {}
for i in range(0, num_stage):
    time_ = {
        'forward': 0.0,
        'backward': 0.0
    }
    memory_ = {
        'model': 0.0,
        'data': 0.0,
        'peak': 0.0
    }
    stage_time['s'+str(i)] = time_
    stage_memory['s'+str(i)] = memory_
if num_stage == 4:
    stage = ['s0', 's1', 's2', 's3']
if num_stage == 2:
    stage = ['s0', 's1']

# optimal data
max_memory_index = 100
max_time_index = 100
efficiency = 100000000

num_layers = len(list(model))
if num_stage == 4:
    for split3 in range(1, num_layers - 2):
        for split2 in range(split3 + 1, num_layers - 1):
            for split1 in range(split2 + 1, num_layers):
                each_time = []
                each_memory = []
                each_weight = []
                error_flag = False
                result = 0.0
                total_time = 0.0
                total_memory = 0.0
                max_time_stage = 100
                for s in stage:
                    stage_time[s]['forward'] = 0.0
                    stage_time[s]['backward'] = 0.0
                    stage_memory[s]['model'] = 0.0
                    stage_memory[s]['data'] = 0.0

                peak = []
                for i in range(0, split3):
                    stage_time['s3']['forward'] += layers_time[names[i]]['forward']
                    stage_time['s3']['backward'] += layers_time[names[i]]['backward']
                    stage_memory['s3']['model'] += layers_memory[names[i]]['model']
                    stage_memory['s3']['data'] += layers_memory[names[i]]['compute']
                    peak.append(layers_memory[names[i]]['peak_compute'])
                # weight
                each_weight.append(stage_memory['s3']['model'])
                # stage_memory['s3']['data'] = stage_memory['s3']['data'] * 2
                stage_memory['s3']['data'] = stage_memory['s3']['data']
                # stage_memory['s3']['data'] += layers_memory[names[split3 - 1]]['inputs'] * 2
                stage_memory['s3']['data'] += layers_memory[names[split3 - 1]]['inputs'] * 3
                stage_memory['s3']['peak'] = max(peak)

                peak = []
                for i in range(split3, split2):
                    stage_time['s2']['forward'] += layers_time[names[i]]['forward']
                    stage_time['s2']['backward'] += layers_time[names[i]]['backward']
                    stage_memory['s2']['model'] += layers_memory[names[i]]['model']
                    stage_memory['s2']['data'] += layers_memory[names[i]]['compute']
                    peak.append(layers_memory[names[i]]['peak_compute'])
                # weight
                each_weight.append(stage_memory['s2']['model'])
                # stage_memory['s2']['data'] = stage_memory['s2']['data'] * 3
                stage_memory['s2']['data'] = stage_memory['s2']['data'] * 2
                # stage_memory['s2']['data'] += layers_memory[names[split2 - 1]]['inputs'] * 4
                stage_memory['s2']['data'] += layers_memory[names[split2 - 1]]['inputs'] * 5
                # stage_memory['s2']['data'] += layers_memory[names[split3]]['out'] * 2
                stage_memory['s2']['data'] += layers_memory[names[split3]]['out'] * 2
                stage_memory['s2']['peak'] = max(peak)

                peak = []
                for i in range(split2, split1):
                    stage_time['s1']['forward'] += layers_time[names[i]]['forward']
                    stage_time['s1']['backward'] += layers_time[names[i]]['backward']
                    stage_memory['s1']['model'] += layers_memory[names[i]]['model']
                    stage_memory['s1']['data'] += layers_memory[names[i]]['compute']
                    peak.append(layers_memory[names[i]]['peak_compute'])
                # weight
                each_weight.append(stage_memory['s1']['model'])
                # stage_memory['s1']['data'] = stage_memory['s1']['data'] * 4
                stage_memory['s1']['data'] = stage_memory['s1']['data'] * 3
                # stage_memory['s1']['data'] += layers_memory[names[split1 - 1]]['inputs'] * 6
                stage_memory['s1']['data'] += layers_memory[names[split1 - 1]]['inputs'] * 7
                # stage_memory['s1']['data'] += layers_memory[names[split2]]['out'] * 3
                stage_memory['s1']['data'] += layers_memory[names[split2]]['out'] * 2
                stage_memory['s1']['peak'] = max(peak)

                peak = []
                for i in range(split1, num_layers):
                    stage_time['s0']['forward'] += layers_time[names[i]]['forward']
                    stage_time['s0']['backward'] += layers_time[names[i]]['backward']
                    stage_memory['s0']['model'] += layers_memory[names[i]]['model']
                    stage_memory['s0']['data'] += layers_memory[names[i]]['compute']
                    peak.append(layers_memory[names[i]]['peak_compute'])
                # weight
                each_weight.append(stage_memory['s0']['model'])
                # stage_memory['s0']['data'] = stage_memory['s0']['data'] * 5
                stage_memory['s0']['data'] = stage_memory['s0']['data'] * 4
                stage_memory['s0']['data'] += layers_memory[names[num_layers - 1]]['inputs'] * 8
                # stage_memory['s0']['data'] += layers_memory[names[split1]]['out'] * 4
                stage_memory['s0']['data'] += layers_memory[names[split1]]['out'] * 2
                stage_memory['s0']['peak'] = max(peak)

                if model_name == 'gnmt-8' or model_name == 'gnmt-16':
                    stage_memory['s0']['model'] = stage_memory['s0']['model'] * 5  # Adam + grad
                    stage_memory['s1']['model'] = stage_memory['s1']['model'] * 5  # Adam + grad
                    stage_memory['s2']['model'] = stage_memory['s2']['model'] * 4  # Adam
                    stage_memory['s3']['model'] = stage_memory['s3']['model'] * 4  # Adam
                if model_name == 'vgg-16' or model_name == 'amoebanet-18':
                    stage_memory['s0']['model'] = stage_memory['s0']['model'] * 4  # sgd + grad
                    stage_memory['s1']['model'] = stage_memory['s1']['model'] * 4  # sgd + grad
                    stage_memory['s2']['model'] = stage_memory['s2']['model'] * 3  # sgd
                    stage_memory['s3']['model'] = stage_memory['s3']['model'] * 3  # sgd

                # for comb in itertools.permutations(stage, 2):
                #     for1 = stage_time[comb[0]]['forward']
                #     for2 = stage_time[comb[1]]['forward']
                #     back1 = stage_time[comb[0]]['backward']
                #     back2 = stage_time[comb[1]]['backward']
                #     if back1 > (for2 + back2) or for1 > (for2 + back2):
                #         error_flag = True
                #         break
                # if error_flag:
                #     continue

                # calculate time and memory
                for s in stage:
                    # time 4*(for and back)
                    each_time.append(stage_time[s]['forward'] + stage_time[s]['backward'])
                    total_time += (stage_time[s]['forward'] + stage_time[s]['backward'])
                    # memory
                    each_memory.append(stage_memory[s]['model'] + stage_memory[s]['data'] + stage_memory[s]['peak'])
                max_time_stage = each_time.index(max(each_time))
                # time rest of (for and back)
                for s in range(0, num_stage - max_time_stage):
                    total_time += each_time[num_stage - 1 - s]
                # time of longest
                total_time += (train_num - 5 + max_time_stage) * each_time[max_time_stage]
                total_memory = max(each_memory)
                if type == 'time':
                    result = total_time
                elif type == 'memory':
                    result = total_memory
                elif type == 'efficiency':
                    result = total_time * total_memory
                elif type == 'weight':
                    result = max(each_weight)
                else:
                    raise ValueError('not support type')

                if result < efficiency:
                    max_memory_index = each_memory.index(total_memory)
                    max_time_index = max_time_stage
                    optimal_each_time = each_time
                    optimal_each_memory = each_memory
                    optimal_each_weight = each_weight
                    efficiency = result
                    optimal_split = (split3, split2, split1)
    print("result", optimal_split)
    print("stage3:")
    for i in range(0, optimal_split[0]):
        print(names[i])
    print(' ')
    print("stage2:")
    for i in range(optimal_split[0], optimal_split[1]):
        print(names[i])
    print(' ')
    print("stage1:")
    for i in range(optimal_split[1], optimal_split[2]):
        print(names[i])
    print(' ')
    print("stage0:")
    for i in range(optimal_split[2], num_layers):
        print(names[i])

    print("optimal_data:", max_memory_index, max_time_index)
    print("optimal_memory:", optimal_each_memory)
    print("optimal_time:", optimal_each_time)
    print("optimal_weight:", optimal_each_weight)
    print("efficiency:", efficiency)
    # print("Model", model)
    # model3 = model[num_layers - optimal_split[0]:num_layers]
    # model2 = model[num_layers - optimal_split[1]:num_layers - optimal_split[0]]
    # model1 = model[num_layers - optimal_split[2]:num_layers - optimal_split[1]]
    # model0 = model[0:num_layers - optimal_split[2]]
    print(num_layers - optimal_split[2],
          optimal_split[2] - optimal_split[1], optimal_split[1] - optimal_split[0], optimal_split[0])

if num_stage == 2:
    for split in range(1, num_layers):
        each_time = []
        each_memory = []
        each_weight = []
        error_flag = False
        result = 0.0
        total_time = 0.0
        total_memory = 0.0
        max_time_stage = 100
        for s in stage:
            stage_time[s]['forward'] = 0.0
            stage_time[s]['backward'] = 0.0
            stage_memory[s]['model'] = 0.0
            stage_memory[s]['data'] = 0.0

        peak = []
        for i in range(0, split):
            stage_time['s1']['forward'] += layers_time[names[i]]['forward']
            stage_time['s1']['backward'] += layers_time[names[i]]['backward']
            stage_memory['s1']['model'] += layers_memory[names[i]]['model']
            stage_memory['s1']['data'] += layers_memory[names[i]]['compute']
            peak.append(layers_memory[names[i]]['peak_compute'])
        # weight
        each_weight.append(stage_memory['s1']['model'])
        stage_memory['s1']['data'] = stage_memory['s1']['data']
        # stage_memory['s1']['data'] += layers_memory[names[split - 1]]['inputs'] * 2
        stage_memory['s1']['data'] += layers_memory[names[split - 1]]['inputs'] * 3
        stage_memory['s1']['peak'] = max(peak)

        peak = []
        for i in range(split, num_layers):
            stage_time['s0']['forward'] += layers_time[names[i]]['forward']
            stage_time['s0']['backward'] += layers_time[names[i]]['backward']
            stage_memory['s0']['model'] += layers_memory[names[i]]['model']
            stage_memory['s0']['data'] += layers_memory[names[i]]['compute']
            peak.append(layers_memory[names[i]]['peak_compute'])
        # weight
        each_weight.append(stage_memory['s0']['model'])
        stage_memory['s0']['data'] = stage_memory['s0']['data'] * 2
        stage_memory['s0']['data'] += layers_memory[names[num_layers - 1]]['inputs'] * 4
        stage_memory['s0']['data'] += layers_memory[names[split]]['out'] * 2  # clone
        stage_memory['s0']['peak'] = max(peak)

        if model_name == 'gnmt-8' or model_name == 'gnmt-16':
            stage_memory['s0']['model'] = stage_memory['s0']['model'] * 5  # Adam
            stage_memory['s1']['model'] = stage_memory['s1']['model'] * 4  # Adam
        if model_name == 'vgg-16' or model_name == 'amoebanet-18':
            stage_memory['s0']['model'] = stage_memory['s0']['model'] * 4  # sgd
            stage_memory['s1']['model'] = stage_memory['s1']['model'] * 3  # sgd
        each_weight.reverse()

        # for comb in itertools.permutations(stage, 2):
        #     for1 = stage_time[comb[0]]['forward']
        #     for2 = stage_time[comb[1]]['forward']
        #     back1 = stage_time[comb[0]]['backward']
        #     back2 = stage_time[comb[1]]['backward']
        #     if back1 > (for2 + back2) or for1 > (for2 + back2):
        #         error_flag = True
        #         break
        # if error_flag:
        #     continue

        # calculate time and memory
        for s in stage:
            # time 4*(for and back)
            each_time.append(stage_time[s]['forward'] + stage_time[s]['backward'])
            total_time += (stage_time[s]['forward'] + stage_time[s]['backward'])
            # memory
            each_memory.append(stage_memory[s]['model'] + stage_memory[s]['data'] + stage_memory[s]['peak'])
        max_time_stage = each_time.index(max(each_time))
        # time rest of (for and back)
        for s in range(0, num_stage - max_time_stage):
            total_time += each_time[num_stage - 1 - s]
        # time of longest
        total_time += (train_num - 3 + max_time_stage) * each_time[max_time_stage]
        total_memory = max(each_memory)

        if type == 'time':
            result = total_time
        elif type == 'memory':
            result = total_memory
        elif type == 'efficiency':
            result = total_time * total_memory
        elif type == 'weight':
            result = max(each_weight)
        else:
            raise ValueError('not support type')

        if result < efficiency:
            max_memory_index = each_memory.index(total_memory)
            max_time_index = max_time_stage
            optimal_each_time = each_time
            optimal_each_memory = each_memory
            optimal_each_weight = each_weight
            efficiency = result
            optimal_split = split
    print("result", optimal_split)
    print("stage1:")
    for i in range(0, optimal_split):
        print(names[i])
    print(' ')
    print("stage0:")
    for i in range(optimal_split, num_layers):
        print(names[i])

    print("optimal_data:", max_memory_index, max_time_index)
    print("optimal_memory:", optimal_each_memory)
    print("optimal_time:", optimal_each_time)
    print("optimal_weight:", optimal_each_weight)

    # model1 = model[num_layers - optimal_split:num_layers]
    # model0 = model[0:num_layers - optimal_split]



