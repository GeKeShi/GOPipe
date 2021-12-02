User Guide
==========

Installation
~~~~~~~~~~~~

:mod:`torchgpipe` is available on PyPI_. Install by ``pip``:

.. sourcecode:: console

   $ pip install torchgpipe

.. _PyPI: https://pypi.org/project/torchgpipe

Python 3.6+ (CPython) is required.

PyTorch 1.1+ will be installed automatically if you don't have a satisfied one.
However, we highly recommend you to use the latest version of PyTorch.

Applying GPipe
~~~~~~~~~~~~~~

To train a module with GPipe, simply wrap it with :class:`torchgpipe.GPipe`.
Your module must be a :class:`nn.Sequential <torch.nn.Sequential>` as
:class:`~torchgpipe.GPipe` will automatically split the module into partitions.
A partition is a group of consecutive layers that run on a single device
together. `balance` argument determines the number of layers in each partition.
`chunks` argument specifies the number of micro-batches. Input, output, and
intermediate tensors must be :class:`~torch.Tensor` or ``Tuple[Tensor, ...]``.
See also `Restrictions`_ for more details.

The below example code shows how to split a module with four layers into two
partitions each having two layers. This code also splits a mini-batch into 8
micro-batches::

   from torchgpipe import GPipe

   model = nn.Sequential(a, b, c, d)
   model = GPipe(model, balance=[2, 2], chunks=8)

   # 1st partition: nn.Sequential(a, b) on cuda:0
   # 2nd partition: nn.Sequential(c, d) on cuda:1

   for input in data_loader:
       output = model(input)

:class:`~torchgpipe.GPipe` optimizes training using CUDA. You should not move
the module to a GPU yourself, because :class:`~torchgpipe.GPipe` automatically
moves each partition over different devices. By default, available GPUs
starting from ``cuda:0`` are selected in order for each partition. You can also
specify GPUs to use with `devices` parameter::

   model = GPipe(model,
                 balance=[2, 2],
                 devices=[4, 2],  # Specify GPUs.
                 chunks=8)

Input and Output Device
-----------------------

Unlike a typical module, with :class:`~torchgpipe.GPipe`, the input device is
different from the output device except for when there is only one partition.
This is because the first partition and last partition are placed in different
devices.

Therefore, you have to move the input and target to the corresponding devices.
It can be done with :attr:`GPipe.devices <torchgpipe.GPipe.devices>`, which
holds the list of devices for each partition::

   in_device = model.devices[0]
   out_device = model.devices[-1]

   for input, target in data_loader:
       # input on in_device
       input = input.to(in_device, non_blocking=True)

       # target on out_device
       target = target.to(out_device, non_blocking=True)

       # output on out_device
       output = model(input)
       loss = F.cross_entropy(output, target)
       loss.backward()
       ...

Nested Sequentials
------------------

When :class:`~torchgpipe.GPipe` splits a :class:`nn.Sequential
<torch.nn.Sequential>` module, it regards every child of the module as a
single, non-divisible layer. However, it may be the case that some child is
another sequential module and one may want to split them further.

This kind of recursive split of a nested sequential module is not intended nor
supported by :class:`~torchgpipe.GPipe`. It's your responsibility to flatten
the module. Fortunately, this is not hard in PyTorch. Follow this code snippet
which shows how a nested sequential module can be flattened::

   _3_layers = nn.Sequential(...)  # len(_3_layers) == 3
   _4_layers = nn.Sequential(...)  # len(_4_layers) == 4
   model = nn.Sequential(_3_layers, _4_layers)  # len(model) == 2

   def flatten_sequential(module):
       def _flatten(module):
           for name, child in module.named_children():
               if isinstance(child, nn.Sequential):
                   for sub_name, sub_child in _flatten(child):
                       yield (f'{name}_{sub_name}', sub_child)
               else:
                   yield (name, child)
       return nn.Sequential(OrderedDict(_flatten(module)))

   model = flatten_sequential(model)  # len(model) == 7
   model = GPipe(model, balance=[2, 3, 2], chunks=4)

Typical Model Parallelism
-------------------------

The typical model parallelism is a special case of GPipe. Model parallelism is
equivalent to GPipe if micro-batching and checkpointing are disabled. Set
``chunks=1`` and ``checkpoint='never'`` for this::

   model = GPipe(model, balance=[2, 2], chunks=1, checkpoint='never')

Automatic Balancing
~~~~~~~~~~~~~~~~~~~

It could be hard to determine the optimal balance of a model. In particular, if
you are still designing a model, the model architecture may change over time.
In this case, we highly recommend :mod:`torchgpipe.balance` for automatic
balancing. This won't give you the optimal balance, but a good-enough balance.
Note that this is provided by :mod:`torchgpipe`, and is not from the GPipe
paper by Huang et al.

There are two balancing tools, :func:`~torchgpipe.balance.balance_by_time` and
:func:`~torchgpipe.balance.balance_by_size`. Both are based on per-layer
profiling. Just like `PyTorch JIT`_, you need to feed a sample input into the
model. :func:`~torchgpipe.balance.balance_by_time` traces elapsed time of each
layer, while :func:`~torchgpipe.balance.balance_by_size` detects the CUDA
memory usage of each layer. Choose the balancing tool for your needs::

   from torchgpipe import GPipe
   from torchgpipe.balance import balance_by_time

   partitions = torch.cuda.device_count()
   sample = torch.rand(128, 3, 224, 224)
   balance = balance_by_time(partitions, model, sample)

   model = GPipe(model, balance, chunks=8)

.. _PyTorch JIT: https://pytorch.org/docs/stable/jit.html

Trade-offs
~~~~~~~~~~

Number of Micro-batches
-----------------------

Number of micro-batches has a trade-off between GPU utilization per micro-batch
and total area of bubble. You need to find the best number of micro-batches for
your model.

GPU may slow down when processing many small micro-batches compared to larger
micro-batches. GPU will not be fully utilized if each CUDA kernel is too cheap
to compute, hence too small micro-batches cause underutilization. On the other
hand, the area of bubble is minimized when the size of each micro-batch is
minimal. Ideally, you should choose the largest number of micro-batches that
doesn't underutilize GPUs.

As a side note, BatchNorm tends to perform worse with smaller batch size. Large
number of micro-batches may affect the final performance of model using
BatchNorm negatively just like in :class:`nn.DataParallel
<torch.nn.DataParallel>`.

Checkpointing
-------------

Checkpointing drastically helps to reduce memory usage, but the overall
training would slow down by about 25%. You can handle how to apply
checkpointing on your model. There are three options:

- ``'always'`` -- Apply checkpointing over all micro-batches.
- ``'except_last'`` (default) -- Apply checkpointing except the last
  micro-batch.
- ``'never'`` -- Checkpointing is never applied.

Usually, checkpointing at the last micro-batch may not be useful because the
saved memory will be reconstructed immediately. That's why we choose
``'except_last'`` as the default option.

If you decide not to use checkpointing at all, :class:`nn.DataParallel
<torch.nn.DataParallel>` might be more efficient than GPipe.

Referential Transparency
~~~~~~~~~~~~~~~~~~~~~~~~

Checkpointing executes forward propagation again at backpropagation, which is
called `recomputation`. We assume that both the executions are identical.
Hence, all layers should be `referentially transparent
<https://en.wikipedia.org/wiki/Referential_transparency>`_ in forward
propagation. Here are the typical cases that break referential transparency:

In-place Operations:
   We do not recommend using in-place operations with checkpointing.
   Especially, if an in-place operation such as ``add_(1)`` is applied to the
   input of a checkpointed partition, then the recomputation can't recover the
   original input.

Randomness not managed by PyTorch:
   The randomness managed by PyTorch, including :func:`torch.manual_seed`,
   :func:`torch.rand`, or :class:`nn.Dropout <torch.nn.Dropout>`, is
   deterministically reproduced in recomputation. But other randomnesses, such
   as Python standard :mod:`random` or :mod:`numpy.random`, are not. We highly
   recommend to use PyTorch randomness for referential transparency.

Side Effects:
   Some modules such as BatchNorm update their state in forward propagation.
   Hence, updated state in recomputation might not be identical to the original
   state.

Restrictions
~~~~~~~~~~~~

If you get any errors, check the following restrictions first.

Sequential:
   Your module must be :class:`nn.Sequential <torch.nn.Sequential>`. For
   example, the models in :mod:`torchvision` are not sequential. They can't be
   wrapped by :class:`~torchgpipe.GPipe` directly::

      >>> from torchvision.models.resnet import resnet101
      >>> model = resnet101()
      >>> type(model)
      torchvision.models.resnet.ResNet
      >>> GPipe(model, balance=..., chunks=...)
      Traceback (most recent call last)
        ...
      TypeError: module must be nn.Sequential to be partitioned

   See `the sequential ResNet example`_ to figure out how to make a  model into
   a :class:`nn.Sequential <torch.nn.Sequential>` model.

   .. _the sequential ResNet example:
      https://github.com/kakaobrain/torchgpipe/tree/master/benchmarks/models/resnet

   :class:`nn.Sequential <torch.nn.Sequential>` assumes that every underlying
   layer takes only one argument. Calling ``forward(x)`` on
   ``nn.Sequential(A(), B(), C())`` is essentially the same as calling
   ``C(B(A(x)))``. Hence, you can't design an underlying layer with multiple
   arguments::

      class MyModule(nn.Module):
          def forward(self, a, b, c):
              return a + b - c

      model = nn.Sequential(..., MyModule(), ...)
      model(input)  # FAILS!

Tensor or Tensors:
   As we discussed above, each layer must take only one argument due to
   :class:`nn.Sequential <torch.nn.Sequential>`. There is one more restriction.
   Every underlying layers' input and output must be :class:`~torch.Tensor` or
   ``Tuple[Tensor, ...]``::

      # OK
      def forward(input: Tensor) -> Tensor: ...
      def forward(input: Tensor) -> Tuple[Tensor, Tensor]: ...
      def forward(input: Tuple[Tensor, Tensor]) -> Tensor: ...

      # Error
      def forward(input1: Tensor, input2: Tensor) -> Tensor: ...
      def forward(input: Tensor, label: str) -> Tensor: ...
      def forward(input: Tensor) -> Dict[str, Tensor]: ...
      def forward(input: Tensor) -> Tuple[Tensor, str]: ...

   The reason is that :class:`~torchgpipe.GPipe` can't assume how the
   non-tensor inputs for a mini-batch can be split for micro-batches.

Unique Parameters:
   :class:`~torchgpipe.GPipe` places each partition on the corresponding
   device. When placing a partition, the parameters of the partition are also
   moved to the destination. :class:`~torchgpipe.GPipe` cannot support a module
   with a parameter on two or more devices::

      >>> conv1 = nn.Conv2d(3, 3, 1)
      >>> conv2 = nn.Conv2d(3, 3, 1)
      >>> conv1.weight = conv2.weight
      >>> model = nn.Sequential(conv1, conv2)
      >>> model = GPipe(model, balance=[1, 1], ...)
      Traceback (most recent call last)
        ...
      ValueError: module with duplicate parameters in distinct children is not supported

Complex Modules
~~~~~~~~~~~~~~~

This part of the documentation discusses how to implement a complex module
compatible with :class:`~torchgpipe.GPipe`. First, you should understand how
GPipe works. See :ref:`Understanding GPipe`.

Skip Connections
----------------

Many deep learning models, such as ResNet, AmoebaNet, or U-Net, contain skip
connections. There are two ways to implement skip connections. Let's assume we
have to implement a skip connection like this::

   latent = layer1(input)
   latent = layer2(latent)
   output = layer3(latent) + input  # skip connection

To make this module sequential, we define modules for each layer. Simply,
a skip connection can be implemented by making underlying layers with
``Tuple[Tensor, Tensor]`` parameter and return type::

   class Layer1(nn.Module):
       #         ┌────────────────┐
       # input --│-+-> layer1 ----│--> output
       #         │ '--------------│--> skip
       #         └────────────────┘
       def forward(self, input):
           skip = input
           return layer1(input), skip

   class Layer2(nn.Module):
       #         ┌────────────────┐
       # input --│---> layer2 ----│--> output
       #  skip --│----------------│--> skip
       #         └────────────────┘
       def forward(self, input_and_skip):
           input, skip = input_and_skip
           return layer2(input), skip

   class Layer3(nn.Module):
       #         ┌────────────────┐
       # input --│---> layer3 --+-│--> output
       #  skip --│--------------' │
       #         └────────────────┘
       def forward(self, input_and_skip):
           input, skip = input_and_skip
           return layer3(input) + skip

   model = nn.Sequential(Layer1(), Layer2(), Layer3())

Because of the skip connection being represented as a normal parameter,
:class:`~torchgpipe.GPipe` can move the tensors from partition to partition::

   model = GPipe(model, balance=[1, 1, 1], chunks=8)

This seems a fairly straightforward way to implement skip connections. But
there is a disadvantage. In the above example, the skip tensor is copied to the
second device, but it is never used at the device. Unnecessary copies of skip
tensors may waste time and memory. The following section introduces an
alternative approach for skip connection.

Long Skip Connections
---------------------

The disadvantage mentioned above might be catastrophic if it involves
unnecessary copies of a large tensor, and/or over many devices. The second case
often occurs when implementing long skip connections.

Let's assume now we have 8 layers between input and output::

   latent = layer1(input)
   latent = layer2(latent)
   latent = layer3(latent)
   latent = layer4(latent)
   latent = layer5(latent)
   latent = layer6(latent)
   latent = layer7(latent)
   output = layer8(latent) + input  # skip connection

With the prior approach, the skip tensor will be copied to every device, but
six devices do not need it. The alternative approach is to expose in which
layer the skip tensor is produced and consumed. We introduce the
:func:`@skippable <torchgpipe.skip.skippable>` class decorator to toss the
tensor directly, without needing to pass it to irrelevant layers in between. A
module can stash a tensor into the storage or pop. This functionality works
perfectly fine even when the module is not wrapped by
:class:`~torchgpipe.GPipe`.

The decorator declares which skip tensors would be stashed or popped in the
decorated module. Let us explain how to implement the 8-layer example above
using :mod:`torchgpipe.skip`. Here we use the name "skip" for the skip
connection between ``Layer1`` and ``Layer8``::

   # Layer1 stashes 'skip'.
   @skippable(stash=['skip'])
   class Layer1(nn.Module):
       ...

   # Layer8 pops 'skip'.
   @skippable(pop=['skip'])
   class Layer8(nn.Module):
       ...

When ``Layer1`` prepares a skip tensor, it can stash the tensor into the hidden
storage by :func:`yield stash() <torchgpipe.skip.stash>`. As you may have
noticed, we define ``forward()`` as a generator_ instead of a normal function::

   @skippable(stash=['skip'])
   class Layer1(nn.Module):
       def forward(self, input):
           skip = input
           yield stash('skip', skip)
           return layer1(input)

.. _generator: https://docs.python.org/3/howto/functional.html#generators

Similarly, ``Layer8`` also can pop the stashed skip tensor by :func:`yield
pop() <torchgpipe.skip.pop>`::

   @skippable(pop=['skip'])
   class Layer8(nn.Module):
       def forward(self, input):
           skip = yield pop('skip')
           return layer8(input) + skip

Now the intermediate layers do not interact with the skip tensor at all::

   class Layer2(nn.Module):
       def forward(self, input):
           return layer2(input)
   ...
   class Layer7(nn.Module):
       def forward(self, input):
           return layer7(input)

You can design any complex skip connections with :func:`@skippable
<torchgpipe.skip.skippable>` since a skippable module could stash and/or pop
multiple skip tensors. However, there are restrictions:

- Every skip name must be unique within a sequential module.
- Every skip tensor must be stashed and popped exactly once.

Then, how can we instantiate multiple skippable modules from the same class in
a sequential module? You can isolate some skip names into a
:class:`~torchgpipe.skip.Namespace`. For example, a conceptual U-Net can be
designed like this. There are 3 pairs of ``Encoder`` and ``Decoder``::

   # 1F. Encoder -------- Decoder -- Segment
   #        \                /
   # 2F.  Encoder ------ Decoder
   #          \            /
   # 3F.   Encoder ---- Decoder
   #            \        /
   # 4F.        Bottleneck

   @skippable(stash=['skip'])
   class Encoder(nn.Module):
       ...

   @skippable(pop=['skip'])
   class Decoder(nn.Module):
       ...

   ns_1f = Namespace()
   ns_2f = Namespace()
   ns_3f = Namespace()

   model = nn.Sequential(
       Encoder().isolate(ns_1f),
       Encoder().isolate(ns_2f),
       Encoder().isolate(ns_3f),
       Bottleneck(),
       Decoder().isolate(ns_3f),
       Decoder().isolate(ns_2f),
       Decoder().isolate(ns_1f),
       Segment(),
   )

Some skip connection may be conditional on input. However, :func:`@skippable
<torchgpipe.skip.skippable>` doesn't allow :func:`~torchgpipe.skip.stash` or
:func:`~torchgpipe.skip.pop` missing. Instead, it allows :data:`None` in place
of skip tensor::

   @skippable(stash=['skip'])
   class MaybeStash(nn.Module):
       def forward(self, input):
           skip = input if test(input) else None
           yield stash('skip', skip)
           return f(input)

   @skippable(pop=['skip'])
   class MaybePop(nn.Module):
       def forward(self, input):
           output = f(input)
           skip = yield pop('skip')
           if skip is not None:
               output += skip
           return output

Detecting Recomputation
-----------------------

Checkpointing in GPipe performs forward propagations twice. The second forward
propagation is called `recomputation`. This may cause a problem when a module
such as :class:`nn.BatchNorm2d <torch.nn.BatchNorm2d>` updates its running
estimates of batch statistics on each forward propagation. It should not update
the running estimates again during the recomputation. To avoid updating the
running estimates twice, modules' ``forward`` method needs be able to detect
that this is the recomputation.

It can be done by :func:`~torchgpipe.is_recomputing`. This function returns
:data:`True` if called during the recomputation::

   class Counter(nn.Module):
       def __init__(self):
           super().__init__()
           self.counter = 0

       def forward(self, input):
           if not is_recomputing():
               self.counter += 1
           return input

.. note::

   ``deferred_batch_norm=True`` on :class:`~torchgpipe.GPipe` will prevent
   updating the running statistics twice.
