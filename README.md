# SISCI NCCL plugin
[NCCL](https://github.com/NVIDIA/nccl) plugin for [Dolphin Interconnect](https://www.dolphinics.com/) PCI-e adapters, providing
low-latency high-bandwith inter-node GPU communication that can be used for distributed training of artificial neural networks
using frameworks such as [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/).

Requirements
------------
* Linux (tested on Ubuntu 18.04)
* NVIDIA GPU (Quadro or Tesla GPU for GPUDirect RDMA support)
* [Dolphin Interconnect Solutions](http://dolphinics.com) software stack and
  supported hardware.
* CUDA (tested with 10.0)
* NCCL 2.4.*

Build
-------------------

### Tools
* GCC (tested with 7.3.0)
* Autotools (autoconf, automake, libtool)

### Instructions
```
./autogen.sh
./configure
make
sudo make install
```

Usage
------

NCCL will automatically detect and load the plugin. Enable debug output

```
export NCCL_DEBUG=INFO
```

and you should see something like this in the terminal when you run your application:

```
NCCL INFO Trying to load SISCI
NCCL INFO NET/SISCI : adapter 0, node id 4
```

### GPUDirect RDMA (GDR)

Use the [`NCCL_NET_GDR_LEVEL` enviromental variable](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-net-gdr-level-formerly-nccl-ib-gdr-level) to control the use of GDR. Run the [NCCL tests](https://github.com/nvidia/nccl-tests) to evaluate performance.


Troubleshooting
--------------

##### `NCCL INFO NET/Plugin : No plugin found (libnccl-net.so)`

Make sure that the plugin is found either by adding the library install path (defaults to `/usr/local/lib`) to the `LD_LIBRARY_PATH`
environment variable or to `/etc/ld.so.conf`.

##### `NCCL WARN SCIAttachPhysicalMemory: Out of hardware resources`

The GPU doesn't support GPUDirect RDMA. To disable GDR, run:
```
export NCCL_NET_GDR_LEVEL=0
```
