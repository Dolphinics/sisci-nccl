/* Copyright 2019 Dolphin Interconnect Solutions AS

   Permission to use, copy, modify, and/or distribute this software
   for any purpose with or without fee is hereby granted, provided
   that the above copyright notice and this permission notice appear
   in all copies.

   THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
   WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE
   AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR
   CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
   OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
   NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
   CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

#include "sisciwrap.h"
#include "sisci_nccl.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <arpa/inet.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define UNUSED(x) (void)(x)

#define NO_FLAGS              0
#define NO_OFFSET             0
#define NO_CALLBACK           NULL
#define NO_ARG                NULL
#define MAX_SCI_DEVS          4
#define MAILBOX_SEGMENT_SIZE  2
#define MAX_NODES             60
#define SEGMENT_PREFIX        0xba000000
#define NOT_MAILBOX           0
#define IS_MAILBOX            1
#define CHANNELS              4
#define IR_DATA_SIZE          2

#define COMM_FLAG_EMPTY   0
#define COMM_FLAG_REQUEST 1
#define COMM_FLAG_NOTIFY  2

struct ncclSisciDev {
    unsigned int adapter_no;
    unsigned int node_id;
    unsigned int node_offset;
};

struct ncclSisciDev ncclSisciDevs[MAX_SCI_DEVS];

int cuda_device;
int ncclSisciNDevs = -1;

pthread_mutex_t ncclSisciLock = PTHREAD_MUTEX_INITIALIZER;

// Initialize the network.
ncclResult_t ncclSisciInit(ncclDebugLogger_t logFunction) {
    ncclDebugLog = logFunction;

    pthread_mutex_lock(&ncclSisciLock);
    if (ncclSisciNDevs == -1) {
        INFO(NCCL_NET|NCCL_INIT, "Trying to load SISCI");
        NCCLCHECK(ncclSCIInitialize(NO_FLAGS));

        cudaError_t err = cudaGetDevice(&cuda_device);
        if (err != cudaSuccess) {
            WARN("Failed to get current GPU: %s", cudaGetErrorString(err));
            return ncclInternalError;
        }

        ncclSisciNDevs = 0;
        for (int i = 0; i < MAX_SCI_DEVS; i++) {
            struct ncclSisciDev *dev = &ncclSisciDevs[i];

            dev->adapter_no = i;

            if (ncclSCIGetLocalNodeId(dev->adapter_no, &dev->node_id, NO_FLAGS) ==
                ncclSuccess) {
                INFO(NCCL_INIT|NCCL_NET, "NET/SISCI : adapter %u, node id %u",
                     dev->adapter_no, dev->node_id);

                dev->node_offset = (dev->node_id >> 2) - 1;

                ncclSisciNDevs++;
            }
            else {
                break;
            }
        }
    }
    pthread_mutex_unlock(&ncclSisciLock);

    if (ncclSisciNDevs == 0) {
        INFO(NCCL_INIT|NCCL_NET, "NET/SISCI : No devices found.");
    }

    return ncclSuccess;
}

// Return the number of adapters.
ncclResult_t ncclSisciDevices(int* ndev) {
    *ndev = ncclSisciNDevs;

    return ncclSuccess;
}

// Return the device path in /sys. NCCL will call free on this path.
ncclResult_t ncclSisciPciPath(int dev, char** path) {
    struct ncclSisciDev *devp = &ncclSisciDevs[dev];
    char devicepath[PATH_MAX];
    sci_query_adapter_t query;
    uint16_t bdf;
    uint8_t bus;
    uint8_t device;

    query.subcommand = SCI_Q_ADAPTER_BDF;
    query.localAdapterNo = devp->adapter_no;
    query.data = &bdf;
    NCCLCHECK(ncclSCIQuery(SCI_Q_ADAPTER, &query,
                           NO_FLAGS));
    bus = bdf >> 8;
    device = bdf & 0x00ff;

    snprintf(devicepath, PATH_MAX,
             "/sys/class/pci_bus/0000:%02x/device/0000:%02x:%02x.0/",
             bus, bus, device);
    *path = realpath(devicepath, NULL);

    INFO(NCCL_NET, "Adapter %d path: %s", devp->adapter_no,
         devicepath);

    return ncclSuccess;
}

// Return whether this device supports host pointers and/or CUDA pointers
// as data from the current GPU. Supported types should be composed with
// NCCL_PTR_HOST and NCCL_PTR_CUDA.
ncclResult_t ncclSisciPtrSupport(int dev, int* supportedTypes) {
    UNUSED(dev);
    *supportedTypes = NCCL_PTR_HOST | NCCL_PTR_CUDA;

    return ncclSuccess;
}

struct ncclSisciHandle {
    unsigned int node_id;
    unsigned int irno;
    int cuda_device;
};

struct ncclSisciListenComm {
    struct ncclSisciDev *dev;
    sci_desc_t sd;
    sci_local_data_interrupt_t ir;
};

struct ncclSisciMailbox {
    struct ncclSisciDev *dev;
    unsigned int remote_node_id;
    sci_desc_t sd;
    uint32_t local_id;
    uint32_t remote_id;
    sci_local_segment_t local_segment;
    sci_remote_segment_t remote_segment;
    sci_map_t local_map;
    sci_map_t remote_map;
    volatile void *local_addr;
    volatile void *remote_addr;
};

enum ncclSisciCommType { SISCI_RECV,
                         SISCI_SEND };

enum ncclSisciCommState {COMM_READY,
                         SEND_POSTED,
                         RECV_WAITING};

struct ncclSisciChannel {
    unsigned int id;
    enum ncclSisciCommState state;
    struct ncclSisciMailbox *mailbox;
    sci_desc_t sd;
    sci_dma_queue_t dq;
};

struct ncclSisciComm {
    enum ncclSisciCommType type;
    unsigned int mem_handle_cnt;
    unsigned int remote_node_offset;
    unsigned int remote_node_id;
    int remote_cuda_device;
    struct ncclSisciDev *dev;
    struct ncclSisciMailbox *mailbox;
    unsigned int request_cnt;
    struct ncclSisciChannel channels[CHANNELS];
};

#define ncclSisciRecvComm ncclSisciComm
#define ncclSisciSendComm ncclSisciComm

struct ncclSisciMemHandle {
    unsigned int id;
    sci_desc_t sd;
    sci_local_segment_t local_segment;
    sci_remote_segment_t remote_segment;
    unsigned int segment_id;
    unsigned int remote_segment_id;
    void *addr;
};

struct ncclSisciRequest {
    unsigned int id;
    enum ncclSisciCommType type;
    void *data;
    unsigned int size;
    unsigned int offset;
    struct ncclSisciMemHandle *memhandle;
    struct ncclSisciChannel *channel;
};

static unsigned int get_segment_id(unsigned int local_offset,
                                   unsigned int remote_offset,
                                   int cuda_device_id,
                                   unsigned int is_mailbox,
                                   enum ncclSisciCommType type,
                                   unsigned int i) {
    return (SEGMENT_PREFIX |
            (local_offset << 16) |
            (remote_offset << 8) |
            (cuda_device_id << 4) |
            (is_mailbox << 2) |
            (type << 1) |
            i);
}

void print_mailbox(volatile void* addr) {
    for (int i = 0; i < MAILBOX_SEGMENT_SIZE*MAX_NODES; i++) {
        printf("%d ", ((uint32_t*)addr)[i]);
    }
    printf("\n");
}

// Create a receiving object and provide a handle to connect to it. The
// handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
// between ranks to create a connection.
ncclResult_t ncclSisciListen(int dev, void* opaqueHandle, void** listenComm) {
    struct ncclSisciListenComm *comm;

    NCCLCHECK(ncclCalloc(&comm, 1));
    comm->dev = &ncclSisciDevs[dev];
    NCCLCHECK(ncclSCIOpen(&comm->sd, NO_FLAGS));
    *listenComm = comm;

    struct ncclSisciHandle* handle = (struct ncclSisciHandle*) opaqueHandle;
    static_assert(sizeof(struct ncclSisciHandle) < NCCL_NET_HANDLE_MAXSIZE,
                  "ncclSisciHandle size too large");
    handle->node_id = comm->dev->node_id;
    handle->cuda_device = cuda_device;

    NCCLCHECK(ncclSCICreateDataInterrupt(comm->sd, &comm->ir, comm->dev->adapter_no, &handle->irno,
                               NO_CALLBACK, NO_ARG, NO_FLAGS));

    INFO(NCCL_NET, "Listening on %d", handle->irno);

    return ncclSuccess;
}

static ncclResult_t ncclSisciCreateMailbox(struct ncclSisciComm *comm,
                                           struct ncclSisciMailbox **mailbox) {
    struct ncclSisciMailbox *mbox;

    NCCLCHECK(ncclCalloc(&mbox, 1));
    mbox->local_id = get_segment_id(comm->dev->node_offset,
                                    comm->remote_node_offset,
                                    cuda_device,
                                    IS_MAILBOX,
                                    0,
                                    comm->type);
    mbox->remote_id = get_segment_id(comm->remote_node_offset,
                                     comm->dev->node_offset,
                                     comm->remote_cuda_device,
                                     IS_MAILBOX,
                                     0,
                                     (comm->type == SISCI_SEND ? SISCI_RECV : SISCI_SEND));
    mbox->dev = comm->dev;
    mbox->remote_node_id = comm->remote_node_id;

    INFO(NCCL_NET, "Mailbox 0x%x connects to 0x%x", mbox->local_id, mbox->remote_id);

    NCCLCHECK(ncclSCIOpen(&mbox->sd, NO_FLAGS));
    NCCLCHECK(ncclSCICreateSegment(mbox->sd, &mbox->local_segment,
                                   mbox->local_id,
                                   MAILBOX_SEGMENT_SIZE*CHANNELS*sizeof(uint32_t),
                                   NO_CALLBACK, NO_ARG, NO_FLAGS));

    NCCLCHECK(ncclSCIPrepareSegment(mbox->local_segment, comm->dev->adapter_no,
                                    NO_FLAGS));

    NCCLCHECK(ncclSCISetSegmentAvailable(mbox->local_segment, comm->dev->adapter_no,
                                         NO_FLAGS));

    NCCLCHECK(ncclSCIMapLocalSegment(&mbox->local_addr,
                                     mbox->local_segment,
                                     &mbox->local_map,
                                     0,
                                     MAILBOX_SEGMENT_SIZE*MAX_NODES*sizeof(uint32_t),
                                     NULL,
                                     NO_FLAGS));

    *mailbox = mbox;

    return ncclSuccess;
}

static ncclResult_t ncclSisciRemoveMailbox(struct ncclSisciDev *dev,
                                           struct ncclSisciMailbox *mailbox) {
    if (mailbox->remote_segment != NULL) {
        NCCLCHECK(ncclSCIUnmapSegment(mailbox->remote_map, NO_FLAGS));
        NCCLCHECK(ncclSCIDisconnectSegment(mailbox->remote_segment, NO_FLAGS));
    }

    NCCLCHECK(ncclSCIUnmapSegment(mailbox->local_map, NO_FLAGS));
    NCCLCHECK(ncclSCISetSegmentUnavailable(mailbox->local_segment,
                                           dev->adapter_no,
                                           NO_FLAGS));
    NCCLCHECK(ncclSCIRemoveSegment(mailbox->local_segment, NO_FLAGS));
    NCCLCHECK(ncclSCIClose(mailbox->sd, NO_FLAGS));

    return ncclSuccess;
}

static ncclResult_t ncclSisciConnectMailbox(struct ncclSisciMailbox *mailbox) {
    while (ncclSCIConnectSegment(mailbox->sd, &mailbox->remote_segment,
                                 mailbox->remote_node_id,
                                 mailbox->remote_id, mailbox->dev->adapter_no,
                                 NO_CALLBACK, NO_ARG, SCI_INFINITE_TIMEOUT,
                                 NO_FLAGS) != ncclSuccess) {
        sleep(1);
    }

    NCCLCHECK(ncclSCIMapRemoteSegment(&mailbox->remote_addr,
                                      mailbox->remote_segment,
                                      &mailbox->remote_map,
                                      NO_OFFSET,
                                      MAILBOX_SEGMENT_SIZE*MAX_NODES*sizeof(uint32_t),
                                      NULL, NO_FLAGS));

    return ncclSuccess;
}

static ncclResult_t ncclSisciInitChannels(struct ncclSisciChannel *channels,
                                          struct ncclSisciMailbox *mailbox,
                                          enum ncclSisciCommType type) {
    for (unsigned int i = 0; i < CHANNELS; i++) {
        struct ncclSisciChannel *channel = channels + i;

        channel->id = i;
        channel->mailbox = mailbox;
        channel->state = COMM_READY;
        if (type == SISCI_SEND) {
            NCCLCHECK(ncclSCIOpen(&channel->sd, NO_FLAGS));
            NCCLCHECK(ncclSCICreateDMAQueue(channel->sd, &channel->dq,
                                            mailbox->dev->adapter_no,
                                            1, NO_FLAGS));
        }

    }

    return ncclSuccess;
}

static ncclResult_t ncclSisciRemoveChannels(struct ncclSisciChannel *channels) {
    for (unsigned int i = 0; i < CHANNELS; i++) {
        struct ncclSisciChannel *channel = channels + i;

        if (channel->dq != NULL) {
            NCCLCHECK(ncclSCIRemoveDMAQueue(channel->dq, NO_FLAGS));
            NCCLCHECK(ncclSCIClose(channel->sd, NO_FLAGS));
        }
    }

    return ncclSuccess;
}

static void mailbox_write(struct ncclSisciChannel *channel,
                          uint32_t command,
                          uint32_t value) {
    uint32_t *remote_command = (uint32_t*)channel->mailbox->remote_addr + channel->id*2;
    uint32_t *remote_value = remote_command + 1;

    *remote_value = value;
    *remote_command = command;
}

static int mailbox_read(struct ncclSisciChannel *channel,
                         uint32_t command,
                         uint32_t *value) {
    uint32_t *local_command = (uint32_t*)channel->mailbox->local_addr + channel->id*2;
    uint32_t *local_value = local_command + 1;

    if (*local_command == command) {
        if (value != NULL) {
            *value = *local_value;
        }

        *local_command = COMM_FLAG_EMPTY;
        return 1;
    }

    return 0;
}

// Connect to a handle and return a sending comm object for that peer.
ncclResult_t ncclSisciConnect(int dev, void* opaqueHandle, void** sendComm) {
    struct ncclSisciSendComm *comm;
    struct ncclSisciHandle* handle = (struct ncclSisciHandle*) opaqueHandle;

    NCCLCHECK(ncclCalloc(&comm, 1));
    comm->dev = &ncclSisciDevs[dev];
    comm->remote_node_id = handle->node_id;
    comm->remote_node_offset = (comm->remote_node_id >> 2) - 1;
    comm->type = SISCI_SEND;
    comm->remote_cuda_device = handle->cuda_device;

    INFO(NCCL_NET, "Connecting to node %d on %d", handle->node_id, handle->irno);

    sci_desc_t sd;
    sci_remote_data_interrupt_t ir;
    uint32_t data[IR_DATA_SIZE];
    data[0] = htons(comm->dev->node_offset);
    data[1] = htons(cuda_device);

    NCCLCHECK(ncclSCIOpen(&sd, NO_FLAGS));
    NCCLCHECK(ncclSCIConnectDataInterrupt(sd, &ir, handle->node_id,
                                comm->dev->adapter_no, handle->irno,
                                SCI_INFINITE_TIMEOUT, NO_FLAGS));
    NCCLCHECK(ncclSCITriggerDataInterrupt(ir, &data, sizeof(*data)*IR_DATA_SIZE,
                                          NO_FLAGS));
    NCCLCHECK(ncclSCIDisconnectDataInterrupt(ir, NO_FLAGS));
    NCCLCHECK(ncclSCIClose(sd, NO_FLAGS));

    NCCLCHECK(ncclSisciCreateMailbox(comm, &comm->mailbox));
    NCCLCHECK(ncclSisciInitChannels(comm->channels, comm->mailbox,
                                    comm->type));

    *sendComm = comm;

    return ncclSuccess;
}

// Finalize connection establishment after remote peer has called connectHandel
ncclResult_t ncclSisciAccept(void* listenComm, void** recvComm) {
    struct ncclSisciListenComm *lcomm = (struct ncclSisciListenComm*)listenComm;

    struct ncclSisciRecvComm *rcomm;

    uint32_t data[IR_DATA_SIZE];
    unsigned int size = IR_DATA_SIZE*sizeof(*data);

    NCCLCHECK(ncclCalloc(&rcomm, 1));
    rcomm->dev = lcomm->dev;
    rcomm->type = SISCI_RECV;

    NCCLCHECK(ncclSCIWaitForDataInterrupt(lcomm->ir, &data, &size, SCI_INFINITE_TIMEOUT,
                                            NO_FLAGS));
    rcomm->remote_node_offset = ntohs(data[0]);
    rcomm->remote_cuda_device = ntohs(data[1]);
    rcomm->remote_node_id = (rcomm->remote_node_offset+1)*4;

    NCCLCHECK(ncclSisciCreateMailbox(rcomm, &rcomm->mailbox));
    NCCLCHECK(ncclSisciInitChannels(rcomm->channels, rcomm->mailbox,
                                    rcomm->type));

    INFO(NCCL_NET, "Accepted connection from node %d", rcomm->remote_node_id);

    *recvComm = rcomm;

    return ncclSuccess;
}

ncclResult_t devptr_set_sync_memops(void* dev_ptr)
{
    unsigned flag = 1;
    CUresult err = cuPointerSetAttribute(&flag,
                                         CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                         (CUdeviceptr) dev_ptr);

    if (err != CUDA_SUCCESS)
    {
        WARN("Failed to set pointer attribute CU_POINTER_ATTRIBYTE_SYNC_MEMOPS");
        return ncclInternalError;
    }

    return ncclSuccess;
}

/* NCCL calls RegMr twice for each comm, with two pointers that point
   to the same memory block, but with different offsets and
   sizes. There is an overlapping page between these two areas, which
   for some reasons results in non-contiguous pages when using
   nvidia_p2p_get_pages in SCIAttachPhysicalMemory. As a workaround,
   we attach the whole CUDA memory block for both pointers, using
   offsets to read/write the correct memory location. This is a bit
   hackish, because we make som assumptions on the size and alignement
   of the GPU memory, causing any upstream changes will to this. */
#define GPU_SIZE          0x441000 /* We know this size from net.cc */
#define GPU_BOUND_SHIFT   20       /* The memory returned by
                                      cudaMalloc appears to always be
                                      1MB aligned. Not sure if this
                                      holds across platforms and
                                      version. */
#define GPU_BOUND_SIZE    ((uint64_t)1 << GPU_BOUND_SHIFT)
#define GPU_BOUND_OFFSET  (GPU_BOUND_SIZE-1)
#define GPU_BOUND_MASK    (~GPU_BOUND_OFFSET)

// Register/Deregister memory. Comm can be either a sendComm or a recvComm.
// Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
ncclResult_t ncclSisciRegMr(void* comm, void* data, int size, int type, void** mhandle) {
    struct ncclSisciComm *gcomm = (struct ncclSisciComm*)comm;


    struct ncclSisciMemHandle *memhandle;
    NCCLCHECK(ncclCalloc(&memhandle, 1));
    memhandle->id = gcomm->mem_handle_cnt++;
    memhandle->segment_id = get_segment_id(gcomm->dev->node_offset,
                                           gcomm->remote_node_offset,
                                           cuda_device,
                                           NOT_MAILBOX,
                                           memhandle->id,
                                           gcomm->type);

    memhandle->remote_segment_id = get_segment_id(gcomm->remote_node_offset,
                                                  gcomm->dev->node_offset,
                                                  gcomm->remote_cuda_device,
                                                  NOT_MAILBOX,
                                                  memhandle->id,
                                                  (gcomm->type == SISCI_SEND ? SISCI_RECV : SISCI_SEND));

    INFO(NCCL_NET, "RegMr: ptr=%p, size=0x%x, type=%s, local_segment=0x%x, remote_segment=0x%x",
         data, size, type == NCCL_PTR_HOST ? "host" : "cuda",
         memhandle->segment_id, memhandle->remote_segment_id);

    memhandle->addr = data;

    NCCLCHECK(ncclSCIOpen(&memhandle->sd, NO_FLAGS));

    if (type == NCCL_PTR_CUDA) {
        NCCLCHECK(devptr_set_sync_memops(memhandle->addr));

        /* Map the whole memory block */
        uint64_t start = (uint64_t)memhandle->addr & GPU_BOUND_MASK;
        size_t pin_size = GPU_SIZE;

        /* Set addr to start of block for correct offset calculations
           in send/recv. */
        memhandle->addr = (void*)start;

        INFO(NCCL_NET, "CUDA: aligned=%p, pin_size=0x%lx",
             (void*)start, pin_size);

        NCCLCHECK(ncclSCICreateSegment(memhandle->sd, &memhandle->local_segment,
                                       memhandle->segment_id, pin_size,
                                       NO_CALLBACK, NO_ARG, SCI_FLAG_EMPTY));

        NCCLCHECK(ncclSCIAttachPhysicalMemory(0, (void*)start, 0, pin_size,
                                              memhandle->local_segment,
                                              SCI_FLAG_CUDA_BUFFER));

    } else {
        NCCLCHECK(ncclSCICreateSegment(memhandle->sd, &memhandle->local_segment,
                                       memhandle->segment_id, size,
                                       NO_CALLBACK, NO_ARG, SCI_FLAG_EMPTY));

        NCCLCHECK(ncclSCIRegisterSegmentMemory(memhandle->addr, size,
                                               memhandle->local_segment,
                                               SCI_FLAG_LOCK_USER_MEM));
    }

    NCCLCHECK(ncclSCIPrepareSegment(memhandle->local_segment, gcomm->dev->adapter_no,
                                    NO_FLAGS));
    NCCLCHECK(ncclSCISetSegmentAvailable(memhandle->local_segment, gcomm->dev->adapter_no,
                                         NO_FLAGS));

    *mhandle = memhandle;

    return ncclSuccess;
}
ncclResult_t ncclSisciDeregMr(void* comm, void* mhandle) {
    struct ncclSisciMemHandle *memhandle = (struct ncclSisciMemHandle*)mhandle;
    struct ncclSisciComm *gcomm = (struct ncclSisciComm*)comm;

    if (memhandle->remote_segment != NULL) {
        NCCLCHECK(ncclSCIDisconnectSegment(memhandle->remote_segment, NO_FLAGS));
    }

    NCCLCHECK(ncclSCISetSegmentUnavailable(memhandle->local_segment,
                                           gcomm->dev->adapter_no,
                                           NO_FLAGS));

    while (ncclSCIRemoveSegment(memhandle->local_segment, NO_FLAGS) !=
           ncclSuccess) {}

    NCCLCHECK(ncclSCIClose(memhandle->sd, NO_FLAGS));

    free(mhandle);

    return ncclSuccess;
}

// Asynchronous send to a peer.
// May return request == NULL if the call cannot be performed (or would block)
ncclResult_t ncclSisciIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
    struct ncclSisciSendComm *comm = (struct ncclSisciSendComm*)sendComm;
    struct ncclSisciMemHandle *memhandle = (struct ncclSisciMemHandle*)mhandle;
    struct ncclSisciRequest *req;
    size_t local_offset = (uint8_t*)data - (uint8_t*)memhandle->addr;
    struct ncclSisciChannel *channel = &comm->channels[comm->request_cnt % CHANNELS];

    if (comm->mailbox->remote_addr == NULL) {
        NCCLCHECK(ncclSisciConnectMailbox(comm->mailbox));
    }

    *request = NULL;

    if (channel->state != COMM_READY) {
        return ncclSuccess;
    }

    uint32_t remote_offset = 0;

    if (!mailbox_read(channel, COMM_FLAG_REQUEST, &remote_offset)) {
        return ncclSuccess;
    }

    NCCLCHECK(ncclCalloc(&req, 1));

    if (memhandle->remote_segment == NULL) {
        while (ncclSCIConnectSegment(memhandle->sd, &memhandle->remote_segment,
                                       comm->remote_node_id, memhandle->remote_segment_id,
                                       comm->dev->adapter_no, NO_CALLBACK, NO_ARG,
                                       SCI_INFINITE_TIMEOUT, NO_FLAGS) != ncclSuccess) {
            sleep(1);
        }
    }

    req->type = SISCI_SEND;
    req->id = comm->request_cnt++;
    req->data = data;
    req->size = size;
    req->memhandle = memhandle;
    req->offset = local_offset;

    INFO(NCCL_NET, "Sending request %d: node=%d, size=0x%x, local_offset=0x%x, remote_offset=0x%x, local_segment=0x%x, remote_segment=0x%x",
         req->id, comm->remote_node_id, size, local_offset, remote_offset,
         memhandle->segment_id, memhandle->remote_segment_id);

    if (size > 0) {
        NCCLCHECK(ncclSCIStartDmaTransfer(channel->dq, memhandle->local_segment,
                                          memhandle->remote_segment, local_offset, size,
                                          remote_offset, NO_CALLBACK, NO_ARG, NO_FLAGS));
    }

    channel->state = SEND_POSTED;
    req->channel = channel;

    *request = req;

    return ncclSuccess;
}

// Asynchronous recv from a peer.
// May return request == NULL if the call cannot be performed (or would block)
ncclResult_t ncclSisciIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
    struct ncclSisciRequest *req;
    struct ncclSisciMemHandle *memhandle = (struct ncclSisciMemHandle*)mhandle;
    struct ncclSisciRecvComm *comm = (struct ncclSisciRecvComm*)recvComm;
    size_t offset = (uint8_t*)data - (uint8_t*)memhandle->addr;
    struct ncclSisciChannel *channel = &comm->channels[comm->request_cnt % CHANNELS];

    if (comm->mailbox->remote_addr == NULL) {
        NCCLCHECK(ncclSisciConnectMailbox(comm->mailbox));
    }

    INFO(NCCL_NET, "Try recv: state=%u, memory_id=%d",
         channel->state,
         memhandle->id);

    *request = NULL;

    if (channel->state != COMM_READY) {
        return ncclSuccess;
    }

    mailbox_write(channel, COMM_FLAG_REQUEST, offset);

    NCCLCHECK(ncclCalloc(&req, 1));

    req->type = SISCI_RECV;
    req->id = comm->request_cnt++;
    req->data = data;
    req->size = size;
    req->memhandle = memhandle;
    req->offset = offset;

    channel->state = RECV_WAITING;
    req->channel = channel;

    INFO(NCCL_NET, "Receiving request %d: node=%d, size=0x%x, offset=0x%x, segment=0x%x",
         req->id, comm->remote_node_id, size, offset, memhandle->segment_id);

    *request = req;

    return ncclSuccess;
}

// Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
// visible to the GPU
ncclResult_t ncclSisciFlush(void* recvComm, void* data, int size, void* mhandle) {
    UNUSED(recvComm);
    UNUSED(data);
    UNUSED(size);
    UNUSED(mhandle);

    /* This is a no-op now, but not sure if we should be doing something here. */

    return ncclSuccess;
}

// Test whether a request is complete. If size is not NULL, it returns the
// number of bytes sent/received.
ncclResult_t ncclSisciTest(void* request, int* done, int* size) {
    struct ncclSisciRequest *req = (struct ncclSisciRequest*)request;
    *done = 0;

    if (req->type == SISCI_SEND) {
        if (req->channel->state == SEND_POSTED) {
            sci_dma_queue_state_t state;
            NCCLCHECK(ncclSCIDMAQueueState(&state, req->channel->dq));
            if (state == SCI_DMAQUEUE_IDLE || state == SCI_DMAQUEUE_DONE) {
                INFO(NCCL_NET, "req->id=%d, SEND_POSTED->COMM_READY",
                     req->id);

                mailbox_write(req->channel, COMM_FLAG_NOTIFY, req->size);
                req->channel->state = COMM_READY;

                *done = 1;
            }
        }
    }
    else {
        if (req->channel->state == RECV_WAITING &&
            mailbox_read(req->channel, COMM_FLAG_NOTIFY, &req->size)) {

            req->channel->state = COMM_READY;

            *done = 1;

            INFO(NCCL_NET, "req->id=%d, RECV_WAITING->COMM_READY",
                 req->id);
            INFO(NCCL_NET, "Received data: size=0x%x, offset=0x%x",
                 req->size, req->offset);
        }
    }

    if (size) *size = req->size;

    if (*done) {
        free(request);
    }

    return ncclSuccess;
}

// Close and free send/recv comm objects
ncclResult_t ncclSisciCloseSend(void* sendComm) {
    struct ncclSisciSendComm *comm = (struct ncclSisciSendComm*)sendComm;

    NCCLCHECK(ncclSisciRemoveMailbox(comm->dev, comm->mailbox));
    NCCLCHECK(ncclSisciRemoveChannels(comm->channels));

    free(sendComm);

    return ncclSuccess;
}
ncclResult_t ncclSisciCloseRecv(void* recvComm) {
    struct ncclSisciRecvComm *comm = (struct ncclSisciRecvComm*)recvComm;

    NCCLCHECK(ncclSisciRemoveMailbox(comm->dev, comm->mailbox));
    NCCLCHECK(ncclSisciRemoveChannels(comm->channels));

    free(recvComm);

    return ncclSuccess;
}
ncclResult_t ncclSisciCloseListen(void* listenComm) {
    struct ncclSisciListenComm *comm = (struct ncclSisciListenComm*)listenComm;

    NCCLCHECK(ncclSCIRemoveDataInterrupt(comm->ir, NO_FLAGS));
    NCCLCHECK(ncclSCIClose(comm->sd, NO_FLAGS));

    free(listenComm);

    return ncclSuccess;
}

ncclNet_t NCCL_PLUGIN_SYMBOL = {
  "Sisci",
  ncclSisciInit,
  ncclSisciDevices,
  ncclSisciPciPath,
  ncclSisciPtrSupport,
  ncclSisciListen,
  ncclSisciConnect,
  ncclSisciAccept,
  ncclSisciRegMr,
  ncclSisciDeregMr,
  ncclSisciIsend,
  ncclSisciIrecv,
  ncclSisciFlush,
  ncclSisciTest,
  ncclSisciCloseSend,
  ncclSisciCloseRecv,
  ncclSisciCloseListen
};
