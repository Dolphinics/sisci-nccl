#include "sisciwrap.h"
#include "sisci_nccl.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
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
#define INFINITE_TIMEOUT      0xffffffff
#define MAX_NODES             60
#define SEGMENT_PREFIX        0xbaba0000
#define MAILBOX_SEGMENT_ID    SEGMENT_PREFIX | 1
#define MEMORY_SEGMENTS       2
#define MEMORY_SEGMENT_PREFIX SEGMENT_PREFIX | 0x0000ba00
#define REQUEST_BUFFER_SIZE   32

#define COMM_FLAG_EMPTY   0
#define COMM_FLAG_REQUEST 1
#define COMM_FLAG_NOTIFY  2

struct ncclSisciDev {
    unsigned int available;
    unsigned int adapter_no;
    unsigned int node_id;
    unsigned int node_offset;
};

struct ncclSisciDev ncclSisciDevs[MAX_SCI_DEVS];

pthread_mutex_t ncclSisciLock = PTHREAD_MUTEX_INITIALIZER;

// Initialize the network.
ncclResult_t ncclSisciInit(ncclDebugLogger_t logFunction) {
    ncclDebugLog = logFunction;

    INFO(NCCL_NET|NCCL_INIT, "Trying to load SISCI");
    NCCLCHECK(ncclSCIInitialize(NO_FLAGS));

    for (int i = 0; i < MAX_SCI_DEVS; i++) {
        struct ncclSisciDev *dev = &ncclSisciDevs[i];

        dev->adapter_no = i;

        if (ncclSCIGetLocalNodeId(dev->adapter_no, &dev->node_id, NO_FLAGS) ==
            ncclSuccess) {
            INFO(NCCL_INIT|NCCL_NET, "NET/SISCI : adapter %u, node id %u",
                 dev->adapter_no, dev->node_id);

            dev->node_offset = (dev->node_id >> 2) - 1;

            dev->available = 1;
        }
        else {
            break;
        }
    }

    return ncclSuccess;
}

// Return the number of adapters.
ncclResult_t ncclSisciDevices(int* ndev) {
    for (int i = 0; i < MAX_SCI_DEVS; i++) {
        if (ncclSisciDevs[i].available == 0) {
            *ndev = i;
            break;
        }
    }

    return ncclSuccess;
}

// Return the device path in /sys. NCCL will call free on this path.
ncclResult_t ncclSisciPciPath(int dev, char** path) {
    UNUSED(dev);
    char devicepath[PATH_MAX];
    strcpy(devicepath, "/sys/class/pci_bus/0000:da/device/0000:da:00.0/");
    *path = realpath(devicepath, NULL);

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
};

struct ncclSisciListenComm {
    struct ncclSisciDev *dev;
    sci_desc_t sd;
    sci_local_data_interrupt_t ir;
    sci_local_segment_t segment;
    sci_map_t map;
    volatile void *addr;
};

struct ncclSisciMailbox {
    sci_desc_t sd;
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
};

struct ncclSisciComm {
    sci_desc_t sd;
    enum ncclSisciCommType type;
    unsigned int mem_handle_cnt;
    unsigned int remote_node_offset;
    unsigned int remote_node_id;
    struct ncclSisciDev *dev;
    struct ncclSisciMailbox *mailbox;
    sci_dma_queue_t dq;
    unsigned int request_cnt;
    struct ncclSisciChannel channels[REQUEST_BUFFER_SIZE];
};

#define ncclSisciRecvComm ncclSisciComm
#define ncclSisciSendComm ncclSisciComm

struct ncclSisciMemHandle {
    sci_desc_t sd;
    sci_local_segment_t local_segment;
    sci_remote_segment_t remote_segment;
    unsigned int segment_id;
    unsigned int remote_segment_id;
    unsigned int memory_id;
    sci_map_t map;
    volatile void *segment_addr;
    void *addr;
    unsigned int busy;
    uint64_t offset;
    int type;
};

struct ncclSisciRequest {
    enum ncclSisciCommType type;
    void *comm;
    unsigned int memory_id;
    unsigned int id;
    void *data;
    unsigned int size;
    unsigned int offset;
    struct ncclSisciMemHandle *memhandle;
    struct ncclSisciChannel *channel;
};

static unsigned int memory_segment_id(unsigned int node_offset,
                                      enum ncclSisciCommType type,
                                      unsigned int i) {
    return MEMORY_SEGMENT_PREFIX | (type << 8) | (node_offset << 1) | i;
}

static unsigned int get_mailbox_id(enum ncclSisciCommType type,
                                   unsigned int i) {
    return SEGMENT_PREFIX | (type << 8) | i;
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

    NCCLCHECK(ncclSCICreateDataInterrupt(comm->sd, &comm->ir, comm->dev->adapter_no, &handle->irno,
                               NO_CALLBACK, NO_ARG, NO_FLAGS));

    return ncclSuccess;
}

static ncclResult_t ncclSisciCreateMailbox(struct ncclSisciDev *dev,
                                           struct ncclSisciMailbox **mailbox,
                                           unsigned int segment_id) {
    struct ncclSisciMailbox *mbox;

    NCCLCHECK(ncclCalloc(&mbox, 1));
    NCCLCHECK(ncclSCIOpen(&mbox->sd, NO_FLAGS));
    NCCLCHECK(ncclSCICreateSegment(mbox->sd, &mbox->local_segment,
                                     segment_id,
                                     MAILBOX_SEGMENT_SIZE*REQUEST_BUFFER_SIZE*sizeof(uint32_t),
                                     NO_CALLBACK, NO_ARG, NO_FLAGS));

    NCCLCHECK(ncclSCIPrepareSegment(mbox->local_segment, dev->adapter_no,
                                      NO_FLAGS));

    NCCLCHECK(ncclSCISetSegmentAvailable(mbox->local_segment, dev->adapter_no,
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
    NCCLCHECK(ncclSCIUnmapSegment(mailbox->remote_map, NO_FLAGS));
    NCCLCHECK(ncclSCIDisconnectSegment(mailbox->remote_segment, NO_FLAGS));

    NCCLCHECK(ncclSCIUnmapSegment(mailbox->local_map, NO_FLAGS));
    NCCLCHECK(ncclSCISetSegmentUnavailable(mailbox->local_segment,
                                           dev->adapter_no,
                                           NO_FLAGS));
    NCCLCHECK(ncclSCIRemoveSegment(mailbox->local_segment, NO_FLAGS));
    NCCLCHECK(ncclSCIClose(mailbox->sd, NO_FLAGS));

    return ncclSuccess;
}

static ncclResult_t ncclSisciConnectMailbox(struct ncclSisciDev *dev,
                                            struct ncclSisciMailbox *mailbox,
                                            unsigned int segment_id,
                                            unsigned int remote_node) {
    while (ncclSCIConnectSegment(mailbox->sd, &mailbox->remote_segment, remote_node,
                                   segment_id, dev->adapter_no,
                                   NO_CALLBACK, NO_ARG, INFINITE_TIMEOUT,
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

static void ncclSisciInitChannels(struct ncclSisciChannel *channels,
                                  struct ncclSisciMailbox *mailbox) {
    for (unsigned int i = 0; i < REQUEST_BUFFER_SIZE; i++) {
        struct ncclSisciChannel *channel = channels + i;

        channel->id = i;
        channel->mailbox = mailbox;
        channel->state = COMM_READY;
    }
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

    sci_remote_data_interrupt_t ir;
    uint32_t data = htons(comm->dev->node_offset);

    NCCLCHECK(ncclSCIOpen(&comm->sd, NO_FLAGS));
    NCCLCHECK(ncclSCIConnectDataInterrupt(comm->sd, &ir, handle->node_id,
                                comm->dev->adapter_no, handle->irno,
                                INFINITE_TIMEOUT, NO_FLAGS));
    NCCLCHECK(ncclSCITriggerDataInterrupt(ir, &data, sizeof(data), NO_FLAGS));
    NCCLCHECK(ncclSCIDisconnectDataInterrupt(ir, NO_FLAGS));

    NCCLCHECK(ncclSisciCreateMailbox(comm->dev, &comm->mailbox,
                                     get_mailbox_id(SISCI_SEND,
                                                    comm->remote_node_offset)));
    ncclSisciInitChannels(comm->channels, comm->mailbox);

    NCCLCHECK(ncclSCICreateDMAQueue(comm->sd, &comm->dq, comm->dev->adapter_no,
                                      128, NO_FLAGS));

    *sendComm = comm;

    return ncclSuccess;
}

// Finalize connection establishment after remote peer has called connectHandel
ncclResult_t ncclSisciAccept(void* listenComm, void** recvComm) {
    struct ncclSisciListenComm *lcomm = (struct ncclSisciListenComm*)listenComm;

    struct ncclSisciRecvComm *rcomm;

    uint32_t data;
    unsigned int size = sizeof(data);

    NCCLCHECK(ncclCalloc(&rcomm, 1));
    rcomm->dev = lcomm->dev;
    rcomm->type = SISCI_RECV;

    NCCLCHECK(ncclSCIWaitForDataInterrupt(lcomm->ir, &data, &size, INFINITE_TIMEOUT,
                                            NO_FLAGS));
    rcomm->remote_node_offset = ntohs(data);
    rcomm->remote_node_id = (rcomm->remote_node_offset+1)*4;

    NCCLCHECK(ncclSisciCreateMailbox(rcomm->dev, &rcomm->mailbox,
                                     get_mailbox_id(SISCI_RECV,
                                                    rcomm->remote_node_offset)));
    ncclSisciInitChannels(rcomm->channels, rcomm->mailbox);


    *recvComm = rcomm;

    return ncclSuccess;
}

void devptr_set_sync_memops(void* dev_ptr)
{
    unsigned flag = 1;

    CUresult err = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr) dev_ptr);

    if (err != CUDA_SUCCESS)
    {
        INFO(NCCL_INIT, "Failed to set pointer attribute CU_POINTER_ATTRIBYTE_SYNC_MEMOPS");
    }
}

void* devptr(void* ptr)
{
    struct cudaPointerAttributes attrs;
    int gpu = 1;

    cudaError_t err = cudaSetDevice(gpu);
    if (err != cudaSuccess)
    {
        WARN("Failed to set GPU: %s", cudaGetErrorString(err));
        return NULL;
    }

    err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess)
    {
        WARN("Failed to get pointer attributes: %s", cudaGetErrorString(err));
        return NULL;
    }

    INFO(NCCL_INIT, "CUDA device buffer %p has device ptr %p", ptr, attrs.devicePointer);
    return attrs.devicePointer;
}

#define GPU_BOUND_SHIFT   16
#define GPU_BOUND_SIZE    ((uint64_t)1 << GPU_BOUND_SHIFT)
#define GPU_BOUND_OFFSET  (GPU_BOUND_SIZE-1)
#define GPU_BOUND_MASK    (~GPU_BOUND_OFFSET)

// Register/Deregister memory. Comm can be either a sendComm or a recvComm.
// Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
ncclResult_t ncclSisciRegMr(void* comm, void* data, int size, int type, void** mhandle) {
    struct ncclSisciComm *gcomm = (struct ncclSisciComm*)comm;

    printf("RegMr: ptr=%p, size=0x%x, type=%s\n",
           data, size, type == NCCL_PTR_HOST ? "host" : "cuda");

    struct ncclSisciMemHandle *memhandle;
    NCCLCHECK(ncclCalloc(&memhandle, 1));
    memhandle->memory_id = gcomm->mem_handle_cnt++;
    memhandle->segment_id = memory_segment_id(gcomm->remote_node_offset,
                                              gcomm->type,
                                              memhandle->memory_id);
    memhandle->remote_segment_id = memory_segment_id(gcomm->dev->node_offset,
                                                     (gcomm->type == SISCI_SEND ? SISCI_RECV : SISCI_SEND),
                                                     memhandle->memory_id);
    // memhandle->addr = devptr(data);
    memhandle->addr = data;
    memhandle->type = type;

    NCCLCHECK(ncclSCIOpen(&memhandle->sd, NO_FLAGS));

    if (type == NCCL_PTR_CUDA) {

        // memhandle->addr = devptr(data);
        void *dptr = devptr(memhandle->addr);
        devptr_set_sync_memops(dptr);
        uint64_t start = (uint64_t)dptr & GPU_BOUND_MASK;
        size_t pin_size = (uint64_t)dptr + size - start;
        memhandle->offset = (uint64_t)dptr - start;

        memhandle->addr = (void*)start;

        printf("CUDA: aligned=%p, pin_size=0x%lx, offset=0x%lx\n",
               (void*)start, pin_size, memhandle->offset);

        NCCLCHECK(ncclSCICreateSegment(memhandle->sd, &memhandle->local_segment,
                                       memhandle->segment_id, size,
                                       NO_CALLBACK, NO_ARG, SCI_FLAG_EMPTY));

        NCCLCHECK(ncclSCIAttachPhysicalMemory(0, (void*)start, 0, size,
                                              memhandle->local_segment,
                                              SCI_FLAG_CUDA_BUFFER));

    } else {
        NCCLCHECK(ncclSCICreateSegment(memhandle->sd, &memhandle->local_segment,
                                       memhandle->segment_id, size,
                                       NO_CALLBACK, NO_ARG, SCI_FLAG_EMPTY));

        NCCLCHECK(ncclSCIRegisterSegmentMemory(memhandle->addr, size,
                                               memhandle->local_segment,
                                               NO_FLAGS));
        memhandle->offset = 0;
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

uint16_t fletcher16( uint8_t *data, int count )
{
   uint16_t sum1 = 0;
   uint16_t sum2 = 0;
   int index;

   for ( index = 0; index < count; ++index )
   {
      sum1 = (sum1 + data[index]) % 255;
      sum2 = (sum2 + sum1) % 255;
   }

   return (sum2 << 8) | sum1;
}

// Asynchronous send to a peer.
// May return request == NULL if the call cannot be performed (or would block)
ncclResult_t ncclSisciIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
    struct ncclSisciSendComm *comm = (struct ncclSisciSendComm*)sendComm;
    struct ncclSisciMemHandle *memhandle = (struct ncclSisciMemHandle*)mhandle;
    struct ncclSisciRequest *req;
    size_t local_offset = (uint8_t*)data - (uint8_t*)memhandle->addr;
    struct ncclSisciChannel *channel = &comm->channels[comm->request_cnt % REQUEST_BUFFER_SIZE];

    if (comm->mailbox->remote_addr == NULL) {
        NCCLCHECK(ncclSisciConnectMailbox(comm->dev, comm->mailbox,
                                          get_mailbox_id(SISCI_RECV, comm->dev->node_offset),
                                          comm->remote_node_id));
    }

    printf("Try send: state=%u, memory_id=%d, req->id=%u\n",
           channel->state,
           memhandle->memory_id,
           comm->request_cnt);

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

    if (size > 0) {
        NCCLCHECK(ncclSCIStartDmaTransfer(comm->dq, memhandle->local_segment,
                                            memhandle->remote_segment, local_offset, size,
                                            remote_offset, NO_CALLBACK, NO_ARG, NO_FLAGS));
    }

    req->type = SISCI_SEND;
    req->comm = sendComm;
    req->memory_id = memhandle->memory_id;
    req->id = comm->request_cnt++;
    req->data = data;
    req->size = size;
    req->memhandle = memhandle;
    req->offset = local_offset;

    channel->state = SEND_POSTED;
    req->channel = channel;

    uint16_t checksum = 0;

    if (memhandle->type == NCCL_PTR_HOST) {
        checksum = fletcher16((uint8_t*)data, size);
    }

    printf("Sending request %d: size=%d, local_offset=%lu, remote_offset=%u, local_segment=%x, remote_segment=%x, checksum=%04x\n",
           req->id, size, local_offset, remote_offset, memhandle->segment_id,
           memhandle->remote_segment_id, checksum);

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
    struct ncclSisciChannel *channel = &comm->channels[comm->request_cnt % REQUEST_BUFFER_SIZE];

    if (comm->mailbox->remote_addr == NULL) {
        NCCLCHECK(ncclSisciConnectMailbox(comm->dev, comm->mailbox,
                                          get_mailbox_id(SISCI_SEND, comm->dev->node_offset),
                                          comm->remote_node_id));
    }

    printf("Try recv: state=%u, memory_id=%d\n",
           channel->state,
           memhandle->memory_id);

    *request = NULL;

    if (channel->state != COMM_READY) {
        return ncclSuccess;
    }

    mailbox_write(channel, COMM_FLAG_REQUEST, offset);

    NCCLCHECK(ncclCalloc(&req, 1));

    req->type = SISCI_RECV;
    req->comm = recvComm;
    req->memory_id = memhandle->memory_id;
    req->id = comm->request_cnt++;
    req->data = data;
    req->size = size;
    req->memhandle = memhandle;
    req->offset = offset;

    channel->state = RECV_WAITING;
    req->channel = channel;

    printf("Receiving request %d: size=%d, offset=%lu, segment=%x\n",
           req->id, size, offset, memhandle->segment_id);

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

    return ncclInternalError;
}

// Test whether a request is complete. If size is not NULL, it returns the
// number of bytes sent/received.
ncclResult_t ncclSisciTest(void* request, int* done, int* size) {
    struct ncclSisciRequest *req = (struct ncclSisciRequest*)request;
    *done = 0;

    if (req->type == SISCI_SEND) {
        struct ncclSisciSendComm *comm = (struct ncclSisciSendComm*)req->comm;

        if (req->channel->state == SEND_POSTED) {
            sci_dma_queue_state_t state;
            NCCLCHECK(ncclSCIDMAQueueState(&state, comm->dq));
            if (state == SCI_DMAQUEUE_IDLE || state == SCI_DMAQUEUE_DONE) {
                printf("req->id=%d, SEND_POSTED->COMM_READY\n",
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

            printf("req->id=%d, RECV_WAITING->COMM_READY\n",
                   req->id);

            uint16_t checksum = 0;

            if (req->memhandle->type == NCCL_PTR_HOST) {
                checksum = fletcher16((uint8_t*)req->data, req->size);
            }

            printf("Received data: size=%u, offset=%u, checksum=%04x\n",
                   req->size, req->offset, checksum);
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
    NCCLCHECK(ncclSCIRemoveDMAQueue(comm->dq, NO_FLAGS));
    NCCLCHECK(ncclSCIClose(comm->sd, NO_FLAGS));

    free(sendComm);

    return ncclSuccess;
}
ncclResult_t ncclSisciCloseRecv(void* recvComm) {
    struct ncclSisciRecvComm *comm = (struct ncclSisciRecvComm*)recvComm;

    NCCLCHECK(ncclSisciRemoveMailbox(comm->dev, comm->mailbox));

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
