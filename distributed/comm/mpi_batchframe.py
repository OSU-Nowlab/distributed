"""
This is an MVAPICH2-based (http://mvapich.cse.ohio-state.edu) communication 
backend for the Dask Distributed library.

This is based on the UCX device. UCX: https://github.com/openucx/ucx
"""
from __future__ import annotations
from mpi4py import MPI
from contextlib import suppress
import functools
import logging
import os
import struct
import warnings
import weakref
from collections.abc import Awaitable, Callable, Collection
from typing import TYPE_CHECKING, Any

import dask
from dask.utils import parse_bytes
import asyncio
import socket

import random
import time
import sys
import datetime
from .addressing import parse_host_port, unparse_host_port
from .core import Comm, Connector, Listener, CommClosedError
from .registry import Backend, backends
from .utils import (
    ensure_concrete_host,
    from_frames,
    host_array,
    to_frames,
)
from ..utils import (
    ensure_ip,
    get_ip,
    get_ipv6,
    nbytes,
    log_errors,
    CancelledError,
    parse_bytes,
    nbytes
)

import numpy as np
logger = logging.getLogger(__name__)

# logging.basicConfig(level=logging.INFO)


device_array = None
pre_existing_cuda_context = False
cuda_context_created = False

_warning_suffix = (
    "This is often the result of a CUDA-enabled library calling a CUDA runtime function before "
    "Dask-CUDA can spawn worker processes. Please make sure any such function calls don't happen "
    "at import time or in the global scope of a program."
)

# logger.setLevel(logging.INFO)

def _warn_existing_cuda_context(ctx, pid):
    warnings.warn(
        f"A CUDA context for device {ctx} already exists on process ID {pid}. {_warning_suffix}"
    )


def _warn_cuda_context_wrong_device(ctx_expected, ctx_actual, pid):
    warnings.warn(
        f"Worker with process ID {pid} should have a CUDA context assigned to device "
        f"{ctx_expected}, but instead the CUDA context is on device {ctx_actual}. "
        f"{_warning_suffix}"
    )

INITIAL_TAG_OFFSET = 100
TAG_QUOTA_PER_CONNECTION = 15000000

# workers and the scheduler randomly select ports for listening to 
# incoming connections between this range
LOWER_PORT_RANGE=30000 
UPPER_PORT_RANGE=40000

# This constant is used to split large message into chunks of this size
CHUNK_SIZE = 2 ** 30

initialized = False
print_rank = 0

# The tag_table dictionary is used to maintain 'tag offset' for a pair of 
# processes. Imagine a Dask program executing with 4 MPI processes - a scheduler,
# a client, and 2 workers. The MPI.COMM_WORLD communicator is used by all for
# exchanging messages. However, during the connection establishment, all processes
# need to make sure that they have a unique tag range to use with MPI.COMM_WORLD
# for a pair of processes. Otherwise there is a possibility of message interference 
# because processes connect with one another dynamically and also there are 
# control and data channels. This situation is avoided using this dictionary
# of tags.
tag_table = dict()
recv_count_times = dict()
send_count_times = dict()

def synchronize_stream(stream=0):
    import numba.cuda

    ctx = numba.cuda.current_context()
    cu_stream = numba.cuda.driver.drvapi.cu_stream(stream)
    stream = numba.cuda.driver.Stream(ctx, cu_stream, None)
    stream.synchronize()

def init_tag_table():
    """ this function initializes tag_table dictionary."""
    global tag_table, INITIAL_TAG_OFFSET, recv_count_times, send_count_times
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # each process maintains an entry for every other process in the tag_table
    recv_count_times.update({rank: 0})
    send_count_times.update({rank: 0})
    recv_count_times.update({"cum_size": 0})
    send_count_times.update({"cum_size": 0})
    recv_count_times.update({"cum_time": 0})
    send_count_times.update({"cum_time": 0})
    recv_count_times.update({"max": 0})
    send_count_times.update({"max": 0})
    for peer in range(size):
        if rank != peer: 
            tag_table.update({peer : INITIAL_TAG_OFFSET})

    logger.debug(tag_table)

def init_once():
    global initialized, host_array, device_array
    global pre_existing_cuda_context, cuda_context_created

    if initialized is True:
        return
     
    initialized = True

    random.seed(random.randrange(sys.maxsize))

    name = MPI.Get_processor_name() # rank=5, name=gpu26.cluster
    init_tag_table()
    # logger.info("rank=%s, name=%s", MPI.COMM_WORLD.Get_rank(), name)
    pool_size_str = dask.config.get("distributed.rmm.pool-size")
    try:
        global rmm
        import rmm

        def device_array(n):
            return rmm.DeviceBuffer(size=n)

        if pool_size_str is not None:
            pool_size = parse_bytes(pool_size_str)
            rmm.reinitialize(
                pool_allocator=True, managed_memory=False, initial_pool_size=pool_size
            )
    except ImportError:
        try:
            import numba.cuda

            def numba_device_array(n):
                a = numba.cuda.device_array((n,), dtype="u1")
                weakref.finalize(a, numba.cuda.current_context)
                return a

            device_array = numba_device_array

        except ImportError:

            def device_array(n):
                raise RuntimeError(
                    "In order to send/recv CUDA arrays, Numba or RMM is required"
                )

    if pool_size_str is not None:
            warnings.warn(
                "Initial RMM pool size defined, but RMM is not available. "
                "Please consider installing RMM or removing the pool size option."
            )


def _close_comm(ref):
    """Callback to close Dask Comm when UCX Endpoint closes or errors

    Parameters
    ----------
        ref: weak reference to a Dask UCX comm
    """
    comm = ref()
    if comm is not None:
        comm._closed = True



class MPI4Dask(Comm):
    """Comm object using MPI.

    Parameters
    ----------
    peer_rank : int
        The MPI rank of the peer process
    suggested_tag : int
        The suggested tag offet to avoid message interference
    local_addr : str
        The local address, prefixed with `mpi://`
    peer_addr : str
        The peer address, prefixed with `mpi://`
    p_addr : str
        The peer address, prefixed with `mpi://`
    deserialize : bool, default True
        Whether to deserialize data in :meth:`distributed.protocol.loads`

    Notes
    -----
    The read-write cycle uses the following pattern:

    Each msg is serialized into a number of "data" frames. We prepend these
    real frames with two additional frames

        1. is_gpus: Boolean indicator for whether the frame should be
           received into GPU memory. Packed in '?' format. Unpack with
           ``<n_frames>?`` format.
        2. frame_size : Unsigned int describing the size of frame (in bytes)
           to receive. Packed in 'Q' format, so a length-0 frame is equivalent
           to an unsized frame. Unpacked with ``<n_frames>Q``.

    The expected read cycle is

    1. Read the frame describing if connection is closing and number of frames
    2. Read the frame describing whether each data frame is gpu-bound
    3. Read the frame describing whether each data frame is sized
    4. Read all the data frames.
    """

    def __init__(self, peer_rank: int, suggested_tag: int, local_addr: str, \
                                           peer_addr: str, p_addr: str, deserialize: bool = True):
                                                    
        Comm.__init__(self)
        self._suggested_tag = suggested_tag
        self._peer_rank = peer_rank
        if local_addr:
            assert local_addr.startswith("mpi")
        assert peer_addr.startswith("mpi")
        self._local_addr = local_addr
        self._peer_addr = peer_addr
        self._p_addr = p_addr
        self.deserialize = deserialize
        self._closed = False
        logger.debug("MPI4Dask.__init__ %s", self)

    @property
    def suggested_tag(self) -> int:
        return self._suggested_tag

    @property
    def peer_rank(self) -> int:
        return self._peer_rank

    @property
    def local_address(self) -> str:
        return self._local_addr

    @property
    def peer_address(self) -> str:
        return self._peer_addr

    @log_errors
    async def write(
        self,
        msg: dict,
        serializers=("cuda", "dask", "pickle", "error"),
        on_error: str = "message",
    ):
        logger.debug("write() function is called")
        # if self.closed():
            #logger.debug("inside self.closed()")
            # raise CommClosedError("Endpoint is closed -- unable to send message")
        try:
            if serializers is None:
                serializers = ("cuda", "dask", "pickle", "error")
            # msg can also be a list of dicts when sending batched messages
            frames = await to_frames(
                msg,
                serializers=serializers,
                on_error=on_error,
                allow_offload=self.allow_offload,
            )
            nframes = len(frames)
            cuda_frames = tuple(hasattr(f, "__cuda_array_interface__") for f in frames)
            sizes = tuple(nbytes(f) for f in frames)
            # logging.info("{}->{}***, cuda_frames={}, sizes={}".format(MPI.COMM_WORLD.Get_rank(), self.peer_rank, cuda_frames, sizes))
            cuda_send_frames, send_frames = zip(
                *(
                    (is_cuda, each_frame)
                    for is_cuda, each_frame in zip(cuda_frames, frames)
                    if nbytes(each_frame) > 0
                )
            )

            # Send meta data

            # Send close flag and number of frames (_Bool, int64)
            #await self.ep.send(struct.pack("?Q", False, nframes))
            #initialize tag with the assigned tag range
            tag = self.suggested_tag
            msg_ = np.zeros((2), np.int32)
            msg_[0] = 0
            msg_[1] = nframes
            logger.debug("write_1: me=%s, you=%s, tag=%s", MPI.COMM_WORLD.Get_rank(), self.peer_rank, tag)

            await self.mpi_send(msg_, 8, tag)
            tag = tag + 1

            # Send which frames are CUDA (bool) and
            # how large each frame is (uint64)
            msg_ = np.zeros((nframes*2), np.int32)
            for i in range(nframes):
                msg_[i] = cuda_frames[i]
                msg_[i+nframes] = sizes[i]
            await self.mpi_send(msg_, nframes*2*4, tag)
            # logging.info("{}->{}, cuda_frames={}, sizes={}".format(MPI.COMM_WORLD.Get_rank(), self.peer_rank, msg[:nframes], msg[nframes:]))
            tag = tag + 1
            logger.debug("write_2: me=%s, you=%s, tag=%s", MPI.COMM_WORLD.Get_rank(), self.peer_rank, tag)

            # Send frames

            # It is necessary to first synchronize the default stream 
            # before start sending. 

            # non-blocking CUDA streams.
            if any(cuda_send_frames):
                synchronize_stream(0)

            sizes_new = tuple(i for i in sizes if i != 0) # remove all 0's from this list
            read_counter = 0
            # if MPI.COMM_WORLD.Get_rank() == print_rank:
            #     logging.info("{},  {}->{}, msg={}, tag_start={}".format(datetime.datetime.now(), MPI.COMM_WORLD.Get_rank(), self.peer_rank, msg, tag))
            req_list = []
            for each_frame in send_frames:
                s_size = sizes_new[read_counter]
                # if MPI.COMM_WORLD.Get_rank() == print_rank:
                #     logging.info("{},  {}->{}, tag={}, size={}, total={}/{}".format(datetime.datetime.now(), MPI.COMM_WORLD.Get_rank(), self.peer_rank, tag, s_size, read_counter+1, len(send_frames)))
                
                
                r = MPI.COMM_WORLD.Isend([each_frame, s_size, MPI.BYTE], dest=self.peer_rank, tag=tag)
                tag = tag + 1
                read_counter = read_counter + 1
                req_list.append(r)
            
            assert len(req_list) == nframes
            flag = MPI.Request.Testall(req_list)

            while flag is False:
                await asyncio.sleep(0)
                flag = MPI.Request.Testall(req_list)
            
            if MPI.COMM_WORLD.Get_rank() == print_rank:
                logging.info("{}  {}->{}, msg={}".format(datetime.datetime.now(), MPI.COMM_WORLD.Get_rank(), self.peer_rank, msg))
            return sum(sizes_new)
        except (Exception):
            self.abort()
            raise CommClosedError("While writing, the connection was closed")

    @log_errors
    async def read(self, deserializers=("cuda", "dask", "pickle", "error")):
        logger.debug("read() function is called")
        if self.closed():
            #logger.debug("inside self.closed()")
            raise CommClosedError("Endpoint is closed -- unable to read message")

        if deserializers is None:
            deserializers = ("cuda", "dask", "pickle", "error")

        try:
            # Recv meta data

            # Recv close flag and number of frames (_Bool, int64)
            msg = np.zeros((2), np.int32)
            tag = self.suggested_tag
            await self.mpi_recv(msg, 8, tag)
            tag = tag + 1
            shutdown, nframes = msg[0], msg[1]
            # logging.info("{}<-{}, shutdown={}, nframes={}".format(MPI.COMM_WORLD.Get_rank(), self.peer_rank, shutdown, nframes))
            logger.debug("read_1: me=%s, you=%s, tag=%s", MPI.COMM_WORLD.Get_rank(), self.peer_rank, tag)

            if shutdown:  # The writer is closing the connection
                raise CommClosedError("Connection closed by writer")

            # Recv which frames are CUDA (bool) and
            # how large each frame is (uint64)
            header = np.zeros((nframes*2), np.int32)
            await self.mpi_recv(header, nframes*2*4, tag)
            tag = tag + 1
            cuda_frames, sizes = header[:nframes], header[nframes:]
            # logging.info("{}<-{}, cuda_frames={}, sizes={}".format(MPI.COMM_WORLD.Get_rank(), self.peer_rank, cuda_frames, sizes))
        except (Exception, CancelledError):
            self.abort()
            raise CommClosedError("While reading, the connection was closed")
        else:
            # Recv frames
            # frames = [
            #     device_array(each_size) if is_cuda else np.empty((each_size,), dtype="u1").data
            #     for is_cuda, each_size in zip(cuda_frames, sizes)
            # ]
            frames = [
                device_array(each_size) if is_cuda else np.zeros(each_size, 'B')
                for is_cuda, each_size in zip(cuda_frames, sizes)
            ]
            cuda_recv_frames, recv_frames, print_sizes = zip(
                *(
                    (is_cuda, each_frame, size_)
                    for is_cuda, each_frame, size_ in zip(cuda_frames, frames, sizes)
                    if nbytes(each_frame) > 0
                )
            )

            # It is necessary to first populate `frames` with CUDA arrays 
            # and synchronize the default stream before starting receiving 
            # to ensure buffers have been allocated
            if any(cuda_recv_frames):
                synchronize_stream(0)


            sizes_new = tuple(i for i in sizes if i != 0) # remove all 0's from this list

            if len(sizes_new) > 1000:
                logger.debug("len(sizes_new) = %s, len(sizes)=%s", len(sizes_new), len(sizes))

            assert len(sizes_new) < TAG_QUOTA_PER_CONNECTION - 2

            logger.debug("read_3: me=%s, you=%s, tag=%s, sizes=%s, sizes_new=%s, rf_len=%s, crf_len=%s", \
                                        MPI.COMM_WORLD.Get_rank(), self.peer_rank, tag, sizes, sizes_new, len(recv_frames), len(cuda_recv_frames))

            read_counter = 0
            req_list = []
            for each_frame in recv_frames:
                s_size = sizes_new[read_counter]
                # if MPI.COMM_WORLD.Get_rank() == print_rank:
                #     logging.info("{}  {}<-{}, tag={}, size={}".format(datetime.datetime.now(), MPI.COMM_WORLD.Get_rank(), self.peer_rank, tag, print_sizes[read_counter]))
                
                r = MPI.COMM_WORLD.Irecv([each_frame, s_size, MPI.BYTE], source=self.peer_rank, tag=tag)
                tag = tag + 1
                read_counter = read_counter + 1
                req_list.append(r)
                
            assert len(req_list) == nframes
            flag = MPI.Request.Testall(req_list)

            while flag is False:
                await asyncio.sleep(0)
                flag = MPI.Request.Testall(req_list)

            msg = await from_frames(
                frames,
                deserialize=self.deserialize,
                deserializers=deserializers,
                allow_offload=self.allow_offload,
            )
            if MPI.COMM_WORLD.Get_rank() == print_rank:
                logging.info("{}  {}<-{}, msg={}".format(datetime.datetime.now(), MPI.COMM_WORLD.Get_rank(), self.peer_rank, msg))
            return msg


    # a utility function for sending messages larger than CHUNK_SIZE data
    async def mpi_send_large(self, buf, size, _tag):

        me = MPI.COMM_WORLD.Get_rank()
        you = self.peer_rank
         
        logger.debug("mpi_send_large: host=%s, me=%s, you=%s, tag=%s, size=%s, type(buf)=%s", \
                                               socket.gethostname(), me, you, _tag, size, type(buf))

        blk_size = CHUNK_SIZE
        num_of_blks   = int(size / blk_size)
        last_blk_size =     size % blk_size

        logger.debug("mpi_send_large: blk_size=%s, num_of_blks=%s, last_blk_size=%s", \
                                                          blk_size, num_of_blks, last_blk_size)

        num_of_reqs = num_of_blks

        if last_blk_size is not 0:
            num_of_reqs = num_of_reqs + 1

        reqs = []
        for i in range(num_of_blks):
            s_idx = (i)   * blk_size
            e_idx = (i+1) * blk_size

            if type(buf) == rmm._lib.device_buffer.DeviceBuffer:
                # need this if because rmm.DeviceBuffer is not subscriptable
                shadow_buf = rmm.DeviceBuffer(ptr=(buf.ptr+s_idx), \
                                                           size=blk_size)
                r = MPI.COMM_WORLD.Isend([shadow_buf, blk_size, MPI.BYTE], dest=you, tag=_tag)
            else:
                r = MPI.COMM_WORLD.Isend([buf[s_idx:e_idx], blk_size, MPI.BYTE], dest=you, tag=_tag)
            
            
            _tag = _tag + 1
            reqs.append(r)

        if last_blk_size is not 0:
            s_idx = num_of_blks*blk_size
            e_idx = s_idx+last_blk_size

            if type(buf) == rmm._lib.device_buffer.DeviceBuffer:
                # need this if because rmm.DeviceBuffer is not subscriptable
                shadow_buf = rmm.DeviceBuffer(ptr=(buf.ptr+s_idx), \
                                                           size=last_blk_size)
                r = MPI.COMM_WORLD.Isend([shadow_buf, last_blk_size, MPI.BYTE], dest=you, tag=_tag)
            else:
                r = MPI.COMM_WORLD.Isend([buf[s_idx:e_idx], last_blk_size, MPI.BYTE], \
                                                           dest=you, tag=_tag)
            
            
            _tag = _tag + 1
            reqs.append(r)

        assert len(reqs) == num_of_reqs

        flag = MPI.Request.Testall(reqs)

        while flag is False:
            await asyncio.sleep(0)
            flag = MPI.Request.Testall(reqs)

    # a utility function for sending messages through MPI
    async def mpi_send(self, buf, size, _tag):

        me = MPI.COMM_WORLD.Get_rank()
        you = self.peer_rank
        rank = MPI.COMM_WORLD.Get_rank()

        if me == 0:
            logger.debug("mpi_send: host=%s, suggested_tag=%s, peer_rank=%s, me=%s, you=%s, rank=%s, tag=%s, size=%s, type(buf)=%s", \
                                             socket.gethostname(), self.suggested_tag, self.peer_rank, me, you, rank, _tag, size, type(buf))

        
        # logger.info("send size:{}".format(size))
        if size > CHUNK_SIZE:
            # if message size is larger than CHUNK_SIZE, split it into chunks and communicate
            logger.debug("mpi_send: host=%s, comm=%s, me=%s, you=%s, rank=%s, tag=%s, size=%s, type(buf)=%s", \
                                         socket.gethostname(), MPI.COMM_WORLD, me, you, rank, _tag, size, type(buf))
            #if type(buf) == cupy.core.core.ndarray:
            #h_buf = numpy.empty((size,), dtype="u1")
            #h_buf = numpy.frombuffer(buf.tobytes(), dtype="u1")
            await self.mpi_send_large(buf, size, _tag)
            return
            
        req = MPI.COMM_WORLD.Isend([buf, size, MPI.BYTE], dest=you, tag=_tag)

        status = req.Test()

        try:
            while status is False:
                await asyncio.sleep(0)
                status = req.Test()
        
        except isinstance(asyncio.exceptions.CancelledError):
            logger.info("send status check fail!!!!!!!!!!!!!!!!!!!!")
            sys.exit()

    # a utility function for receiving messages larger than CHUNK_SIZE data
    async def mpi_recv_large(self, buf, size, _tag):

        me = MPI.COMM_WORLD.Get_rank()
        you = self.peer_rank

        logger.debug("mpi_recv_large: host=%s, me=%s, you=%s, tag=%s, size=%s, type(buf)=%s", \
                                                socket.gethostname(), me, you, _tag, size, type(buf))

        blk_size = CHUNK_SIZE
        num_of_blks   = int(size / blk_size)
        last_blk_size =     size % blk_size

        logger.debug("mpi_recv_large: blk_size=%s, num_of_blks=%s, last_blk_size=%s", \
                                                        blk_size, num_of_blks, last_blk_size)

        num_of_reqs = num_of_blks

        if last_blk_size is not 0:
            num_of_reqs = num_of_reqs + 1

        reqs = []        

        for i in range(num_of_blks):
            s_idx = (i)   * blk_size
            e_idx = (i+1) * blk_size

            if type(buf) == rmm._lib.device_buffer.DeviceBuffer:
                # need this if because rmm.DeviceBuffer is not subscriptable
                shadow_buf = rmm.DeviceBuffer(ptr=(buf.ptr+s_idx), \
                                                           size=blk_size)
                r = MPI.COMM_WORLD.Irecv([shadow_buf, blk_size, MPI.BYTE], source=you, tag=_tag)
            else:
                r = MPI.COMM_WORLD.Irecv([buf[s_idx:e_idx], blk_size, MPI.BYTE], source=you, \
                                                                   tag=_tag)
            
            _tag = _tag + 1
            reqs.append(r)

        if last_blk_size is not 0:
            s_idx = num_of_blks*blk_size
            e_idx = s_idx+last_blk_size

            if type(buf) == rmm._lib.device_buffer.DeviceBuffer:
                # need this if because rmm.DeviceBuffer is not subscriptable
                shadow_buf = rmm.DeviceBuffer(ptr=(buf.ptr+s_idx), \
                                                           size=last_blk_size)
                r = MPI.COMM_WORLD.Irecv([shadow_buf, last_blk_size, MPI.BYTE], source=you, tag=_tag)
            else:
                r = MPI.COMM_WORLD.Irecv([buf[s_idx:e_idx], last_blk_size, MPI.BYTE], \
                                                    source=you, tag=_tag)
            
            _tag = _tag + 1
            reqs.append(r)

        assert len(reqs) == num_of_reqs

        flag = MPI.Request.Testall(reqs)

        while flag is False:
            await asyncio.sleep(0)
            flag = MPI.Request.Testall(reqs)

    # a utility function for receiving messages through MPI
    async def mpi_recv(self, buf, size, _tag):

        import numpy as np
        rank = MPI.COMM_WORLD.Get_rank()
        me = rank
        you = self.peer_rank

        # logger.info("receive size:".format(size))
        if size > CHUNK_SIZE:
            logger.debug("mpi_recv: me=%s, you=%s, tag=%s, size=%s, type(buf)=%s", \
                                                        me, you, _tag, size, type(buf))
            await self.mpi_recv_large(buf, size, _tag)
            return


        req = MPI.COMM_WORLD.Irecv([buf, size, MPI.BYTE], source=you, tag=_tag)
        status = req.Test()
        with suppress(asyncio.CancelledError, TimeoutError):
            while status is False:

                await asyncio.sleep(0)
                status = req.Test()


        if you == 0 and isinstance(buf, np.ndarray):
            logger.debug("mpi_recv: me=%s, you=%s, tag=%s, size=%s, type(buf)=%s, buf[:10]=%s", \
                                                        me, you, _tag, size, type(buf), bytearray(buf[:10]))

    async def close(self):
        self._closed = True
        tag = self.suggested_tag
        msg_ = np.zeros((2), np.int32)
        msg_[0] = 1
        msg_[1] = 0

        await self.mpi_send(msg_, 8, tag)

    def abort(self):
        if self._closed is False:
            self._closed = True

    def closed(self):
        return self._closed

# tag set for all workers
# initial tag_table:
# scheduler, tag=100
# client, tag=100
# worker#1, tag=100
# worker#2, tag=100

# tags are updated on the receiver's side,
# rank 0: scheduler; 
# rank 1: client; 
# rank 2-: workers

# 1. Register worker. Scheduler first communicate with workers, 
#       scheduler: listener
#       workers:   connector
# 2. Start worker compute stream, 
#       scheduler: listener
#       workers:   connector
# 3. after workers been created, scheduler communicates with client,
#       scheduler: listener
#       client:    connector
# 4. workers communicate with each other,
#       worker#1:  listener
#       worker#2:  connector

# Start computing ......

# 5. scheduler communicates with workers,
#       scheduler: connector
#       workers:   listener
# 6. scheduler communicates with client,
#       scheduler: listener
#       client:    connector
# 6. scheduler communicates with client,
#       scheduler: listener
#       client:    connector

# worker - INFO - Run out-of-band function 'lambda'

# 7. remove client,
#       scheduler: listener
#       client:    connector

# 8. scheduler closing.

class MPI4DaskConnector(Connector):
    prefix = "mpi://"
    comm_class = MPI4Dask
    encrypted = False

    async def connect(self, address: str, \
                      deserialize: bool = True, \
                      **connection_args: Any
        ) -> MPI4Dask:

        rand_num = 0.001 * random.randint(0, 1000)
        await asyncio.sleep(rand_num)
        logger.debug("%s: connecting to address=%s with delay=%s", \
                             socket.gethostname(), address, rand_num)

        ip, port = parse_host_port(address)
        init_once()
        reader, writer = await asyncio.open_connection(ip, port)

        peer_addr = writer.get_extra_info('peername')
        local_addr = writer.get_extra_info('sockname')
        local_rank = MPI.COMM_WORLD.Get_rank()

        logger.debug("%s: connect: local_addr=%s, peer_addr=%s", \
                        socket.gethostname(), local_addr[0], peer_addr[0])

        assert reader.at_eof() == False 
        data = await reader.read(4) # worker<-scheduler: scheduler's rank
        logger.debug("data: %s", data)
        peer_rank = int(data.decode())

        data = str(local_rank).encode()
        writer.write(data) # worker->scheduler: worker's rank
        await writer.drain()

        suggested_tag = tag_table.get(peer_rank)
        
        tag_table.update({peer_rank : + (suggested_tag+TAG_QUOTA_PER_CONNECTION)}) # worker set specific tag upper bound for mpi communication with the scheduler
        

        return self.comm_class(
            peer_rank,
            suggested_tag,
            local_addr = self.prefix + local_addr[0] + ":" + \
                                                str(local_addr[1]),
            peer_addr = self.prefix + peer_addr[0] + ":" + \
                                                str(peer_addr[1]),
            p_addr = peer_addr[0],
            deserialize=deserialize,
        )


class MPI4DaskListener(Listener):
    prefix = MPI4DaskConnector.prefix
    comm_class = MPI4DaskConnector.comm_class
    encrypted = MPI4DaskConnector.encrypted

    def __init__(
        self,
        address: str,
        comm_handler: Callable[[MPI], Awaitable[None]] | None = None,
        deserialize: bool = False,
        allow_offload: bool = True,
        **connection_args: Any,
    ):
        global LOWER_PORT_RANGE, UPPER_PORT_RANGE

        if not address.startswith("mpi"):
            address = "mpi://" + address

        logger.debug("%s: MPI4DaskListener.__init__ %s", \
                                              socket.gethostname(), address)
        self.ip, self._input_port = parse_host_port(address, default_port=0)
        # choose a random port between LOWER_PORT_RANGE and UPPER_PORT_RANGE
        self._input_port = random.randint(LOWER_PORT_RANGE, UPPER_PORT_RANGE)
        self.comm_handler = comm_handler
        self.deserialize = deserialize
        self.allow_offload = allow_offload
        self.mpi_server = None
        self.connection_args = connection_args

    @property
    def port(self):
        return self._input_port

    @property
    def address(self):
        return "mpi://" + self.ip + ":" + str(self.port)

    async def start(self):
        async def serve_forever(reader, writer):

            peer_addr = writer.get_extra_info('peername')
            local_addr = writer.get_extra_info('sockname')
            logger.debug("%s: listen(): local=%s, peer=%s", \
                        socket.gethostname(), local_addr[0], peer_addr[0])

            local_rank = MPI.COMM_WORLD.Get_rank()

            data = str(local_rank).encode()
            writer.write(data) # scheduler->worker: scheduler's rank
            await writer.drain()
            logger.debug("listen(): wrote data")

            assert reader.at_eof() == False 
            data = await reader.read(4)
            peer_rank = int(data.decode()) # worker->scheduler: client's rank
            logger.debug("listen(): read data")

            suggested_tag = tag_table.get(peer_rank)
            
            tag_table.update({peer_rank : (suggested_tag + TAG_QUOTA_PER_CONNECTION)}) # scheduler set specific tag upper bound for mpi communication with this worker
            logger.debug("%s: listen(): data=%s, rank=%s, peer_rank=%s, suggested_tag=%s", \
                                                      socket.gethostname(), data, local_rank, peer_rank, suggested_tag)
                                                      
            mpi = MPI4Dask(
                peer_rank,
                suggested_tag,
                local_addr = self.prefix + local_addr[0] + ":" + \
                                                   str(local_addr[1]),
                peer_addr = self.prefix + peer_addr[0] + ":" + \
                                                   str(peer_addr[1]),
                p_addr = peer_addr[0],
                deserialize=self.deserialize,
            )
            mpi.allow_offload = self.allow_offload
            try:
                await self.on_connection(mpi)
            except CommClosedError:
                logger.debug("Connection closed before handshake completed")
                return
            if self.comm_handler:
                logger.debug("%s: calling comm_handler(mpi)", socket.gethostname())
                await self.comm_handler(mpi)

        init_once()
        
        coro = asyncio.start_server(serve_forever, \
                                    None, \
                                    port=self._input_port, \
                                    backlog=1024)
        task = asyncio.create_task(coro)
        self.mpi_server = await task
        

    def stop(self):
        self.mpi_server = None

    def get_host_port(self):
        # TODO: TCP raises if this hasn't started yet.
        return self.ip, self.port

    @property
    def listen_address(self):
        return self.prefix + unparse_host_port(*self.get_host_port())

    @property
    def contact_address(self):
        host, port = self.get_host_port()
        host = ensure_concrete_host(host)  # TODO: ensure_concrete_host
        return self.prefix + unparse_host_port(host, port)

    @property
    def bound_address(self):
        # TODO: Does this become part of the base API? Kinda hazy, since
        # we exclude in for inproc.
        return self.get_host_port()

class MPI4DaskBackend(Backend):
    # I / O

    def get_connector(self):
        return MPI4DaskConnector()

    def get_listener(self, loc, handle_comm, deserialize, **connection_args):
        return MPI4DaskListener(loc, handle_comm, deserialize, **connection_args)

    # Address handling
    # This duplicates BaseTCPBackend

    def get_address_host(self, loc):
        return parse_host_port(loc)[0]

    def get_address_host_port(self, loc):
        return parse_host_port(loc)

    def resolve_address(self, loc):
        host, port = parse_host_port(loc)
        return unparse_host_port(ensure_ip(host), port)

    def get_local_address_for(self, loc):
        host, port = parse_host_port(loc)
        host = ensure_ip(host)
        if ":" in host:
            local_host = get_ipv6(host)
        else:
            local_host = get_ip(host)
        return unparse_host_port(local_host, None)

backends["mpi"] = MPI4DaskBackend()
