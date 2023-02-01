"""
:ref:`UCR`_ based communications for distributed.

See :ref:`communications` for more.

.. _UCR: https://github.com/openucx/ucx
"""
import logging
import struct
import weakref
import asyncio
import threading
import concurrent.futures

import socket

import dask
import random
import sys

from .addressing import parse_host_port, unparse_host_port
from .core import Comm, Connector, Listener, CommClosedError
from .registry import Backend, backends
from .utils import ensure_concrete_host, to_frames, from_frames
from ..utils import (
    ensure_ip,
    get_ip,
    get_ipv6,
    nbytes,
    log_errors,
    CancelledError,
    parse_bytes,
)

logger = logging.getLogger(__name__)
DEBUG = 0
write_counter = 0
read_counter = 0

# In order to avoid double init when forking/spawning new processes (multiprocess),
# we make sure only to import and initialize UCR once at first use. This is also
# required to ensure Dask configuration gets propagated to UCR, which needs
# variables to be set before being imported.
ucp = None
ucr = None
context = None
#host_array = None TMP
device_array = None

c_socket = None
#ucr_port = 12345
CB_FUNC_ID = 101
UCR_EP_STATUS_CONNECTED = 3
client_num = 100
seed_value = 0

def synchronize_stream(stream=0):
    import numba.cuda

    ctx = numba.cuda.current_context()
    cu_stream = numba.cuda.driver.drvapi.cu_stream(stream)
    stream = numba.cuda.driver.Stream(ctx, cu_stream, None)
    stream.synchronize()


def init_once():
    if DEBUG == 1:
        print("init_once")
    global ucp, ucr, context, device_array, seed_value #TMP host_array
    if ucp is not None:
        return

    import ucp as _ucp
    import ucrpy as _ucr

    ucp = _ucp
    ucr = _ucr

    # remove/process dask.ucx flags for valid ucx options
    ucr_config = _scrub_ucr_config()

    ucp.init(options=ucr_config, env_takes_precedence=True)
    
    seed_value = random.randrange(sys.maxsize)
    random.seed(seed_value)

    ucr.open()
    context = ucr.init(0)
    #ucr.finalize(context)

    # Find the function, `host_array()`, to use when allocating new host arrays
    # REMOVED TMP
    #try:
    #    import numpy
    #
    #    host_array = lambda n: numpy.empty((n,), dtype="u1")
    #except ImportError:
    #    host_array = lambda n: bytearray(n)

    # Find the function, `cuda_array()`, to use when allocating new CUDA arrays
    try:
        import rmm

        if hasattr(rmm, "DeviceBuffer"):
            device_array = lambda n: rmm.DeviceBuffer(size=n)
        else:  # pre-0.11.0
            import numba.cuda

            def rmm_device_array(n):
                a = rmm.device_array(n, dtype="u1")
                weakref.finalize(a, numba.cuda.current_context)
                return a

            device_array = rmm_device_array
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

    pool_size_str = dask.config.get("rmm.pool-size")
    if pool_size_str is not None:
        pool_size = parse_bytes(pool_size_str)
        rmm.reinitialize(
            pool_allocator=True, managed_memory=False, initial_pool_size=pool_size
        )


class UCR(Comm):
    """Comm object using UCP.

    Parameters
    ----------
    ep : ucp.Endpoint
        The UCP endpoint.
    address : str
        The address, prefixed with `ucr://` to use.
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

    1. Read the frame describing number of frames
    2. Read the frame describing whether each data frame is gpu-bound
    3. Read the frame describing whether each data frame is sized
    4. Read all the data frames.
    """

    #def __init__(self, ep, local_addr: str,\
    #                       peer_addr: str, 
    #                       deserialize=True):
    def __init__(self, ep, endpoint, my_ep_info, rem_ep_info, cb_func_id,\
                           local_addr: str,\
                           peer_addr: str, 
                           deserialize=True):
        if DEBUG == 1: 
            print("ucr:__init__() function")
        Comm.__init__(self)
        self._ep = ep
        self._endpoint = endpoint
        self._my_ep_info = my_ep_info
        self._rem_ep_info = rem_ep_info
        self.cb_func_id = cb_func_id
        if local_addr:
            assert local_addr.startswith("ucr")
        assert peer_addr.startswith("ucr")
        self._local_addr = local_addr
        self._peer_addr = peer_addr
        self.deserialize = deserialize
        self.comm_flag = None
        logger.debug("UCR.__init__ %s", self)

    @property
    def local_address(self) -> str:
        return self._local_addr

    @property
    def peer_address(self) -> str:
        return self._peer_addr

    async def write(
        self,
        msg: dict,
        serializers=("cuda", "dask", "pickle", "error"),
        on_error: str = "message",
    ):    
        #if DEBUG == 1: 
        #    print("ucr: UCR.write() begins ..., id = ", self.cb_func_id, \
        #                                     "write_counter =", local_counter)
        with log_errors():
            if self.closed():
                raise CommClosedError("Endpoint is closed -- unable to send message")
            try:
                if serializers is None:
                    serializers = ("cuda", "dask", "pickle", "error")
                # msg can also be a list of dicts when sending batched messages
                frames = await to_frames(
                    msg, serializers=serializers, on_error=on_error
                )
                nframes = len(frames)
                cuda_frames = tuple(
                    hasattr(f, "__cuda_array_interface__") for f in frames
                )
                sizes = tuple(nbytes(f) for f in frames)
                send_frames = [
                    each_frame
                    for each_frame, each_size in zip(frames, sizes)
                    if each_size
                ]

                # Send meta data

                # Send # of frames (uint64)
                #if self.ep is not None: 
                #    await self.ep.send(struct.pack("Q", nframes))
                #else: 
                sbuf1 = struct.pack("Q", nframes)

                if DEBUG == 1:
                    print("ucr: write():1, id=", self.cb_func_id, \
                                  "len=", len(sbuf1), \
                                  "addr=", hex(id(sbuf1)))
                await ucr.msg_send(self.endpoint, sbuf1, \
                                                self.cb_func_id)


                # Send which frames are CUDA (bool) and
                # how large each frame is (uint64)

                #if self.ep is not None: 
                #    await self.ep.send(
                #        struct.pack(nframes * "?" + nframes * "Q", *cuda_frames, *sizes)
                #    )
                #else: 
                sbuf2 = struct.pack(nframes * "?" + nframes * "Q", *cuda_frames, *sizes)

                if DEBUG == 1:
                    print("ucr: write():2, id=", self.cb_func_id, \
                          "len=", len(sbuf2), \
                          "addr=", hex(id(sbuf2))) 
                await ucr.msg_send( self.endpoint, sbuf2, self.cb_func_id )

                # Send frame

                # It is necessary to first synchronize the default stream before start sending
                # We synchronize the default stream because UCR is not stream-ordered and
                #  syncing the default stream will wait for other non-blocking CUDA streams.
                # Note this is only sufficient if the memory being sent is not currently in use on
                # non-blocking CUDA streams.
                if any(cuda_frames):
                    synchronize_stream(0)

                for each_frame in send_frames:
                    #if self.ep is not None:
                    #    await self.ep.send(each_frame)
                    #else: 
                    if DEBUG == 1:
                        print("ucr: write():3, id=", self.cb_func_id, \
                                               "len=", len(each_frame), \
                                               "addr=", hex(id((each_frame))))
                        print(self.cb_func_id,"ucr: write() 4:")
                    await ucr.msg_send(self.endpoint,\
                                       each_frame, self.cb_func_id)
                                       
                            
                if DEBUG == 1: 
                    print("ucr: UCR.write() ends ... ", self.cb_func_id, \
                                             "write_counter =", write_counter)
                return sum(sizes)
            except (ucp.exceptions.UCXBaseException):
                print("write() exception")
                self.abort()
                raise CommClosedError("While writing, the connection was closed")

    async def read(self, deserializers=("cuda", "dask", "pickle", "error")):
        loop = asyncio.get_running_loop()

        if DEBUG == 1:
            print("")
            print("ucr: read() begins ..., id=", self.cb_func_id, \
                                           "read_counter = ", local_counter)
        with log_errors():
            if self.closed():
                raise CommClosedError("Endpoint is closed -- unable to read message")

            if deserializers is None:
                deserializers = ("cuda", "dask", "pickle", "error")

            try:
                import numpy

                host_array = lambda n: numpy.empty((n,), dtype="u1")
            except ImportError:
                host_array = lambda n: bytearray(n)

            try:
                # Recv meta data

                # Recv # of frames (uint64)
                nframes_fmt = "Q"
                nframes = host_array(struct.calcsize(nframes_fmt))

                #print(self.cb_func_id,"UCR.msg_recv()0", self.cb_func_id)
                #if self.ep is not None:
                #    print("UCR.msg_recv()0.5")
                #    await self.ep.recv(nframes)
                #else:
                #if DEBUG == 1:
                #    print("ucr: msg_recv():1, id= ", self.cb_func_id, \
                #                              "len=", len(nframes), \
                #                              "addr=", hex(id(nframes)))

                await ucr.msg_recv(self.endpoint, nframes,\
                                   context, self.cb_func_id)
                #print(self.cb_func_id,"UCR.msg_recv() 1.5")
                #print(self.cb_func_id,"UCR.msg_recv() 1.75", self, threading.get_ident())
                #await loop.run_in_executor( 
                #    None, 
                #    ucr.msg_recv(self.endpoint, nframes, context)
                #)

                #with concurrent.futures.ThreadPoolExecutor() as pool:
                #    result = await loop.run_in_executor(
                #        pool, 
                #        ucr.msg_recv(self.endpoint, nframes, context)
                #    )

                (nframes,) = struct.unpack(nframes_fmt, nframes)

                # Recv which frames are CUDA (bool) and
                # how large each frame is (uint64)
                header_fmt = nframes * "?" + nframes * "Q"
                header = host_array(struct.calcsize(header_fmt))

                #if self.ep is not None:
                #    await self.ep.recv(header)
                #else:
                if DEBUG == 1:
                    print("ucr: msg_recv():2, id= ", self.cb_func_id, \
                                             "len=", len(header), \
                                             "addr=", hex(id(header)))
                await ucr.msg_recv(self.endpoint, header,\
                                   context, self.cb_func_id)

                header = struct.unpack(header_fmt, header)
                cuda_frames, sizes = header[:nframes], header[nframes:]
            except (ucp.exceptions.UCXBaseException, CancelledError):
                print("exception 1")
                self.abort()
                raise CommClosedError("While reading, the connection was closed")
            else:
                # Recv frames
                frames = [
                    device_array(each_size) if is_cuda else host_array(each_size)
                    for is_cuda, each_size in zip(cuda_frames, sizes)
                ]
                recv_frames = [
                    each_frame for each_frame in frames if len(each_frame) > 0
                ]

                # It is necessary to first populate `frames` with CUDA arrays and synchronize
                # the default stream before starting receiving to ensure buffers have been allocated
                if any(cuda_frames):
                    synchronize_stream(0)

                for each_frame in recv_frames:
                    #if self.ep is not None: 
                    #    await self.ep.recv(each_frame)
                    #else:
                    if DEBUG == 1:
                        print("ucr: msg_recv():3, id= ", self.cb_func_id, \
                                                 "len=", len(each_frame), \
                                                "addr=", hex(id(each_frame)))
                    await ucr.msg_recv(self.endpoint, each_frame,\
                                           context, self.cb_func_id)

                msg = await from_frames(
                    frames, 
                    deserialize=self.deserialize, 
                    deserializers=deserializers
                )
                if DEBUG == 1:
                    print("ucr: read() ends ..., id=", self.cb_func_id, \
                                           "read_counter = ", local_counter)
                return msg

    async def close(self):
        if self._ep is not None:
            await self._ep.close()
            self._ep = None

        if self._endpoint is not None:
            ucr.ep_destroy(self._endpoint)
            ucr.close(self._my_ep_info, self._rem_ep_info)
            self._endpoint = None
            self._my_ep_info = None
            self._rem_ep_info = None 

    def abort(self):
        if self._ep is not None:
            self._ep.abort()
            self._ep = None

        if self._endpoint is not None:
            ucr.ep_destroy(self._endpoint)
            ucr.close(self._my_ep_info, self._rem_ep_info)
            self._endpoint = None
            self._my_ep_info = None
            self._rem_ep_info = None 

    @property
    def endpoint(self):
        if self._endpoint is not None:
            return self._endpoint
        else:
            raise CommClosedError("UCR Endpoint is closed")

    @property
    def ep(self):
        if self._ep is not None:
            return self._ep
        else:
            raise CommClosedError("UCR Endpoint is closed")

    def closed(self):
        return self._endpoint is None
        #return self._ep is None


class UCRConnector(Connector):
    prefix = "ucr://"
    comm_class = UCR
    encrypted = False
    connect_in_progress = False;

    async def connect(self, address: str, \
                      deserialize=True, **connection_args) -> UCR:

        if self.connect_in_progress is False:
            self.connect_in_progress = True
        else:
            print("ucr: connect(): connect in progress. return ... ")
            return

        local_host_name = socket.gethostname()
        #global client_num
        client_num = random.randint(1, 255)
        logger.debug("UCRConnector.connect: %s", address)
        ip, port = parse_host_port(address)

        if DEBUG == 1:
            print("")
            print("ucr: connect(): host ", local_host_name, \
                      " is connecting to ", ip, "port", port, \
                                " suggested_ep_id ", client_num)
            #all_tasks = asyncio.all_tasks()
            #print(all_tasks)
            #print("all_tasks = ", len(all_tasks))

        init_once()

        #### UCR connect (start) #####
        endpoint, my_ep_info = ucr.ep_create(context)
        ucr.ep_get_info(endpoint, my_ep_info)
        
        if DEBUG == 1:
            print("ucr: connect(): connecting to ip=", ip, "port=", port)

        reader, writer = await ucr.myconnect(ip, port)
        rem_ep_info, cb_id = await ucr.exchange_ep_info(True, ip, my_ep_info, \
                                         reader, writer, client_num)
        client_num = cb_id

        writer.close()

        if DEBUG == 1:
            print("ucr: connect(): agreed cb_func_id ", cb_id)
        
        ucr.reg_hdr_hndlr(context, cb_id)
        
        if DEBUG == 1:
            print("ucr: connect(): calling ep_start_connect()")

        ret_val = ucr.ep_start_connect(endpoint, rem_ep_info)

        if DEBUG == 1:
            print("ucr: connect(): called ep_start_connect()")
        
        while True:
            ep_status = ucr.ep_get_status(endpoint)
        
            if ep_status.value == UCR_EP_STATUS_CONNECTED:
                break

        if DEBUG == 1:
            print("ucr: connect(): connected ... ")


        self.connect_in_progress = False
        return self.comm_class(
            None, 
            endpoint, 
            my_ep_info, 
            rem_ep_info,
            cb_id,
            local_addr=self.prefix + local_host_name,
            peer_addr=self.prefix + address,
            deserialize=deserialize,
        )

class UCRListener(Listener):
    prefix = UCRConnector.prefix
    comm_class = UCRConnector.comm_class
    encrypted = UCRConnector.encrypted

    def __init__(
        self, address: str, comm_handler: None, \
        deserialize=False, **connection_args
    ):
        #print("UCRListener.__init__()")
        if not address.startswith("ucr"):
            address = "ucr://" + address
        self.ip, self._input_port = parse_host_port(address, default_port=0)
        self.comm_handler = comm_handler
        self.deserialize = deserialize
        self._ep = None  # type: ucp.Endpoint
        self.ucp_server = None
        self.connection_args = connection_args

    @property
    def port(self):
        return self.ucp_server.sockets[0].getsockname()[1]

    @property
    def address(self):
        return "ucr://" + self.ip + ":" + str(self.port)

    async def start(self):

        #print("UCRListener.start()")

        async def serve_forever(client_ep):
            if DEBUG == 1:
                print("")
                print("ucr: serve_forever() ")
          
            #ucr_obj = UCR(
            #    client_ep,
            #    local_addr=self.address,
            #    peer_addr=self.address,
            #    deserialize=self.deserialize,
            #)
            #print("ucr object ", ucr)
            if self.comm_handler:
                await self.comm_handler(ucr_obj)

        #### UCR listener (start) #####
        async def ucr_serve_forever(reader, writer):
            #global client_num
            #client_num += 1
            client_num = random.randint(1, 255)
            if DEBUG == 1:
                print("")
                print("ucr: serve_forever(): calling exchange_ep_info, suggested_id", client_num)
                #all_tasks = asyncio.all_tasks()
                #print(all_tasks)
                #print("all_tasks = ", len(all_tasks))

            endpoint, my_ep_info = ucr.ep_create(context)
            ucr.ep_get_info(endpoint, my_ep_info)
            rem_ep_info, cb_id = \
                await ucr.exchange_ep_info( False, None, my_ep_info, \
                                             reader, writer, client_num)
            client_num = cb_id
             
            writer.close()
                          
            if DEBUG == 1:
                print("ucr: serve_forever(): agreed cb_func_id ", cb_id)

            ucr.reg_hdr_hndlr(context, cb_id)
            ret_val = ucr.ep_start_connect(endpoint, rem_ep_info)

            while True:
                ep_status = ucr.ep_get_status(endpoint)
            
                if ep_status.value == UCR_EP_STATUS_CONNECTED:
                    break

            if DEBUG == 1:
                print("ucr: serve_forever(): client has connected")
            
            ucr_obj = UCR(
                None,
                endpoint,
                my_ep_info, 
                rem_ep_info,
                cb_id,
                local_addr=self.address,
                peer_addr=self.address,
                deserialize=self.deserialize,
            )

            if DEBUG == 1:
                print("ucr: serve_forever(): self.comm_handler")

            if self.comm_handler:
                await self.comm_handler(ucr_obj)
                if DEBUG == 1:
                    print("ucr: called the co-routine")
            else: 
                if DEBUG == 1:
                    print("ucr: comm_handler is null")

            if DEBUG == 1:
                print("ucr: last line of ucr_serve_forever")
        
        init_once()
        self.ucp_server = await ucr.create_listener(ucr_serve_forever,\
                                                     port=self._input_port)

    def stop(self):
        self.ucp_server = None
        ucrpy.finalize(context)
        if server_socket != None:
            server_socket.close()
            server_socket = None

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


class UCRBackend(Backend):
    # I / O

    def get_connector(self):
        if DEBUG == 1:
            print("ucr: get_connector()")
        return UCRConnector()

    def get_listener(self, loc, handle_comm, deserialize, **connection_args):
        if DEBUG == 1:
            print("ucr: get_listener()")
        return UCRListener(loc, handle_comm, deserialize, **connection_args)

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

backends["ucr"] = UCRBackend()

def _scrub_ucr_config():
    """Function to scrub dask config options for valid UCR config options"""

    # configuration of UCR can happen in two ways:
    # 1) high level on/off flags which correspond to UCR configuration
    # 2) explicity defined UCR configuration flags

    # import does not initialize ucp -- this will occur outside this function
    from ucp import get_config

    options = {}

    # if any of the high level flags are set, as long as they are not Null/None,
    # we assume we should configure basic TLS settings for UCR, otherwise we
    # leave UCR to its default configuration
    if any(
        [
            dask.config.get("ucx.tcp"),
            dask.config.get("ucx.nvlink"),
            dask.config.get("ucx.infiniband"),
        ]
    ):
        if dask.config.get("ucx.rdmacm"):
            tls = "tcp,rdmacm"
            tls_priority = "rdmacm"
        else:
            tls = "tcp,sockcm"
            tls_priority = "sockcm"

        # CUDA COPY can optionally be used with ucx -- we rely on the user
        # to define when messages will include CUDA objects.  Note:
        # defining only the Infiniband flag will not enable cuda_copy
        if any([dask.config.get("ucx.nvlink"), dask.config.get("ucx.cuda_copy")]):
            tls = tls + ",cuda_copy"

        if dask.config.get("ucx.infiniband"):
            tls = "rc," + tls
        if dask.config.get("ucx.nvlink"):
            tls = tls + ",cuda_ipc"

        options = {"TLS": tls, "SOCKADDR_TLS_PRIORITY": tls_priority}

        net_devices = dask.config.get("ucx.net-devices")
        if net_devices is not None and net_devices != "":
            options["NET_DEVICES"] = net_devices

    # ANY UCR options defined in config will overwrite high level dask.ucx flags
    valid_ucx_keys = list(get_config().keys())
    for k, v in dask.config.get("ucx").items():
        if k in valid_ucx_keys:
            options[k] = v
        else:
            logger.debug(
                "Key: %s with value: %s not a valid UCR configuration option" % (k, v)
            )

    return options
