import logging
import copy
import numpy as np


logger = logging.getLogger("fv3util")


class ConcurrencyError(Exception):
    """Exception to denote that a rank cannot proceed because it is waiting on a
    call from another rank."""

    pass


class AsyncResult:
    def __init__(self, result):
        self._result = result

    def wait(self):
        return self._result()


class DummyComm:
    def __init__(self, rank, total_ranks, buffer_dict):
        self.rank = rank
        self.total_ranks = total_ranks
        self._buffer = buffer_dict
        self._i_buffer = {}
        self._split_comms = {}
        self._split_buffers = {}

    def __repr__(self):
        return f"DummyComm(rank={self.rank}, total_ranks={self.total_ranks})"

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.total_ranks

    def _get_buffer(self, buffer_type, in_value):
        i_buffer = self._i_buffer.get(buffer_type, 0)
        self._i_buffer[buffer_type] = i_buffer + 1
        if buffer_type not in self._buffer:
            self._buffer[buffer_type] = []
        if self.rank == 0:
            self._buffer[buffer_type].append(in_value)
        return self._buffer[buffer_type][i_buffer]

    def _get_send_recv(self, from_rank):
        key = (from_rank, self.rank)
        if "send_recv" not in self._buffer:
            raise ConcurrencyError(
                "buffer not initialized for send_recv, likely recv called before send"
            )
        elif key not in self._buffer["send_recv"]:
            raise ConcurrencyError(
                f"rank-specific buffer not initialized for send_recv, likely "
                f"recv called before send from rank {from_rank} to rank {self.rank}"
            )
        return_value = self._buffer["send_recv"][key].pop(0)
        return return_value

    def _put_send_recv(self, value, to_rank):
        key = (self.rank, to_rank)
        self._buffer["send_recv"] = self._buffer.get("send_recv", {})
        self._buffer["send_recv"][key] = self._buffer["send_recv"].get(key, [])
        self._buffer["send_recv"][key].append(copy.deepcopy(value))

    @property
    def _bcast_buffer(self):
        if "bcast" not in self._buffer:
            self._buffer["bcast"] = []
        return self._buffer["bcast"]

    @property
    def _scatter_buffer(self):
        if "scatter" not in self._buffer:
            self._buffer["scatter"] = []
        return self._buffer["scatter"]

    @property
    def _gather_buffer(self):
        if "gather" not in self._buffer:
            self._buffer["gather"] = [None for i in range(self.total_ranks)]
        return self._buffer["gather"]

    def bcast(self, value, root=0):
        if root != 0:
            raise NotImplementedError(
                "DummyComm assumes ranks are called in order, so root must be the bcast source"
            )
        value = self._get_buffer("bcast", value)
        logger.debug(f"bcast {value} to rank {self.rank}")
        return value

    def barrier(self):
        return

    def Scatter(self, sendbuf, recvbuf, root=0):
        if root != 0:
            raise NotImplementedError(
                "DummyComm assumes ranks are called in order, so root must be the scatter source"
            )
        sendbuf = self._get_buffer("scatter", sendbuf)
        recvbuf[:] = sendbuf[self.rank]

    def Gather(self, sendbuf, recvbuf, root=0):
        gather_buffer = self._gather_buffer
        gather_buffer[self.rank] = sendbuf
        if self.rank == root:
            # ndarrays are finnicky, have to check for None like this:
            if any(item is None for item in gather_buffer):
                uncalled_ranks = [
                    i for i, val in enumerate(gather_buffer) if val is None
                ]
                raise ConcurrencyError(
                    f"gather called on master rank before ranks {uncalled_ranks}"
                )
            for i, sendbuf in enumerate(gather_buffer):
                recvbuf[i, :] = sendbuf

    def Send(self, sendbuf, dest):
        if isinstance(sendbuf, np.ndarray) and not sendbuf.data.contiguous:
            raise ValueError('ndarray is not contiguous')
        self._put_send_recv(sendbuf, dest)

    def Isend(self, sendbuf, dest):
        return self.Send(sendbuf, dest)

    def Recv(self, recvbuf, source):
        if isinstance(recvbuf, np.ndarray) and not recvbuf.data.contiguous:
            raise ValueError('ndarray is not contiguous')
        recvbuf[:] = self._get_send_recv(source)

    def Irecv(self, recvbuf, source):
        def receive():
            return self.Recv(recvbuf, source)

        return AsyncResult(receive)

    def Split(self, color, key):
        # key argument is ignored, assumes we're calling the ranks from least to
        # greatest when mocking Split
        self._split_comms[color] = self._split_comms.get(color, [])
        self._split_buffers[color] = self._split_buffers.get(color, {})
        rank = len(self._split_comms[color])
        total_ranks = rank + 1
        new_comm = DummyComm(
            rank=rank, total_ranks=total_ranks, buffer_dict=self._split_buffers[color]
        )
        for comm in self._split_comms[color]:
            comm.total_ranks = total_ranks
        self._split_comms[color].append(new_comm)
        return new_comm
