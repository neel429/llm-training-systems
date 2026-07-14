"""
Ring AllReduce implemented from scratch using raw point-to-point send/recv.

This reproduces what dist.all_reduce() does internally, broken into the
two classic phases: reduce-scatter, then allgather. Built to run on CPU
with the "gloo" backend so it costs nothing to test, but the algorithm
is backend-agnostic -- the same code runs over NCCL on GPUs.
"""

import torch
import torch.distributed as dist


def ring_all_reduce(tensor: torch.Tensor, rank: int, world_size: int) -> None:
    if world_size == 1:
        return  # nothing to reduce

    # next/prev neighbors in the ring
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    original_numel = tensor.numel()
    pad_amount = (-original_numel) % world_size  # round up to a multiple of world_size

    if pad_amount > 0:
        # pad with zeros so the tensor splits evenly; zeros don't affect
        # a SUM reduction, and we trim them off again at the very end
        padded = torch.cat([tensor.flatten(), tensor.new_zeros(pad_amount)])
    else:
        padded = tensor.flatten()

    chunks = list(padded.chunk(world_size))
    assert len(chunks) == world_size  # guaranteed by the padding above

    # ---- Phase 1: reduce-scatter ----
    # index of the chunk THIS rank currently owns/is sending
    send_idx = rank
    recv_idx = (rank - 1) % world_size

    for step in range(world_size - 1):
        send_chunk = chunks[send_idx].contiguous()
        recv_buffer = torch.empty_like(chunks[recv_idx])

        # odd/even rank ordering on send/recv avoids deadlock: if every
        # rank calls send() then recv(), and the ring is fully
        # synchronous, you can deadlock waiting on each other. Splitting
        # by parity guarantees a consistent half-duplex ordering.
        if rank % 2 == 0:
            dist.send(send_chunk, dst=send_to)
            dist.recv(recv_buffer, src=recv_from)
        else:
            dist.recv(recv_buffer, src=recv_from)
            dist.send(send_chunk, dst=send_to)

        chunks[recv_idx] += recv_buffer

        # advance which chunk we send/receive next, moving backwards
        # around the logical chunk index each step
        send_idx = recv_idx
        recv_idx = (recv_idx - 1) % world_size

    # ---- Phase 2: allgather ----
    # after reduce-scatter, this rank's fully-summed chunk is at index
    # (rank + 1) % world_size
    send_idx = (rank + 1) % world_size
    recv_idx = rank

    for step in range(world_size - 1):
        send_chunk = chunks[send_idx].contiguous()
        recv_buffer = torch.empty_like(chunks[recv_idx])

        if rank % 2 == 0:
            dist.send(send_chunk, dst=send_to)
            dist.recv(recv_buffer, src=recv_from)
        else:
            dist.recv(recv_buffer, src=recv_from)
            dist.send(send_chunk, dst=send_to)

        # this time we OVERWRITE -- the incoming chunk is already a
        # finished sum, we're just relaying it
        chunks[recv_idx].copy_(recv_buffer)

        send_idx = recv_idx
        recv_idx = (recv_idx - 1) % world_size

    # write the result back into the caller's original tensor, trimming
    # off any zero padding we added and restoring the original shape
    result = padded[:original_numel].view(tensor.shape)
    tensor.copy_(result)