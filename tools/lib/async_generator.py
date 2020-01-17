import functools
import threading
import inspect
import sys
import select
import struct
from math import sqrt
from collections import OrderedDict, deque
from time import time

from tools.lib.pollable_queue import PollableQueue, Empty, Full, ExistentialError

EndSentinel = object()


def _sync_inner_generator(input_queue, *args, **kwargs):
  func = args[0]
  args = args[1:]

  get = input_queue.get
  while True:
    item = get()
    if item is EndSentinel:
      return

    cookie, value = item
    yield cookie, func(value, *args, **kwargs)


def _async_streamer_async_inner(input_queue, output_queue, generator_func, args, kwargs):
  put = output_queue.put
  put_end = True
  try:
    g = generator_func(input_queue, *args, **kwargs)
    for item in g:
      put((time(), item))
    g.close()
  except ExistentialError:
    put_end = False
    raise
  finally:
    if put_end:
      put((None, EndSentinel))

def _running_mean_var(ltc_stats, x):
  old_mean, var = ltc_stats
  mean = min(600., 0.98 * old_mean + 0.02 * x)
  var = min(5., max(0.1, 0.98 * var + 0.02 * (mean - x) * (old_mean - x)))
  return mean, var

def _find_next_resend(sent_messages, ltc_stats):
  if not sent_messages:
    return None, None

  oldest_sent_idx = sent_messages._OrderedDict__root[1][2]
  send_time, _ = sent_messages[oldest_sent_idx]

  # Assume message has been lost if it is >10 standard deviations from mean.
  mean, var = ltc_stats
  next_resend_time = send_time + mean + 40. * sqrt(var)

  return oldest_sent_idx, next_resend_time


def _do_cleanup(input_queue, output_queue, num_workers, sentinels_received, num_outstanding):
  input_fd = input_queue.put_fd()
  output_fd = output_queue.get_fd()

  poller = select.epoll()
  poller.register(input_fd, select.EPOLLOUT)
  poller.register(output_fd, select.EPOLLIN)

  remaining_outputs = []
  end_sentinels_to_send = num_workers - sentinels_received
  while sentinels_received < num_workers:
    evts = dict(poller.poll(-1 if num_outstanding > 0 else 10.))
    if not evts:
      # Workers aren't responding, crash.
      break

    if output_fd in evts:
      _, maybe_sentinel = output_queue.get()
      if maybe_sentinel is EndSentinel:
        sentinels_received += 1
      else:
        remaining_outputs.append(maybe_sentinel[1])
        num_outstanding -= 1

    if input_fd in evts:
      if end_sentinels_to_send > 0:
        input_queue.put_nowait(EndSentinel)
        end_sentinels_to_send -= 1
      else:
        poller.modify(input_fd, 0)

  # TODO: Raise an exception when a queue thread raises one.
  assert sentinels_received == num_workers, (sentinels_received, num_workers)
  assert output_queue.empty()
  return remaining_outputs

def _generate_results(input_stream, input_queue, worker_output_queue, output_queue,
                      num_workers, max_outstanding):
  pack_cookie = struct.pack

  # Maps idx -> (send_time, input)
  sent_messages = OrderedDict()
  oldest_sent_idx = None
  next_resend_time = None
  ltc_stats = 5., 10.

  # Maps idx -> result
  received_messages = {}
  next_out = 0

  # Start things off by pulling the first value.
  next_in_item = next(input_stream, EndSentinel)
  inputs_remain = next_in_item is not EndSentinel
  sentinels_received = 0

  input_fd = input_queue.put_fd()
  worker_output_fd = worker_output_queue.get_fd()
  output_fd = output_queue.put_fd()

  poller = select.epoll()
  poller.register(input_fd, select.EPOLLOUT)
  poller.register(worker_output_fd, select.EPOLLIN)
  poller.register(output_fd, 0)

  # Keep sending/retrying until the input stream and sent messages are all done.
  while sentinels_received < num_workers and (inputs_remain or sent_messages):
    if max_outstanding:
      can_send_new = (len(sent_messages) < max_outstanding and
                      len(received_messages) < max_outstanding and inputs_remain)
    else:
      can_send_new = inputs_remain

    if (next_resend_time and now >= next_resend_time) or can_send_new:
      poller.modify(input_fd, select.EPOLLOUT)
    else:
      poller.modify(input_fd, 0)

    if next_resend_time:
      t = max(0, next_resend_time - now)
      evts = dict(poller.poll(t))
    else:
      evts = dict(poller.poll())
    now = time()

    if output_fd in evts:
      output_queue.put_nowait(received_messages.pop(next_out))
      next_out += 1

      if next_out not in received_messages:
        poller.modify(output_fd, 0)

    if worker_output_fd in evts:
      for receive_time, maybe_sentinel in worker_output_queue.get_multiple_nowait():
        # Check for EndSentinel in case of worker crash.
        if maybe_sentinel is EndSentinel:
          sentinels_received += 1
          continue
        idx_bytes, value = maybe_sentinel
        idx = struct.unpack("<Q", idx_bytes)[0]

        # Don't store duplicates.
        sent_message = sent_messages.pop(idx, None)
        if sent_message is not None:
          received_messages[idx] = value
          ltc_stats = _running_mean_var(ltc_stats, receive_time - sent_message[0])
          if idx == oldest_sent_idx:
            oldest_sent_idx, next_resend_time = _find_next_resend(sent_messages, ltc_stats)

          if idx == next_out:
            poller.modify(output_fd, select.EPOLLOUT)
        else:
          if oldest_sent_idx is not None:
            # The message was resent, so it is at least old as the oldest tracked message.
            ltc_stats = _running_mean_var(ltc_stats, now - sent_messages[oldest_sent_idx][0])
    elif input_fd in evts:
      if can_send_new:
        # We're under the limit, get the next input.
        send_idx, send_value = next_in_item
        input_queue.put_nowait((pack_cookie("<Q", send_idx), send_value))
        sent_messages[next_in_item[0]] = now, next_in_item[1]
        next_in_item = next(input_stream, EndSentinel)
        inputs_remain = next_in_item is not EndSentinel

        if oldest_sent_idx is None:
          oldest_sent_idx, next_resend_time = _find_next_resend(sent_messages, ltc_stats)
      else:
        # Move the resent item to the end.
        send_time, resend_input = sent_messages.pop(oldest_sent_idx)

        sys.stdout.write("Resending {} (ltc, mean, var) = ({}, {}, {})\n".format(
          oldest_sent_idx, now - send_time, ltc_stats[0], ltc_stats[1]))
        input_queue.put_nowait((pack_cookie("<Q", oldest_sent_idx), resend_input))

        sent_messages[oldest_sent_idx] = now, resend_input
        oldest_sent_idx, next_resend_time = _find_next_resend(sent_messages, ltc_stats)

  # Return remaining messages.
  while next_out in received_messages:
    output_queue.put(received_messages.pop(next_out))
    next_out += 1

  _do_cleanup(input_queue, worker_output_queue, num_workers, sentinels_received, 0)
  output_queue.put(EndSentinel)


def _generate_results_unreliable(input_stream, input_queue, worker_output_queue,
                                 output_queue, num_workers, max_outstanding_unused):
  # Start things off by pulling the first value.
  next_in_item = next(input_stream, EndSentinel)
  inputs_remain = next_in_item is not EndSentinel

  # TODO: Use heapq to return oldest message.
  received_messages = deque()
  pack_cookie = struct.pack

  input_fd = input_queue.put_fd()
  worker_output_fd = worker_output_queue.get_fd()
  output_fd = output_queue.put_fd()

  poller = select.epoll()
  poller.register(input_fd, select.EPOLLOUT)
  poller.register(worker_output_fd, select.EPOLLIN)
  poller.register(output_fd, 0)

  # Keep sending/retrying until the input stream and sent messages are all done.
  num_outstanding = 0
  sentinels_received = 0
  while sentinels_received < num_workers and (inputs_remain or received_messages):
    # Avoid poll() if we can easily detect that there is work to do.
    evts = (input_fd if inputs_remain and not input_queue.full() else 0, output_fd
            if not output_queue.full() and len(received_messages) else 0, worker_output_fd
            if not worker_output_queue.empty() else 0)
    if all(evt == 0 for evt in evts):
      evts = dict(poller.poll())

    if output_fd in evts:
      output_queue.put(received_messages.pop())

      if len(received_messages) == 0:
        poller.modify(output_fd, 0)

    if worker_output_fd in evts:
      for receive_time, maybe_sentinel in worker_output_queue.get_multiple():
        # Check for EndSentinel in case of worker crash.
        if maybe_sentinel is EndSentinel:
          sentinels_received += 1
          continue
        received_messages.appendleft(maybe_sentinel[1])
        num_outstanding -= 1
      poller.modify(output_fd, select.EPOLLOUT)

    if input_fd in evts:
      # We're under the limit, get the next input.
      send_idx, send_value = next_in_item
      input_queue.put((pack_cookie("<Q", send_idx), send_value))
      next_in_item = next(input_stream, EndSentinel)
      inputs_remain = next_in_item is not EndSentinel
      num_outstanding += 1

      if not inputs_remain:
        poller.modify(input_fd, 0)

  # TODO: Track latency even though we don't retry.
  for value in _do_cleanup(input_queue, worker_output_queue, num_workers,
                           sentinels_received, num_outstanding):
    output_queue.put(value)
  output_queue.put(EndSentinel)


def _async_generator(func, max_workers, in_q_size, out_q_size, max_outstanding,
                     async_inner, reliable):
  if async_inner:
    assert inspect.isgeneratorfunction(
      func), "async_inner == True but {} is not a generator".format(func)

  @functools.wraps(func)
  def wrapper(input_sequence_or_self, *args, **kwargs):
    # HACK: Determine whether the first arg is "self". ismethod returns False here.
    if inspect.getargspec(func).args[0] == "self":
      inner_func = func.__get__(input_sequence_or_self, type(input_sequence_or_self))
      input_sequence = args[0]
      args = args[1:]
    else:
      inner_func = func
      input_sequence = input_sequence_or_self
    input_stream = enumerate(iter(input_sequence))

    if reliable:
      generate_func = _generate_results
    else:
      generate_func = _generate_results_unreliable

    input_queue = PollableQueue(in_q_size)
    worker_output_queue = PollableQueue(8 * max_workers)
    output_queue = PollableQueue(out_q_size)

    # Start the worker threads.
    if async_inner:
      generator_func = inner_func
    else:
      args = (inner_func,) + args
      generator_func = _sync_inner_generator

    worker_threads = []
    for _ in range(max_workers):
      t = threading.Thread(
        target=_async_streamer_async_inner,
        args=(input_queue, worker_output_queue, generator_func, args, kwargs))
      t.daemon = True
      t.start()
      worker_threads.append(t)

    master_thread = threading.Thread(
      target=generate_func,
      args=(input_stream, input_queue, worker_output_queue, output_queue, max_workers,
            max_outstanding))
    master_thread.daemon = True
    master_thread.start()

    try:
      while True:
        for value in output_queue.get_multiple():
          if value is EndSentinel:
            return
          else:
            yield value
    finally:
      # Make sure work is done and the threads are stopped.
      for t in worker_threads:
        t.join(1)
      master_thread.join(1)

      input_queue.close()
      worker_output_queue.close()
      output_queue.close()

  return wrapper


def async_generator(max_workers=1,
                    in_q_size=10,
                    out_q_size=12,
                    max_outstanding=10000,
                    async_inner=False,
                    reliable=True):
  return (
    lambda f: _async_generator(f, max_workers, in_q_size, out_q_size, max_outstanding, async_inner, reliable)
  )
