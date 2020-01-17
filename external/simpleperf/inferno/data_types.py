#
# Copyright (C) 2016 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


class CallSite:

    def __init__(self, method, dso):
        self.method = method
        self.dso = dso


class Thread:

    def __init__(self, tid, pid):
        self.tid = tid
        self.pid = pid
        self.name = ""
        self.samples = []
        self.flamegraph = FlameGraphCallSite("root", "", 0)
        self.num_samples = 0
        self.num_events = 0

    def add_callchain(self, callchain, symbol, sample):
        self.name = sample.thread_comm
        self.num_samples += 1
        self.num_events += sample.period
        chain = []
        for j in range(callchain.nr):
            entry = callchain.entries[callchain.nr - j - 1]
            if entry.ip == 0:
                continue
            chain.append(CallSite(entry.symbol.symbol_name, entry.symbol.dso_name))

        chain.append(CallSite(symbol.symbol_name, symbol.dso_name))
        self.flamegraph.add_callchain(chain, sample.period)


class Process:

    def __init__(self, name, pid):
        self.name = name
        self.pid = pid
        self.threads = {}
        self.cmd = ""
        self.props = {}
        # num_samples is the count of samples recorded in the profiling file.
        self.num_samples = 0
        # num_events is the count of events contained in all samples. Each sample contains a
        # count of events happened since last sample. If we use cpu-cycles event, the count
        # shows how many cpu-cycles have happened during recording.
        self.num_events = 0

    def get_thread(self, tid, pid):
        thread = self.threads.get(tid)
        if thread is None:
            thread = self.threads[tid] = Thread(tid, pid)
        return thread

    def add_sample(self, sample, symbol, callchain):
        thread = self.get_thread(sample.tid, sample.pid)
        thread.add_callchain(callchain, symbol, sample)
        self.num_samples += 1
        # sample.period is the count of events happened since last sample.
        self.num_events += sample.period


class FlameGraphCallSite:

    callsite_counter = 0
    @classmethod
    def _get_next_callsite_id(cls):
        cls.callsite_counter += 1
        return cls.callsite_counter

    def __init__(self, method, dso, id):
        # map from (dso, method) to FlameGraphCallSite. Used to speed up add_callchain().
        self.child_dict = {}
        self.children = []
        self.method = method
        self.dso = dso
        self.num_events = 0
        self.offset = 0  # Offset allows position nodes in different branches.
        self.id = id

    def weight(self):
        return float(self.num_events)

    def add_callchain(self, chain, num_events):
        self.num_events += num_events
        current = self
        for callsite in chain:
            current = current._get_child(callsite)
            current.num_events += num_events

    def _get_child(self, callsite):
        key = (callsite.dso, callsite.method)
        child = self.child_dict.get(key)
        if child is None:
            child = self.child_dict[key] = FlameGraphCallSite(callsite.method, callsite.dso,
                                               self._get_next_callsite_id())
        return child

    def trim_callchain(self, min_num_events):
        """ Remove call sites with num_events < min_num_events in the subtree.
            Remaining children are collected in a list.
        """
        for key in self.child_dict:
            child = self.child_dict[key]
            if child.num_events >= min_num_events:
                child.trim_callchain(min_num_events)
                self.children.append(child)
        # Relese child_dict since it will not be used.
        self.child_dict = None

    def get_max_depth(self):
        return max([c.get_max_depth() for c in self.children]) + 1 if self.children else 1

    def generate_offset(self, start_offset):
        self.offset = start_offset
        child_offset = start_offset
        for child in self.children:
            child_offset = child.generate_offset(child_offset)
        return self.offset + self.num_events
