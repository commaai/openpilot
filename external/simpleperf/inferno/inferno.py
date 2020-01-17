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

"""
    Inferno is a tool to generate flamegraphs for android programs. It was originally written
    to profile surfaceflinger (Android compositor) but it can be used for other C++ program.
    It uses simpleperf to collect data. Programs have to be compiled with frame pointers which
    excludes ART based programs for the time being.

    Here is how it works:

    1/ Data collection is started via simpleperf and pulled locally as "perf.data".
    2/ The raw format is parsed, callstacks are merged to form a flamegraph data structure.
    3/ The data structure is used to generate a SVG embedded into an HTML page.
    4/ Javascript is injected to allow flamegraph navigation, search, coloring model.

"""

import argparse
import datetime
import os
import subprocess
import sys

scripts_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(scripts_path)
from simpleperf_report_lib import ReportLib
from utils import log_exit, log_info, AdbHelper, open_report_in_browser

from data_types import *
from svg_renderer import *


def collect_data(args):
    """ Run app_profiler.py to generate record file. """
    app_profiler_args = [sys.executable, os.path.join(scripts_path, "app_profiler.py"), "-nb"]
    if args.app:
        app_profiler_args += ["-p", args.app]
    elif args.native_program:
        app_profiler_args += ["-np", args.native_program]
    else:
        log_exit("Please set profiling target with -p or -np option.")
    if args.compile_java_code:
        app_profiler_args.append("--compile_java_code")
    if args.disable_adb_root:
        app_profiler_args.append("--disable_adb_root")
    record_arg_str = ""
    if args.dwarf_unwinding:
        record_arg_str += "-g "
    else:
        record_arg_str += "--call-graph fp "
    if args.events:
        tokens = args.events.split()
        if len(tokens) == 2:
            num_events = tokens[0]
            event_name = tokens[1]
            record_arg_str += "-c %s -e %s " % (num_events, event_name)
        else:
            log_exit("Event format string of -e option cann't be recognized.")
        log_info("Using event sampling (-c %s -e %s)." % (num_events, event_name))
    else:
        record_arg_str += "-f %d " % args.sample_frequency
        log_info("Using frequency sampling (-f %d)." % args.sample_frequency)
    record_arg_str += "--duration %d " % args.capture_duration
    app_profiler_args += ["-r", record_arg_str]
    returncode = subprocess.call(app_profiler_args)
    return returncode == 0


def parse_samples(process, args, sample_filter_fn):
    """Read samples from record file.
        process: Process object
        args: arguments
        sample_filter_fn: if not None, is used to modify and filter samples.
                          It returns false for samples should be filtered out.
    """

    record_file = args.record_file
    symfs_dir = args.symfs
    kallsyms_file = args.kallsyms

    lib = ReportLib()

    lib.ShowIpForUnknownSymbol()
    if symfs_dir:
        lib.SetSymfs(symfs_dir)
    if record_file:
        lib.SetRecordFile(record_file)
    if kallsyms_file:
        lib.SetKallsymsFile(kallsyms_file)
    if args.show_art_frames:
        lib.ShowArtFrames(True)
    process.cmd = lib.GetRecordCmd()
    product_props = lib.MetaInfo().get("product_props")
    if product_props:
        tuple = product_props.split(':')
        process.props['ro.product.manufacturer'] = tuple[0]
        process.props['ro.product.model'] = tuple[1]
        process.props['ro.product.name'] = tuple[2]
    if lib.MetaInfo().get('trace_offcpu') == 'true':
        process.props['trace_offcpu'] = True
        if args.one_flamegraph:
            log_exit("It doesn't make sense to report with --one-flamegraph for perf.data " +
                     "recorded with --trace-offcpu.""")
    else:
        process.props['trace_offcpu'] = False

    while True:
        sample = lib.GetNextSample()
        if sample is None:
            lib.Close()
            break
        symbol = lib.GetSymbolOfCurrentSample()
        callchain = lib.GetCallChainOfCurrentSample()
        if sample_filter_fn and not sample_filter_fn(sample, symbol, callchain):
            continue
        process.add_sample(sample, symbol, callchain)

    if process.pid == 0:
        main_threads = [thread for thread in process.threads.values() if thread.tid == thread.pid]
        if main_threads:
            process.name = main_threads[0].name
            process.pid = main_threads[0].pid

    for thread in process.threads.values():
        min_event_count = thread.num_events * args.min_callchain_percentage * 0.01
        thread.flamegraph.trim_callchain(min_event_count)

    log_info("Parsed %s callchains." % process.num_samples)


def get_local_asset_content(local_path):
    """
    Retrieves local package text content
    :param local_path: str, filename of local asset
    :return: str, the content of local_path
    """
    with open(os.path.join(os.path.dirname(__file__), local_path), 'r') as f:
        return f.read()


def output_report(process, args):
    """
    Generates a HTML report representing the result of simpleperf sampling as flamegraph
    :param process: Process object
    :return: str, absolute path to the file
    """
    f = open(args.report_path, 'w')
    filepath = os.path.realpath(f.name)
    if not args.embedded_flamegraph:
        f.write("<html><body>")
    f.write("<div id='flamegraph_id' style='font-family: Monospace; %s'>" % (
            "display: none;" if args.embedded_flamegraph else ""))
    f.write("""<style type="text/css"> .s { stroke:black; stroke-width:0.5; cursor:pointer;}
            </style>""")
    f.write('<style type="text/css"> .t:hover { cursor:pointer; } </style>')
    f.write('<img height="180" alt = "Embedded Image" src ="data')
    f.write(get_local_asset_content("inferno.b64"))
    f.write('"/>')
    process_entry = ("Process : %s (%d)<br/>" % (process.name, process.pid)) if process.pid else ""
    if process.props['trace_offcpu']:
        event_entry = 'Total time: %s<br/>' % get_proper_scaled_time_string(process.num_events)
    else:
        event_entry = 'Event count: %s<br/>' % ("{:,}".format(process.num_events))
    # TODO: collect capture duration info from perf.data.
    duration_entry = ("Duration: %s seconds<br/>" % args.capture_duration
                      ) if args.capture_duration else ""
    f.write("""<div style='display:inline-block;'>
                  <font size='8'>
                  Inferno Flamegraph Report%s</font><br/><br/>
                  %s
                  Date&nbsp;&nbsp;&nbsp;&nbsp;: %s<br/>
                  Threads : %d <br/>
                  Samples : %d<br/>
                  %s
                  %s""" % (
        (': ' + args.title) if args.title else '',
        process_entry,
        datetime.datetime.now().strftime("%Y-%m-%d (%A) %H:%M:%S"),
        len(process.threads),
        process.num_samples,
        event_entry,
        duration_entry))
    if 'ro.product.model' in process.props:
        f.write(
            "Machine : %s (%s) by %s<br/>" %
            (process.props["ro.product.model"],
             process.props["ro.product.name"],
             process.props["ro.product.manufacturer"]))
    if process.cmd:
        f.write("Capture : %s<br/><br/>" % process.cmd)
    f.write("</div>")
    f.write("""<br/><br/>
            <div>Navigate with WASD, zoom in with SPACE, zoom out with BACKSPACE.</div>""")
    f.write("<script>%s</script>" % get_local_asset_content("script.js"))
    if not args.embedded_flamegraph:
        f.write("<script>document.addEventListener('DOMContentLoaded', flamegraphInit);</script>")

    # Sort threads by the event count in a thread.
    for thread in sorted(process.threads.values(), key=lambda x: x.num_events, reverse=True):
        f.write("<br/><br/><b>Thread %d (%s) (%d samples):</b><br/>\n\n\n\n" % (
                thread.tid, thread.name, thread.num_samples))
        renderSVG(process, thread.flamegraph, f, args.color)

    f.write("</div>")
    if not args.embedded_flamegraph:
        f.write("</body></html")
    f.close()
    return "file://" + filepath


def generate_threads_offsets(process):
    for thread in process.threads.values():
       thread.flamegraph.generate_offset(0)


def collect_machine_info(process):
    adb = AdbHelper()
    process.props = {}
    process.props['ro.product.model'] = adb.get_property('ro.product.model')
    process.props['ro.product.name'] = adb.get_property('ro.product.name')
    process.props['ro.product.manufacturer'] = adb.get_property('ro.product.manufacturer')


def main():
    # Allow deep callchain with length >1000.
    sys.setrecursionlimit(1500)
    parser = argparse.ArgumentParser(description="""Report samples in perf.data. Default option
                                                    is: "-np surfaceflinger -f 6000 -t 10".""")
    record_group = parser.add_argument_group('Record options')
    record_group.add_argument('-du', '--dwarf_unwinding', action='store_true', help="""Perform
                              unwinding using dwarf instead of fp.""")
    record_group.add_argument('-e', '--events', default="", help="""Sample based on event
                              occurences instead of frequency. Format expected is
                              "event_counts event_name". e.g: "10000 cpu-cyles". A few examples
                              of event_name: cpu-cycles, cache-references, cache-misses,
                              branch-instructions, branch-misses""")
    record_group.add_argument('-f', '--sample_frequency', type=int, default=6000, help="""Sample
                              frequency""")
    record_group.add_argument('--compile_java_code', action='store_true',
                              help="""On Android N and Android O, we need to compile Java code
                                      into native instructions to profile Java code. Android O
                                      also needs wrap.sh in the apk to use the native
                                      instructions.""")
    record_group.add_argument('-np', '--native_program', default="surfaceflinger", help="""Profile
                              a native program. The program should be running on the device.
                              Like -np surfaceflinger.""")
    record_group.add_argument('-p', '--app', help="""Profile an Android app, given the package
                              name. Like -p com.example.android.myapp.""")
    record_group.add_argument('--record_file', default='perf.data', help='Default is perf.data.')
    record_group.add_argument('-sc', '--skip_collection', action='store_true', help="""Skip data
                              collection""")
    record_group.add_argument('-t', '--capture_duration', type=int, default=10, help="""Capture
                              duration in seconds.""")

    report_group = parser.add_argument_group('Report options')
    report_group.add_argument('-c', '--color', default='hot', choices=['hot', 'dso', 'legacy'],
                              help="""Color theme: hot=percentage of samples, dso=callsite DSO
                                      name, legacy=brendan style""")
    report_group.add_argument('--embedded_flamegraph', action='store_true', help="""Generate
                              embedded flamegraph.""")
    report_group.add_argument('--kallsyms', help='Set the path to find kernel symbols.')
    report_group.add_argument('--min_callchain_percentage', default=0.01, type=float, help="""
                              Set min percentage of callchains shown in the report.
                              It is used to limit nodes shown in the flamegraph. For example,
                              when set to 0.01, only callchains taking >= 0.01%% of the event
                              count of the owner thread are collected in the report.""")
    report_group.add_argument('--no_browser', action='store_true', help="""Don't open report
                              in browser.""")
    report_group.add_argument('-o', '--report_path', default='report.html', help="""Set report
                              path.""")
    report_group.add_argument('--one-flamegraph', action='store_true', help="""Generate one
                              flamegraph instead of one for each thread.""")
    report_group.add_argument('--symfs', help="""Set the path to find binaries with symbols and
                              debug info.""")
    report_group.add_argument('--title', help='Show a title in the report.')
    report_group.add_argument('--show_art_frames', action='store_true',
                              help='Show frames of internal methods in the ART Java interpreter.')

    debug_group = parser.add_argument_group('Debug options')
    debug_group.add_argument('--disable_adb_root', action='store_true', help="""Force adb to run
                             in non root mode.""")
    args = parser.parse_args()
    process = Process("", 0)

    if not args.skip_collection:
        process.name = args.app or args.native_program
        log_info("Starting data collection stage for process '%s'." % process.name)
        if not collect_data(args):
            log_exit("Unable to collect data.")
        result, output = AdbHelper().run_and_return_output(['shell', 'pidof', process.name])
        if result:
            try:
                process.pid = int(output)
            except:
                process.pid = 0
        collect_machine_info(process)
    else:
        args.capture_duration = 0

    sample_filter_fn = None
    if args.one_flamegraph:
        def filter_fn(sample, symbol, callchain):
            sample.pid = sample.tid = process.pid
            return True
        sample_filter_fn = filter_fn
        if not args.title:
            args.title = ''
        args.title += '(One Flamegraph)'

    parse_samples(process, args, sample_filter_fn)
    generate_threads_offsets(process)
    report_path = output_report(process, args)
    if not args.no_browser:
        open_report_in_browser(report_path)

    log_info("Flamegraph generated at '%s'." % report_path)

if __name__ == "__main__":
    main()
