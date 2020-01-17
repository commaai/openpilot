/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
'use strict';

// Use IIFE to avoid leaking names to other scripts.
$(document).ready(function() {

function openHtml(name, attrs={}) {
    let s = `<${name} `;
    for (let key in attrs) {
        s += `${key}="${attrs[key]}" `;
    }
    s += '>';
    return s;
}

function closeHtml(name) {
    return `</${name}>`;
}

function getHtml(name, attrs={}) {
    let text;
    if ('text' in attrs) {
        text = attrs.text;
        delete attrs.text;
    }
    let s = openHtml(name, attrs);
    if (text) {
        s += text;
    }
    s += closeHtml(name);
    return s;
}

function getTableRow(cols, colName, attrs={}) {
    let s = openHtml('tr', attrs);
    for (let col of cols) {
        s += `<${colName}>${col}</${colName}>`;
    }
    s += '</tr>';
    return s;
}

function toPercentageStr(percentage) {
    return percentage.toFixed(2) + '%';
}

function getProcessName(pid) {
    let name = gProcesses[pid];
    return name ? `${pid} (${name})`: pid.toString();
}

function getThreadName(tid) {
    let name = gThreads[tid];
    return name ? `${tid} (${name})`: tid.toString();
}

function getLibName(libId) {
    return gLibList[libId];
}

function getFuncName(funcId) {
    return gFunctionMap[funcId].f;
}

function getLibNameOfFunction(funcId) {
    return getLibName(gFunctionMap[funcId].l);
}

function getFuncSourceRange(funcId) {
    let func = gFunctionMap[funcId];
    if (func.hasOwnProperty('s')) {
        return {fileId: func.s[0], startLine: func.s[1], endLine: func.s[2]};
    }
    return null;
}

function getFuncDisassembly(funcId) {
    let func = gFunctionMap[funcId];
    return func.hasOwnProperty('d') ? func.d : null;
}

function getSourceFilePath(sourceFileId) {
    return gSourceFiles[sourceFileId].path;
}

function getSourceCode(sourceFileId) {
    return gSourceFiles[sourceFileId].code;
}

function isClockEvent(eventInfo) {
    return eventInfo.eventName.includes('task-clock') ||
            eventInfo.eventName.includes('cpu-clock');
}

class TabManager {
    constructor(divContainer) {
        this.div = $('<div>', {id: 'tabs'});
        this.div.appendTo(divContainer);
        this.div.append(getHtml('ul'));
        this.tabs = [];
        this.isDrawCalled = false;
    }

    addTab(title, tabObj) {
        let id = 'tab_' + this.div.children().length;
        let tabDiv = $('<div>', {id: id});
        tabDiv.appendTo(this.div);
        this.div.children().first().append(
            getHtml('li', {text: getHtml('a', {href: '#' + id, text: title})}));
        tabObj.init(tabDiv);
        this.tabs.push(tabObj);
        if (this.isDrawCalled) {
            this.div.tabs('refresh');
        }
        return tabObj;
    }

    findTab(title) {
        let links = this.div.find('li a');
        for (let i = 0; i < links.length; ++i) {
            if (links.eq(i).text() == title) {
                return this.tabs[i];
            }
        }
        return null;
    }

    draw() {
        this.div.tabs({
            active: 0,
        });
        this.tabs.forEach(function(tab) {
            tab.draw();
        });
        this.isDrawCalled = true;
    }

    setActive(tabObj) {
        for (let i = 0; i < this.tabs.length; ++i) {
            if (this.tabs[i] == tabObj) {
                this.div.tabs('option', 'active', i);
                break;
            }
        }
    }
}

// Show global information retrieved from the record file, including:
//   record time
//   machine type
//   Android version
//   record cmdline
//   total samples
class RecordFileView {
    constructor(divContainer) {
        this.div = $('<div>');
        this.div.appendTo(divContainer);
    }

    draw() {
        google.charts.setOnLoadCallback(() => this.realDraw());
    }

    realDraw() {
        this.div.empty();
        // Draw a table of 'Name', 'Value'.
        let rows = [];
        if (gRecordInfo.recordTime) {
            rows.push(['Record Time', gRecordInfo.recordTime]);
        }
        if (gRecordInfo.machineType) {
            rows.push(['Machine Type', gRecordInfo.machineType]);
        }
        if (gRecordInfo.androidVersion) {
            rows.push(['Android Version', gRecordInfo.androidVersion]);
        }
        if (gRecordInfo.recordCmdline) {
            rows.push(['Record cmdline', gRecordInfo.recordCmdline]);
        }
        rows.push(['Total Samples', '' + gRecordInfo.totalSamples]);

        let data = new google.visualization.DataTable();
        data.addColumn('string', '');
        data.addColumn('string', '');
        data.addRows(rows);
        for (let i = 0; i < rows.length; ++i) {
            data.setProperty(i, 0, 'className', 'boldTableCell');
        }
        let table = new google.visualization.Table(this.div.get(0));
        table.draw(data, {
            width: '100%',
            sort: 'disable',
            allowHtml: true,
            cssClassNames: {
                'tableCell': 'tableCell',
            },
        });
    }
}

// Show pieChart of event count percentage of each process, thread, library and function.
class ChartView {
    constructor(divContainer, eventInfo) {
        this.id = divContainer.children().length;
        this.div = $('<div>', {id: 'chartstat_' + this.id});
        this.div.appendTo(divContainer);
        this.eventInfo = eventInfo;
        this.processInfo = null;
        this.threadInfo = null;
        this.libInfo = null;
        this.states = {
            SHOW_EVENT_INFO: 1,
            SHOW_PROCESS_INFO: 2,
            SHOW_THREAD_INFO: 3,
            SHOW_LIB_INFO: 4,
        };
        if (isClockEvent(this.eventInfo)) {
            this.getSampleWeight = function (eventCount) {
                return (eventCount / 1000000.0).toFixed(3) + ' ms';
            }
        } else {
            this.getSampleWeight = (eventCount) => '' + eventCount;
        }
    }

    _getState() {
        if (this.libInfo) {
            return this.states.SHOW_LIB_INFO;
        }
        if (this.threadInfo) {
            return this.states.SHOW_THREAD_INFO;
        }
        if (this.processInfo) {
            return this.states.SHOW_PROCESS_INFO;
        }
        return this.states.SHOW_EVENT_INFO;
    }

    _goBack() {
        let state = this._getState();
        if (state == this.states.SHOW_PROCESS_INFO) {
            this.processInfo = null;
        } else if (state == this.states.SHOW_THREAD_INFO) {
            this.threadInfo = null;
        } else if (state == this.states.SHOW_LIB_INFO) {
            this.libInfo = null;
        }
        this.draw();
    }

    _selectHandler(chart) {
        let selectedItem = chart.getSelection()[0];
        if (selectedItem) {
            let state = this._getState();
            if (state == this.states.SHOW_EVENT_INFO) {
                this.processInfo = this.eventInfo.processes[selectedItem.row];
            } else if (state == this.states.SHOW_PROCESS_INFO) {
                this.threadInfo = this.processInfo.threads[selectedItem.row];
            } else if (state == this.states.SHOW_THREAD_INFO) {
                this.libInfo = this.threadInfo.libs[selectedItem.row];
            }
            this.draw();
        }
    }

    draw() {
        google.charts.setOnLoadCallback(() => this.realDraw());
    }

    realDraw() {
        this.div.empty();
        this._drawTitle();
        this._drawPieChart();
    }

    _drawTitle() {
        // Draw a table of 'Name', 'Event Count'.
        let rows = [];
        rows.push(['Event Type: ' + this.eventInfo.eventName,
                   this.getSampleWeight(this.eventInfo.eventCount)]);
        if (this.processInfo) {
            rows.push(['Process: ' + getProcessName(this.processInfo.pid),
                       this.getSampleWeight(this.processInfo.eventCount)]);
        }
        if (this.threadInfo) {
            rows.push(['Thread: ' + getThreadName(this.threadInfo.tid),
                       this.getSampleWeight(this.threadInfo.eventCount)]);
        }
        if (this.libInfo) {
            rows.push(['Library: ' + getLibName(this.libInfo.libId),
                       this.getSampleWeight(this.libInfo.eventCount)]);
        }
        let data = new google.visualization.DataTable();
        data.addColumn('string', '');
        data.addColumn('string', '');
        data.addRows(rows);
        for (let i = 0; i < rows.length; ++i) {
            data.setProperty(i, 0, 'className', 'boldTableCell');
        }
        let wrapperDiv = $('<div>');
        wrapperDiv.appendTo(this.div);
        let table = new google.visualization.Table(wrapperDiv.get(0));
        table.draw(data, {
            width: '100%',
            sort: 'disable',
            allowHtml: true,
            cssClassNames: {
                'tableCell': 'tableCell',
            },
        });
        if (this._getState() != this.states.SHOW_EVENT_INFO) {
            let button = $('<button>', {text: 'Back'});
            button.appendTo(this.div);
            button.button().click(() => this._goBack());
        }
    }

    _drawPieChart() {
        let state = this._getState();
        let title = null;
        let firstColumn = null;
        let rows = [];
        let thisObj = this;
        function getItem(name, eventCount, totalEventCount) {
            let sampleWeight = thisObj.getSampleWeight(eventCount);
            let percent = (eventCount * 100.0 / totalEventCount).toFixed(2) + '%';
            return [name, eventCount, getHtml('pre', {text: name}) +
                        getHtml('b', {text: `${sampleWeight} (${percent})`})];
        }

        if (state == this.states.SHOW_EVENT_INFO) {
            title = 'Processes in event type ' + this.eventInfo.eventName;
            firstColumn = 'Process';
            for (let process of this.eventInfo.processes) {
                rows.push(getItem('Process: ' + getProcessName(process.pid), process.eventCount,
                                  this.eventInfo.eventCount));
            }
        } else if (state == this.states.SHOW_PROCESS_INFO) {
            title = 'Threads in process ' + getProcessName(this.processInfo.pid);
            firstColumn = 'Thread';
            for (let thread of this.processInfo.threads) {
                rows.push(getItem('Thread: ' + getThreadName(thread.tid), thread.eventCount,
                                  this.processInfo.eventCount));
            }
        } else if (state == this.states.SHOW_THREAD_INFO) {
            title = 'Libraries in thread ' + getThreadName(this.threadInfo.tid);
            firstColumn = 'Library';
            for (let lib of this.threadInfo.libs) {
                rows.push(getItem('Library: ' + getLibName(lib.libId), lib.eventCount,
                                  this.threadInfo.eventCount));
            }
        } else if (state == this.states.SHOW_LIB_INFO) {
            title = 'Functions in library ' + getLibName(this.libInfo.libId);
            firstColumn = 'Function';
            for (let func of this.libInfo.functions) {
                rows.push(getItem('Function: ' + getFuncName(func.g.f), func.g.e,
                                  this.libInfo.eventCount));
            }
        }
        let data = new google.visualization.DataTable();
        data.addColumn('string', firstColumn);
        data.addColumn('number', 'EventCount');
        data.addColumn({type: 'string', role: 'tooltip', p: {html: true}});
        data.addRows(rows);

        let wrapperDiv = $('<div>');
        wrapperDiv.appendTo(this.div);
        let chart = new google.visualization.PieChart(wrapperDiv.get(0));
        chart.draw(data, {
            title: title,
            width: 1000,
            height: 600,
            tooltip: {isHtml: true},
        });
        google.visualization.events.addListener(chart, 'select', () => this._selectHandler(chart));
    }
}


class ChartStatTab {
    constructor() {
    }

    init(div) {
        this.div = div;
        this.recordFileView = new RecordFileView(this.div);
        this.chartViews = [];
        for (let eventInfo of gSampleInfo) {
            this.chartViews.push(new ChartView(this.div, eventInfo));
        }
    }

    draw() {
        this.recordFileView.draw();
        for (let charView of this.chartViews) {
            charView.draw();
        }
    }
}


class SampleTableTab {
    constructor() {
    }

    init(div) {
        this.div = div;
        this.selectorView = null;
        this.sampleTableViews = [];
    }

    draw() {
        this.selectorView = new SampleTableWeightSelectorView(this.div, gSampleInfo[0],
                                                              () => this.onSampleWeightChange());
        this.selectorView.draw();
        for (let eventInfo of gSampleInfo) {
            this.div.append(getHtml('hr'));
            this.sampleTableViews.push(new SampleTableView(this.div, eventInfo));
        }
        this.onSampleWeightChange();
    }

    onSampleWeightChange() {
        for (let i = 0; i < gSampleInfo.length; ++i) {
            let sampleWeightFunction = this.selectorView.getSampleWeightFunction(gSampleInfo[i]);
            let sampleWeightSuffix = this.selectorView.getSampleWeightSuffix(gSampleInfo[i]);
            this.sampleTableViews[i].draw(sampleWeightFunction, sampleWeightSuffix);
        }
    }
}

// Select the way to show sample weight in SampleTableTab.
// 1. Show percentage of event count.
// 2. Show event count (For cpu-clock and task-clock events, it is time in ms).
class SampleTableWeightSelectorView {
    constructor(divContainer, firstEventInfo, onSelectChange) {
        this.div = $('<div>');
        this.div.appendTo(divContainer);
        this.onSelectChange = onSelectChange;
        this.options = {
            SHOW_PERCENT: 0,
            SHOW_EVENT_COUNT: 1,
        };
        if (isClockEvent(firstEventInfo)) {
            this.curOption = this.options.SHOW_EVENT_COUNT;
        } else {
            this.curOption = this.options.SHOW_PERCENT;
        }
    }

    draw() {
        let options = ['Show percentage of event count', 'Show event count'];
        let optionStr = '';
        for (let i = 0; i < options.length; ++i) {
            optionStr += getHtml('option', {value: i, text: options[i]});
        }
        this.div.append(getHtml('select', {text: optionStr}));
        let selectMenu = this.div.children().last();
        selectMenu.children().eq(this.curOption).attr('selected', 'selected');
        let thisObj = this;
        selectMenu.selectmenu({
            change: function() {
                thisObj.curOption = this.value;
                thisObj.onSelectChange();
            },
            width: '100%',
        });
    }

    getSampleWeightFunction(eventInfo) {
        if (this.curOption == this.options.SHOW_PERCENT) {
            return function(eventCount) {
                return (eventCount * 100.0 / eventInfo.eventCount).toFixed(2) + '%';
            }
        }
        if (isClockEvent(eventInfo)) {
            return (eventCount) => (eventCount / 1000000.0).toFixed(3);
        }
        return (eventCount) => '' + eventCount;
    }

    getSampleWeightSuffix(eventInfo) {
        if (this.curOption == this.options.SHOW_EVENT_COUNT && isClockEvent(eventInfo)) {
            return ' ms';
        }
        return '';
    }
}


class SampleTableView {
    constructor(divContainer, eventInfo) {
        this.id = divContainer.children().length;
        this.div = $('<div>');
        this.div.appendTo(divContainer);
        this.eventInfo = eventInfo;
    }

    draw(getSampleWeight, sampleWeightSuffix) {
        // Draw a table of 'Total', 'Self', 'Samples', 'Process', 'Thread', 'Library', 'Function'.
        this.div.empty();
        let eventInfo = this.eventInfo;
        let sampleWeight = getSampleWeight(eventInfo.eventCount);
        this.div.append(getHtml('p', {text: `Sample table for event ${eventInfo.eventName}, ` +
                `total count ${sampleWeight}${sampleWeightSuffix}`}));
        let tableId = 'sampleTable_' + this.id;
        let valueSuffix = sampleWeightSuffix.length > 0 ? `(in${sampleWeightSuffix})` : '';
        let titles = ['Total' + valueSuffix, 'Self' + valueSuffix, 'Samples',
                      'Process', 'Thread', 'Library', 'Function'];
        let tableStr = openHtml('table', {id: tableId, cellspacing: '0', width: '100%'}) +
                        getHtml('thead', {text: getTableRow(titles, 'th')}) +
                        getHtml('tfoot', {text: getTableRow(titles, 'th')}) +
                        openHtml('tbody');
        for (let i = 0; i < eventInfo.processes.length; ++i) {
            let processInfo = eventInfo.processes[i];
            let processName = getProcessName(processInfo.pid);
            for (let j = 0; j < processInfo.threads.length; ++j) {
                let threadInfo = processInfo.threads[j];
                let threadName = getThreadName(threadInfo.tid);
                for (let k = 0; k < threadInfo.libs.length; ++k) {
                    let lib = threadInfo.libs[k];
                    let libName = getLibName(lib.libId);
                    for (let t = 0; t < lib.functions.length; ++t) {
                        let func = lib.functions[t];
                        let key = [i, j, k, t].join('_');
                        let totalValue = getSampleWeight(func.g.s);
                        let selfValue = getSampleWeight(func.g.e);
                        tableStr += getTableRow([totalValue, selfValue, func.c,
                                                 processName, threadName, libName,
                                                 getFuncName(func.g.f)], 'td', {key: key});
                    }
                }
            }
        }
        tableStr += closeHtml('tbody') + closeHtml('table');
        this.div.append(tableStr);
        let table = this.div.find(`table#${tableId}`).dataTable({
            lengthMenu: [10, 20, 50, 100, -1],
            processing: true,
            order: [0, 'desc'],
            responsive: true,
        });

        table.find('tr').css('cursor', 'pointer');
        table.on('click', 'tr', function() {
            let key = this.getAttribute('key');
            if (!key) {
                return;
            }
            let indexes = key.split('_');
            let processInfo = eventInfo.processes[indexes[0]];
            let threadInfo = processInfo.threads[indexes[1]];
            let lib = threadInfo.libs[indexes[2]];
            let func = lib.functions[indexes[3]];
            FunctionTab.showFunction(eventInfo, processInfo, threadInfo, lib, func);
        });
    }
}


// Show embedded flamegraph generated by inferno.
class FlameGraphTab {
    constructor() {
    }

    init(div) {
        this.div = div;
    }

    draw() {
        $('div#flamegraph_id').appendTo(this.div).css('display', 'block');
        flamegraphInit();
    }
}


// FunctionTab: show information of a function.
// 1. Show the callgrpah and reverse callgraph of a function as flamegraphs.
// 2. Show the annotated source code of the function.
class FunctionTab {
    static showFunction(eventInfo, processInfo, threadInfo, lib, func) {
        let title = 'Function';
        let tab = gTabs.findTab(title);
        if (!tab) {
            tab = gTabs.addTab(title, new FunctionTab());
        }
        tab.setFunction(eventInfo, processInfo, threadInfo, lib, func);
    }

    constructor() {
        this.func = null;
        this.selectPercent = 'thread';
    }

    init(div) {
        this.div = div;
    }

    setFunction(eventInfo, processInfo, threadInfo, lib, func) {
        this.eventInfo = eventInfo;
        this.processInfo = processInfo;
        this.threadInfo = threadInfo;
        this.lib = lib;
        this.func = func;
        this.selectorView = null;
        this.callgraphView = null;
        this.reverseCallgraphView = null;
        this.sourceCodeView = null;
        this.disassemblyView = null;
        this.draw();
        gTabs.setActive(this);
    }

    draw() {
        if (!this.func) {
            return;
        }
        this.div.empty();
        this._drawTitle();

        this.selectorView = new FunctionSampleWeightSelectorView(this.div, this.eventInfo,
            this.processInfo, this.threadInfo, () => this.onSampleWeightChange());
        this.selectorView.draw();

        this.div.append(getHtml('hr'));
        let funcName = getFuncName(this.func.g.f);
        this.div.append(getHtml('b', {text: `Functions called by ${funcName}`}) + '<br/>');
        this.callgraphView = new FlameGraphView(this.div, this.func.g, false);

        this.div.append(getHtml('hr'));
        this.div.append(getHtml('b', {text: `Functions calling ${funcName}`}) + '<br/>');
        this.reverseCallgraphView = new FlameGraphView(this.div, this.func.rg, true);

        let sourceFiles = collectSourceFilesForFunction(this.func);
        if (sourceFiles) {
            this.div.append(getHtml('hr'));
            this.div.append(getHtml('b', {text: 'SourceCode:'}) + '<br/>');
            this.sourceCodeView = new SourceCodeView(this.div, sourceFiles);
        }

        let disassembly = collectDisassemblyForFunction(this.func);
        if (disassembly) {
            this.div.append(getHtml('hr'));
            this.div.append(getHtml('b', {text: 'Disassembly:'}) + '<br/>');
            this.disassemblyView = new DisassemblyView(this.div, disassembly);
        }

        this.onSampleWeightChange();  // Manually set sample weight function for the first time.
    }

    _drawTitle() {
        let eventName = this.eventInfo.eventName;
        let processName = getProcessName(this.processInfo.pid);
        let threadName = getThreadName(this.threadInfo.tid);
        let libName = getLibName(this.lib.libId);
        let funcName = getFuncName(this.func.g.f);
        // Draw a table of 'Name', 'Value'.
        let rows = [];
        rows.push(['Event Type', eventName]);
        rows.push(['Process', processName]);
        rows.push(['Thread', threadName]);
        rows.push(['Library', libName]);
        rows.push(['Function', getHtml('pre', {text: funcName})]);
        let data = new google.visualization.DataTable();
        data.addColumn('string', '');
        data.addColumn('string', '');
        data.addRows(rows);
        for (let i = 0; i < rows.length; ++i) {
            data.setProperty(i, 0, 'className', 'boldTableCell');
        }
        let wrapperDiv = $('<div>');
        wrapperDiv.appendTo(this.div);
        let table = new google.visualization.Table(wrapperDiv.get(0));
        table.draw(data, {
            width: '100%',
            sort: 'disable',
            allowHtml: true,
            cssClassNames: {
                'tableCell': 'tableCell',
            },
        });
    }

    onSampleWeightChange() {
        let sampleWeightFunction = this.selectorView.getSampleWeightFunction();
        if (this.callgraphView) {
            this.callgraphView.draw(sampleWeightFunction);
        }
        if (this.reverseCallgraphView) {
            this.reverseCallgraphView.draw(sampleWeightFunction);
        }
        if (this.sourceCodeView) {
            this.sourceCodeView.draw(sampleWeightFunction);
        }
        if (this.disassemblyView) {
            this.disassemblyView.draw(sampleWeightFunction);
        }
    }
}


// Select the way to show sample weight in FunctionTab.
// 1. Show percentage of event count relative to all processes.
// 2. Show percentage of event count relative to the current process.
// 3. Show percentage of event count relative to the current thread.
// 4. Show absolute event count.
// 5. Show event count in milliseconds, only possible for cpu-clock or task-clock events.
class FunctionSampleWeightSelectorView {
    constructor(divContainer, eventInfo, processInfo, threadInfo, onSelectChange) {
        this.div = $('<div>');
        this.div.appendTo(divContainer);
        this.onSelectChange = onSelectChange;
        this.eventCountForAllProcesses = eventInfo.eventCount;
        this.eventCountForProcess = processInfo.eventCount;
        this.eventCountForThread = threadInfo.eventCount;
        this.options = {
            PERCENT_TO_ALL_PROCESSES: 0,
            PERCENT_TO_CUR_PROCESS: 1,
            PERCENT_TO_CUR_THREAD: 2,
            RAW_EVENT_COUNT: 3,
            EVENT_COUNT_IN_TIME: 4,
        };
        let name = eventInfo.eventName;
        this.supportEventCountInTime = isClockEvent(eventInfo);
        if (this.supportEventCountInTime) {
            this.curOption = this.options.EVENT_COUNT_IN_TIME;
        } else {
            this.curOption = this.options.PERCENT_TO_CUR_THREAD;
        }
    }

    draw() {
        let options = [];
        options.push('Show percentage of event count relative to all processes.');
        options.push('Show percentage of event count relative to the current process.');
        options.push('Show percentage of event count relative to the current thread.');
        options.push('Show event count.');
        if (this.supportEventCountInTime) {
            options.push('Show event count in milliseconds.');
        }
        let optionStr = '';
        for (let i = 0; i < options.length; ++i) {
            optionStr += getHtml('option', {value: i, text: options[i]});
        }
        this.div.append(getHtml('select', {text: optionStr}));
        let selectMenu = this.div.children().last();
        selectMenu.children().eq(this.curOption).attr('selected', 'selected');
        let thisObj = this;
        selectMenu.selectmenu({
            change: function() {
                thisObj.curOption = this.value;
                thisObj.onSelectChange();
            },
            width: '100%',
        });
    }

    getSampleWeightFunction() {
        let thisObj = this;
        if (this.curOption == this.options.PERCENT_TO_ALL_PROCESSES) {
            return function(eventCount) {
                let percent = eventCount * 100.0 / thisObj.eventCountForAllProcesses;
                return percent.toFixed(2) + '%';
            };
        }
        if (this.curOption == this.options.PERCENT_TO_CUR_PROCESS) {
            return function(eventCount) {
                let percent = eventCount * 100.0 / thisObj.eventCountForProcess;
                return percent.toFixed(2) + '%';
            };
        }
        if (this.curOption == this.options.PERCENT_TO_CUR_THREAD) {
            return function(eventCount) {
                let percent = eventCount * 100.0 / thisObj.eventCountForThread;
                return percent.toFixed(2) + '%';
            };
        }
        if (this.curOption == this.options.RAW_EVENT_COUNT) {
            return function(eventCount) {
                return '' + eventCount;
            };
        }
        if (this.curOption == this.options.EVENT_COUNT_IN_TIME) {
            return function(eventCount) {
                let timeInMs = eventCount / 1000000.0;
                return timeInMs.toFixed(3) + ' ms';
            };
        }
    }
}


// Given a callgraph, show the flamegraph.
class FlameGraphView {
    // If reverseOrder is false, the root of the flamegraph is at the bottom,
    // otherwise it is at the top.
    constructor(divContainer, callgraph, reverseOrder) {
        this.id = divContainer.children().length;
        this.div = $('<div>', {id: 'fg_' + this.id});
        this.div.appendTo(divContainer);
        this.callgraph = callgraph;
        this.reverseOrder = reverseOrder;
        this.sampleWeightFunction = null;
        this.svgWidth = $(window).width();
        this.svgNodeHeight = 17;
        this.fontSize = 12;

        function getMaxDepth(node) {
            let depth = 0;
            for (let child of node.c) {
                depth = Math.max(depth, getMaxDepth(child));
            }
            return depth + 1;
        }
        this.maxDepth = getMaxDepth(this.callgraph);
        this.svgHeight = this.svgNodeHeight * (this.maxDepth + 3);
    }

    draw(sampleWeightFunction) {
        this.sampleWeightFunction = sampleWeightFunction;
        this.div.empty();
        this.div.css('width', '100%').css('height', this.svgHeight + 'px');
        let svgStr = '<svg xmlns="http://www.w3.org/2000/svg" \
        xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" \
        width="100%" height="100%" style="border: 1px solid black; font-family: Monospace;"> \
        </svg>';
        this.div.append(svgStr);
        this.svg = this.div.find('svg');
        this._renderBackground();
        this._renderSvgNodes(this.callgraph, 0, 0);
        this._renderUnzoomNode();
        this._renderInfoNode();
        this._renderPercentNode();
        // Make the added nodes in the svg visible.
        this.div.html(this.div.html());
        this.svg = this.div.find('svg');
        this._adjustTextSize();
        this._enableZoom();
        this._enableInfo();
        this._adjustTextSizeOnResize();
    }

    _renderBackground() {
        this.svg.append(`<defs > <linearGradient id="background_gradient_${this.id}"
                                  y1="0" y2="1" x1="0" x2="0" > \
                                  <stop stop-color="#eeeeee" offset="5%" /> \
                                  <stop stop-color="#efefb1" offset="90%" /> \
                                  </linearGradient> \
                         </defs> \
                         <rect x="0" y="0" width="100%" height="100%" \
                           fill="url(#background_gradient_${this.id})" />`);
    }

    _getYForDepth(depth) {
        if (this.reverseOrder) {
            return (depth + 3) * this.svgNodeHeight;
        }
        return this.svgHeight - (depth + 1) * this.svgNodeHeight;
    }

    _getWidthPercentage(eventCount) {
        return eventCount * 100.0 / this.callgraph.s;
    }

    _getHeatColor(widthPercentage) {
        return {
            r: Math.floor(245 + 10 * (1 - widthPercentage * 0.01)),
            g: Math.floor(110 + 105 * (1 - widthPercentage * 0.01)),
            b: 100,
        };
    }

    _renderSvgNodes(callNode, depth, xOffset) {
        let x = xOffset;
        let y = this._getYForDepth(depth);
        let width = this._getWidthPercentage(callNode.s);
        if (width < 0.1) {
            return xOffset;
        }
        let color = this._getHeatColor(width);
        let borderColor = {};
        for (let key in color) {
            borderColor[key] = Math.max(0, color[key] - 50);
        }
        let funcName = getFuncName(callNode.f);
        let libName = getLibNameOfFunction(callNode.f);
        let sampleWeight = this.sampleWeightFunction(callNode.s);
        let title = funcName + ' | ' + libName + ' (' + callNode.s + ' events: ' +
                    sampleWeight + ')';
        this.svg.append(`<g> <title>${title}</title> <rect x="${x}%" y="${y}" ox="${x}" \
                        depth="${depth}" width="${width}%" owidth="${width}" height="15.0" \
                        ofill="rgb(${color.r},${color.g},${color.b})" \
                        fill="rgb(${color.r},${color.g},${color.b})" \
                        style="stroke:rgb(${borderColor.r},${borderColor.g},${borderColor.b})"/> \
                        <text x="${x}%" y="${y + 12}" font-size="${this.fontSize}" \
                        font-family="Monospace"></text></g>`);

        let childXOffset = xOffset;
        for (let child of callNode.c) {
            childXOffset = this._renderSvgNodes(child, depth + 1, childXOffset);
        }
        return xOffset + width;
    }

    _renderUnzoomNode() {
        this.svg.append(`<rect id="zoom_rect_${this.id}" style="display:none;stroke:rgb(0,0,0);" \
        rx="10" ry="10" x="10" y="10" width="80" height="30" \
        fill="rgb(255,255,255)"/> \
         <text id="zoom_text_${this.id}" x="19" y="30" style="display:none">Zoom out</text>`);
    }

    _renderInfoNode() {
        this.svg.append(`<clipPath id="info_clip_path_${this.id}"> \
                         <rect style="stroke:rgb(0,0,0);" rx="10" ry="10" x="120" y="10" \
                         width="789" height="30" fill="rgb(255,255,255)"/> \
                         </clipPath> \
                         <rect style="stroke:rgb(0,0,0);" rx="10" ry="10" x="120" y="10" \
                         width="799" height="30" fill="rgb(255,255,255)"/> \
                         <text clip-path="url(#info_clip_path_${this.id})" \
                         id="info_text_${this.id}" x="128" y="30"></text>`);
    }

    _renderPercentNode() {
        this.svg.append(`<rect style="stroke:rgb(0,0,0);" rx="10" ry="10" \
                         x="934" y="10" width="150" height="30" \
                         fill="rgb(255,255,255)"/> \
                         <text id="percent_text_${this.id}" text-anchor="end" \
                         x="1074" y="30"></text>`);
    }

    _adjustTextSizeForNode(g) {
        let text = g.find('text');
        let width = parseFloat(g.find('rect').attr('width')) * this.svgWidth * 0.01;
        if (width < 28) {
            text.text('');
            return;
        }
        let methodName = g.find('title').text().split(' | ')[0];
        let numCharacters;
        for (numCharacters = methodName.length; numCharacters > 4; numCharacters--) {
            if (numCharacters * 7.5 <= width) {
                break;
            }
        }
        if (numCharacters == methodName.length) {
            text.text(methodName);
        } else {
            text.text(methodName.substring(0, numCharacters - 2) + '..');
        }
    }

    _adjustTextSize() {
        this.svgWidth = $(window).width();
        let thisObj = this;
        this.svg.find('g').each(function(_, g) {
            thisObj._adjustTextSizeForNode($(g));
        });
    }

    _enableZoom() {
        this.zoomStack = [this.svg.find('g').first().get(0)];
        this.svg.find('g').css('cursor', 'pointer').click(zoom);
        this.svg.find(`#zoom_rect_${this.id}`).css('cursor', 'pointer').click(unzoom);
        this.svg.find(`#zoom_text_${this.id}`).css('cursor', 'pointer').click(unzoom);

        let thisObj = this;
        function zoom() {
            thisObj.zoomStack.push(this);
            displayFromElement(this);
            thisObj.svg.find(`#zoom_rect_${thisObj.id}`).css('display', 'block');
            thisObj.svg.find(`#zoom_text_${thisObj.id}`).css('display', 'block');
        }

        function unzoom() {
            if (thisObj.zoomStack.length > 1) {
                thisObj.zoomStack.pop();
                displayFromElement(thisObj.zoomStack[thisObj.zoomStack.length - 1]);
                if (thisObj.zoomStack.length == 1) {
                    thisObj.svg.find(`#zoom_rect_${thisObj.id}`).css('display', 'none');
                    thisObj.svg.find(`#zoom_text_${thisObj.id}`).css('display', 'none');
                }
            }
        }

        function displayFromElement(g) {
            g = $(g);
            let clickedRect = g.find('rect');
            let clickedOriginX = parseFloat(clickedRect.attr('ox'));
            let clickedDepth = parseInt(clickedRect.attr('depth'));
            let clickedOriginWidth = parseFloat(clickedRect.attr('owidth'));
            let scaleFactor = 100.0 / clickedOriginWidth;
            thisObj.svg.find('g').each(function(_, g) {
                g = $(g);
                let text = g.find('text');
                let rect = g.find('rect');
                let depth = parseInt(rect.attr('depth'));
                let ox = parseFloat(rect.attr('ox'));
                let owidth = parseFloat(rect.attr('owidth'));
                if (depth < clickedDepth || ox < clickedOriginX - 1e-9 ||
                    ox + owidth > clickedOriginX + clickedOriginWidth + 1e-9) {
                    rect.css('display', 'none');
                    text.css('display', 'none');
                } else {
                    rect.css('display', 'block');
                    text.css('display', 'block');
                    let nx = (ox - clickedOriginX) * scaleFactor + '%';
                    let ny = thisObj._getYForDepth(depth - clickedDepth);
                    rect.attr('x', nx);
                    rect.attr('y', ny);
                    rect.attr('width', owidth * scaleFactor + '%');
                    text.attr('x', nx);
                    text.attr('y', ny + 12);
                    thisObj._adjustTextSizeForNode(g);
                }
            });
        }
    }

    _enableInfo() {
        this.selected = null;
        let thisObj = this;
        this.svg.find('g').on('mouseenter', function() {
            if (thisObj.selected) {
                thisObj.selected.css('stroke-width', '0');
            }
            // Mark current node.
            let g = $(this);
            thisObj.selected = g;
            g.css('stroke', 'black').css('stroke-width', '0.5');

            // Parse title.
            let title = g.find('title').text();
            let methodAndInfo = title.split(' | ');
            thisObj.svg.find(`#info_text_${thisObj.id}`).text(methodAndInfo[0]);

            // Parse percentage.
            // '/system/lib64/libhwbinder.so (4 events: 0.28%)'
            let regexp = /.* \(.*:\s+(.*)\)/g;
            let match = regexp.exec(methodAndInfo[1]);
            let percentage = '';
            if (match && match.length > 1) {
                percentage = match[1];
            }
            thisObj.svg.find(`#percent_text_${thisObj.id}`).text(percentage);
        });
    }

    _adjustTextSizeOnResize() {
        function throttle(callback) {
            let running = false;
            return function() {
                if (!running) {
                    running = true;
                    window.requestAnimationFrame(function () {
                        callback();
                        running = false;
                    });
                }
            };
        }
        $(window).resize(throttle(() => this._adjustTextSize()));
    }
}


class SourceFile {

    constructor(fileId) {
        this.path = getSourceFilePath(fileId);
        this.code = getSourceCode(fileId);
        this.showLines = {};  // map from line number to {eventCount, subtreeEventCount}.
        this.hasCount = false;
    }

    addLineRange(startLine, endLine) {
        for (let i = startLine; i <= endLine; ++i) {
            if (i in this.showLines || !(i in this.code)) {
                continue;
            }
            this.showLines[i] = {eventCount: 0, subtreeEventCount: 0};
        }
    }

    addLineCount(lineNumber, eventCount, subtreeEventCount) {
        let line = this.showLines[lineNumber];
        if (line) {
            line.eventCount += eventCount;
            line.subtreeEventCount += subtreeEventCount;
            this.hasCount = true;
        }
    }
}

// Return a list of SourceFile related to a function.
function collectSourceFilesForFunction(func) {
    if (!func.hasOwnProperty('s')) {
        return null;
    }
    let hitLines = func.s;
    let sourceFiles = {};  // map from sourceFileId to SourceFile.

    function getFile(fileId) {
        let file = sourceFiles[fileId];
        if (!file) {
            file = sourceFiles[fileId] = new SourceFile(fileId);
        }
        return file;
    }

    // Show lines for the function.
    let funcRange = getFuncSourceRange(func.g.f);
    if (funcRange) {
        let file = getFile(funcRange.fileId);
        file.addLineRange(funcRange.startLine);
    }

    // Show lines for hitLines.
    for (let hitLine of hitLines) {
        let file = getFile(hitLine.f);
        file.addLineRange(hitLine.l - 5, hitLine.l + 5);
        file.addLineCount(hitLine.l, hitLine.e, hitLine.s);
    }

    let result = [];
    // Show the source file containing the function before other source files.
    if (funcRange) {
        let file = getFile(funcRange.fileId);
        if (file.hasCount) {
            result.push(file);
        }
        delete sourceFiles[funcRange.fileId];
    }
    for (let fileId in sourceFiles) {
        let file = sourceFiles[fileId];
        if (file.hasCount) {
            result.push(file);
        }
    }
    return result.length > 0 ? result : null;
}

// Show annotated source code of a function.
class SourceCodeView {

    constructor(divContainer, sourceFiles) {
        this.div = $('<div>');
        this.div.appendTo(divContainer);
        this.sourceFiles = sourceFiles;
    }

    draw(sampleWeightFunction) {
        google.charts.setOnLoadCallback(() => this.realDraw(sampleWeightFunction));
    }

    realDraw(sampleWeightFunction) {
        this.div.empty();
        // For each file, draw a table of 'Line', 'Total', 'Self', 'Code'.
        for (let sourceFile of this.sourceFiles) {
            let rows = [];
            let lineNumbers = Object.keys(sourceFile.showLines);
            lineNumbers.sort((a, b) => a - b);
            for (let lineNumber of lineNumbers) {
                let code = getHtml('pre', {text: sourceFile.code[lineNumber]});
                let countInfo = sourceFile.showLines[lineNumber];
                let totalValue = '';
                let selfValue = '';
                if (countInfo.subtreeEventCount != 0) {
                    totalValue = sampleWeightFunction(countInfo.subtreeEventCount);
                    selfValue = sampleWeightFunction(countInfo.eventCount);
                }
                rows.push([lineNumber, totalValue, selfValue, code]);
            }

            let data = new google.visualization.DataTable();
            data.addColumn('string', 'Line');
            data.addColumn('string', 'Total');
            data.addColumn('string', 'Self');
            data.addColumn('string', 'Code');
            data.addRows(rows);
            for (let i = 0; i < rows.length; ++i) {
                data.setProperty(i, 0, 'className', 'colForLine');
                for (let j = 1; j <= 2; ++j) {
                    data.setProperty(i, j, 'className', 'colForCount');
                }
            }
            this.div.append(getHtml('pre', {text: sourceFile.path}));
            let wrapperDiv = $('<div>');
            wrapperDiv.appendTo(this.div);
            let table = new google.visualization.Table(wrapperDiv.get(0));
            table.draw(data, {
                width: '100%',
                sort: 'disable',
                frozenColumns: 3,
                allowHtml: true,
            });
        }
    }
}

// Return a list of disassembly related to a function.
function collectDisassemblyForFunction(func) {
    if (!func.hasOwnProperty('a')) {
        return null;
    }
    let hitAddrs = func.a;
    let rawCode = getFuncDisassembly(func.g.f);
    if (!rawCode) {
        return null;
    }

    // Annotate disassembly with event count information.
    let annotatedCode = [];
    let codeForLastAddr = null;
    let hitAddrPos = 0;
    let hasCount = false;

    function addEventCount(addr) {
        while (hitAddrPos < hitAddrs.length && hitAddrs[hitAddrPos].a < addr) {
            if (codeForLastAddr) {
                codeForLastAddr.eventCount += hitAddrs[hitAddrPos].e;
                codeForLastAddr.subtreeEventCount += hitAddrs[hitAddrPos].s;
                hasCount = true;
            }
            hitAddrPos++;
        }
    }

    for (let line of rawCode) {
        let code = line[0];
        let addr = line[1];

        addEventCount(addr);
        let item = {code: code, eventCount: 0, subtreeEventCount: 0};
        annotatedCode.push(item);
        // Objdump sets addr to 0 when a disassembly line is not associated with an addr.
        if (addr != 0) {
            codeForLastAddr = item;
        }
    }
    addEventCount(Number.MAX_VALUE);
    return hasCount ? annotatedCode : null;
}

// Show annotated disassembly of a function.
class DisassemblyView {

    constructor(divContainer, disassembly) {
        this.div = $('<div>');
        this.div.appendTo(divContainer);
        this.disassembly = disassembly;
    }

    draw(sampleWeightFunction) {
        google.charts.setOnLoadCallback(() => this.realDraw(sampleWeightFunction));
    }

    realDraw(sampleWeightFunction) {
        this.div.empty();
        // Draw a table of 'Total', 'Self', 'Code'.
        let rows = [];
        for (let line of this.disassembly) {
            let code = getHtml('pre', {text: line.code});
            let totalValue = '';
            let selfValue = '';
            if (line.subtreeEventCount != 0) {
                totalValue = sampleWeightFunction(line.subtreeEventCount);
                selfValue = sampleWeightFunction(line.eventCount);
            }
            rows.push([totalValue, selfValue, code]);
        }
        let data = new google.visualization.DataTable();
        data.addColumn('string', 'Total');
        data.addColumn('string', 'Self');
        data.addColumn('string', 'Code');
        data.addRows(rows);
        for (let i = 0; i < rows.length; ++i) {
            for (let j = 0; j < 2; ++j) {
                data.setProperty(i, j, 'className', 'colForCount');
            }
        }
        let wrapperDiv = $('<div>');
        wrapperDiv.appendTo(this.div);
        let table = new google.visualization.Table(wrapperDiv.get(0));
        table.draw(data, {
            width: '100%',
            sort: 'disable',
            frozenColumns: 2,
            allowHtml: true,
        });
    }
}


function initGlobalObjects() {
    gTabs = new TabManager($('div#report_content'));
    let recordData = $('#record_data').text();
    gRecordInfo = JSON.parse(recordData);
    gProcesses = gRecordInfo.processNames;
    gThreads = gRecordInfo.threadNames;
    gLibList = gRecordInfo.libList;
    gFunctionMap = gRecordInfo.functionMap;
    gSampleInfo = gRecordInfo.sampleInfo;
    gSourceFiles = gRecordInfo.sourceFiles;
}

function createTabs() {
    gTabs.addTab('Chart Statistics', new ChartStatTab());
    gTabs.addTab('Sample Table', new SampleTableTab());
    gTabs.addTab('Flamegraph', new FlameGraphTab());
    gTabs.draw();
}

let gTabs;
let gRecordInfo;
let gProcesses;
let gThreads;
let gLibList;
let gFunctionMap;
let gSampleInfo;
let gSourceFiles;

initGlobalObjects();
createTabs();

});