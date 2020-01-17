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

function flamegraphInit() {
    let flamegraph = document.getElementById('flamegraph_id');
    let svgs = flamegraph.getElementsByTagName('svg');
    for (let i = 0; i < svgs.length; ++i) {
        createZoomHistoryStack(svgs[i]);
        adjust_text_size(svgs[i]);
    }

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
    window.addEventListener('resize', throttle(function() {
        let flamegraph = document.getElementById('flamegraph_id');
        let svgs = flamegraph.getElementsByTagName('svg');
        for (let i = 0; i < svgs.length; ++i) {
            adjust_text_size(svgs[i]);
        }
    }));
}

// Create a stack add the root svg element in it.
function createZoomHistoryStack(svgElement) {
    svgElement.zoomStack = [svgElement.getElementById(svgElement.attributes['rootid'].value)];
}

function adjust_node_text_size(x, svgWidth) {
    let title = x.getElementsByTagName('title')[0];
    let text = x.getElementsByTagName('text')[0];
    let rect = x.getElementsByTagName('rect')[0];

    let width = parseFloat(rect.attributes['width'].value) * svgWidth * 0.01;

    // Don't even bother trying to find a best fit. The area is too small.
    if (width < 28) {
        text.textContent = '';
        return;
    }
    // Remove dso and #samples which are here only for mouseover purposes.
    let methodName = title.textContent.split(' | ')[0];

    let numCharacters;
    for (numCharacters = methodName.length; numCharacters > 4; numCharacters--) {
        // Avoid reflow by using hard-coded estimate instead of
        // text.getSubStringLength(0, numCharacters).
        if (numCharacters * 7.5 <= width) {
            break;
        }
    }

    if (numCharacters == methodName.length) {
        text.textContent = methodName;
        return;
    }

    text.textContent = methodName.substring(0, numCharacters-2) + '..';
}

function adjust_text_size(svgElement) {
    let svgWidth = window.innerWidth;
    let x = svgElement.getElementsByTagName('g');
    for (let i = 0; i < x.length; i++) {
        adjust_node_text_size(x[i], svgWidth);
    }
}

function zoom(e) {
    let svgElement = e.ownerSVGElement;
    let zoomStack = svgElement.zoomStack;
    zoomStack.push(e);
    displaySVGElement(svgElement);
    select(e);

    // Show zoom out button.
    svgElement.getElementById('zoom_rect').style.display = 'block';
    svgElement.getElementById('zoom_text').style.display = 'block';
}

function displaySVGElement(svgElement) {
    let zoomStack = svgElement.zoomStack;
    let e = zoomStack[zoomStack.length - 1];
    let clicked_rect = e.getElementsByTagName('rect')[0];
    let clicked_origin_x;
    let clicked_origin_y = clicked_rect.attributes['oy'].value;
    let clicked_origin_width;

    if (zoomStack.length == 1) {
        // Show all nodes when zoomStack only contains the root node.
        // This is needed to show flamegraph containing more than one node at the root level.
        clicked_origin_x = 0;
        clicked_origin_width = 100;
    } else {
        clicked_origin_x = clicked_rect.attributes['ox'].value;
        clicked_origin_width = clicked_rect.attributes['owidth'].value;
    }


    let svgBox = svgElement.getBoundingClientRect();
    let svgBoxHeight = svgBox.height;
    let svgBoxWidth = 100;
    let scaleFactor = svgBoxWidth / clicked_origin_width;

    let callsites = svgElement.getElementsByTagName('g');
    for (let i = 0; i < callsites.length; i++) {
        let text = callsites[i].getElementsByTagName('text')[0];
        let rect = callsites[i].getElementsByTagName('rect')[0];

        let rect_o_x = parseFloat(rect.attributes['ox'].value);
        let rect_o_y = parseFloat(rect.attributes['oy'].value);

        // Avoid multiple forced reflow by hiding nodes.
        if (rect_o_y > clicked_origin_y) {
            rect.style.display = 'none';
            text.style.display = 'none';
            continue;
        }
        rect.style.display = 'block';
        text.style.display = 'block';

        let newrec_x = rect.attributes['x'].value = (rect_o_x - clicked_origin_x) * scaleFactor +
                                                    '%';
        let newrec_y = rect.attributes['y'].value = rect_o_y + (svgBoxHeight - clicked_origin_y
                                                            - 17 - 2);

        text.attributes['y'].value = newrec_y + 12;
        text.attributes['x'].value = newrec_x;

        rect.attributes['width'].value = (rect.attributes['owidth'].value * scaleFactor) + '%';
    }

    adjust_text_size(svgElement);
}

function unzoom(e) {
    let svgOwner = e.ownerSVGElement;
    let stack = svgOwner.zoomStack;

    // Unhighlight whatever was selected.
    if (selected) {
        selected.classList.remove('s');
    }

    // Stack management: Never remove the last element which is the flamegraph root.
    if (stack.length > 1) {
        let previouslySelected = stack.pop();
        select(previouslySelected);
    }

    // Hide zoom out button.
    if (stack.length == 1) {
        svgOwner.getElementById('zoom_rect').style.display = 'none';
        svgOwner.getElementById('zoom_text').style.display = 'none';
    }

    displaySVGElement(svgOwner);
}

function search(e) {
    let term = prompt('Search for:', '');
    let callsites = e.ownerSVGElement.getElementsByTagName('g');

    if (!term) {
        for (let i = 0; i < callsites.length; i++) {
            let rect = callsites[i].getElementsByTagName('rect')[0];
            rect.attributes['fill'].value = rect.attributes['ofill'].value;
        }
        return;
    }

    for (let i = 0; i < callsites.length; i++) {
        let title = callsites[i].getElementsByTagName('title')[0];
        let rect = callsites[i].getElementsByTagName('rect')[0];
        if (title.textContent.indexOf(term) != -1) {
            rect.attributes['fill'].value = 'rgb(230,100,230)';
        } else {
            rect.attributes['fill'].value = rect.attributes['ofill'].value;
        }
    }
}

let selected;
document.addEventListener('keydown', (e) => {
    if (!selected) {
        return false;
    }

    let nav = selected.attributes['nav'].value.split(',');
    let navigation_index;
    switch (e.keyCode) {
    // case 38: // ARROW UP
    case 87: navigation_index = 0; break; // W

        // case 32 : // ARROW LEFT
    case 65: navigation_index = 1; break; // A

        // case 43: // ARROW DOWN
    case 68: navigation_index = 3; break; // S

        // case 39: // ARROW RIGHT
    case 83: navigation_index = 2; break; // D

    case 32: zoom(selected); return false; // SPACE

    case 8: // BACKSPACE
        unzoom(selected); return false;
    default: return true;
    }

    if (nav[navigation_index] == '0') {
        return false;
    }

    let target_element = selected.ownerSVGElement.getElementById(nav[navigation_index]);
    select(target_element);
    return false;
});

function select(e) {
    if (selected) {
        selected.classList.remove('s');
    }
    selected = e;
    selected.classList.add('s');

    // Update info bar
    let titleElement = selected.getElementsByTagName('title')[0];
    let text = titleElement.textContent;

    // Parse title
    let method_and_info = text.split(' | ');
    let methodName = method_and_info[0];
    let info = method_and_info[1];

    // Parse info
    // '/system/lib64/libhwbinder.so (4 events: 0.28%)'
    let regexp = /(.*) \((.*)\)/g;
    let match = regexp.exec(info);
    if (match.length > 2) {
        let percentage = match[2];
        // Write percentage
        let percentageTextElement = selected.ownerSVGElement.getElementById('percent_text');
        percentageTextElement.textContent = percentage;
    // console.log("'" + percentage + "'")
    }

    // Set fields
    let barTextElement = selected.ownerSVGElement.getElementById('info_text');
    barTextElement.textContent = methodName;
}