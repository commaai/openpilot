#pragma once

// Shared hover/selection state between binary_view.cc and signal_view.cc
// (both owned by this workstream). Mirrors the BinaryView::highlight() /
// signalHovered / signalClicked <-> SignalView::signalHovered() /
// selectSignal() Qt signal/slot wiring in detailwidget.cc, but as plain free
// functions: both panels are invoked directly, in order, from
// draw_detail_panel() -> draw_msg_tab() every frame (binary view first, then
// the signal list), so there's no separate glue object or queued-signal hop
// needed -- a panel just calls the setter and the other panel reads the
// getter. State written by the *earlier*-drawn panel (binary view) is
// visible to the later one (signal list) the very same frame; state written
// the other way around lags by at most one frame -- the same one-frame
// latency Qt's queued connections have under a timer-driven event loop, so
// it isn't a user-visible regression at interactive frame rates.
//
// Storage lives in binary_view.cc (where this state originated pre-Phase-3);
// signal_view.cc only sees this header.

namespace cabana {
class Signal;
}

// Hover: the signal currently hovered in *either* panel -- darkens/brightens
// its bits in the binary view and its row in the signal list.
const cabana::Signal *hovered_signal();
void set_hovered_signal(const cabana::Signal *sig);

// Selection ("current signal"): set by clicking a signal's bits in the
// binary view (mirrors BinaryView::signalClicked -> SignalView::
// selectSignal(sig, /*expand=*/true), which toggles that row's expanded
// state and scrolls it into view) or by clicking a row in the signal list
// itself (selectSignal(sig, /*expand=*/false) equivalent -- select without
// forcing an expand toggle). Drives the row's "current" highlight in the
// signal list and -- a small value-add beyond Qt, which has no equivalent --
// an outline of its bits in the binary view.
//
// `from_binary_view` marks a selection made by a bit click so the signal
// list knows to toggle-expand + scroll to it; consume_selection_from_binary_view()
// clears the flag on read so that only happens once per click, not every frame.
const cabana::Signal *selected_signal();
void set_selected_signal(const cabana::Signal *sig, bool from_binary_view = false);
bool consume_selection_from_binary_view();
