// one undo history per route workspace (keyed by its clock), in the order things were done: layout edits (layout.js:
// closing, splitting, docking panes, tabs) and the plots' views and curves (panes/plot/view.js). ctrl+z undoes the
// newest, whichever it is. An entry is {label, restore()}: restore() puts things back as the entry has them and returns
// the entry that undoes that. The CAN view's DBC edits keep a history of their own, with its buttons.
import {listeners} from "/dom.js";

const LIMIT = 100;

class History {
  past = [];
  future = [];
  changes = listeners();  // fn(label done or undone, "undo" | "redo" | null for a new entry)

  push(entry) {
    this.past.push(entry);
    if (this.past.length > LIMIT) this.past.shift();
    this.future.length = 0;
    this.changes.emit(entry.label, null);
  }

  undo() { return this.step(this.past, this.future, "undo"); }
  redo() { return this.step(this.future, this.past, "redo"); }

  step(from, to, how) {
    const entry = from.pop();
    if (!entry) return null;
    to.push(entry.restore());
    this.changes.emit(entry.label, how);
    return entry.label;
  }

  // drops the entries keep(entry) is false for (they'd undo nothing, and keep what they name alive)
  prune(keep) {
    for (const list of [this.past, this.future]) list.splice(0, Infinity, ...list.filter(keep));
  }
}

const histories = new WeakMap();  // clock → History, so it goes with its workspace
export const undoHistory = (clock) => histories.get(clock) || histories.set(clock, new History()).get(clock);
