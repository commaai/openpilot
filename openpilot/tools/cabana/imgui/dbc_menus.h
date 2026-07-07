#pragma once

#include <string>

struct GLFWwindow;
struct AppState;

// DBC file management for the imgui shell: File/Edit menu items, open/save
// flows, recent files, opendbc loading, fingerprint auto-load, window title.
// Parity target: tools/cabana/mainwin.{h,cc} (frozen Qt reference).

// Stores the GLFW window pointer used for the title bar and clipboard I/O.
// Call once, after the window is created and before the first frame.
void dbc_menus_init(GLFWwindow *window);

// Ensures at least one (possibly empty) DBC file is open -- mirrors
// MainWindow::startStream()'s "if (!dbc()->nonEmptyDBCCount()) newFile();".
// Call once, right after the stream starts.
void dbc_menus_ensure_dbc_open();

// Per-frame upkeep: wires the core event subscriptions on first call
// (DBCFileChanged, UndoStack::cleanChanged, AbstractStream::eventsMerged for
// fingerprint auto-load), refreshes the GLFW window title when dirty, and
// handles the global DBC keyboard shortcuts (Ctrl+N/O/S, Ctrl+Shift+S,
// Ctrl+Z, Ctrl+Shift+Z), gated on !io.WantTextInput. Call once per frame,
// before any menu is drawn.
void dbc_menus_update();

// Draws the DBC-owned File-menu items (New/Open/Manage/Recent/opendbc/
// clipboard/Save/Save As/Copy) -- call inside an open
// ImGui::BeginMenu("File") block. `app` is used only for has_stream(app),
// which gates the "Manage DBC Files" submenu exactly like Qt disables it for
// DummyStream.
void draw_dbc_file_menu_items(AppState &app);

// Draws the DBC-owned Edit-menu items (Undo/Redo/Command List) -- call
// inside an open ImGui::BeginMenu("Edit") block.
void draw_dbc_edit_menu_items();

// Draws every DBC-owned modal popup (file browser, error/info dialogs,
// unsaved-changes reminder). Call once per frame, outside any menu.
void draw_dbc_modals();

// Starts the close-confirmation flow (mirrors MainWindow::closeEvent's
// remindSaveChanges()). Call when AppState.request_close transitions to
// true; closes the GLFW window (via dbc_menus_init's pointer) once resolved
// -- immediately if there are no unsaved changes, otherwise after the user
// resolves the "Unsaved Changes" modal.
void dbc_menus_begin_close();

// Records `fn` as the most-recently-used DBC file (MAX_RECENT_FILES = 15,
// MRU order) and updates settings.last_dir -- mirrors
// MainWindow::updateRecentFiles(). Exposed so main.cc's CLI --dbc load
// path gets the same recent-files bookkeeping the Qt loadFile() always
// applies, regardless of caller.
void dbc_menus_note_recent_file(const std::string &fn);
