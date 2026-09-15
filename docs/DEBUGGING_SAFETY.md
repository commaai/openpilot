# Debugging Panda Safety with Replay Drive + LLDB

## 1. Start the debugger in VS Code

* Select **Replay drive + Safety LLDB**.
* Enter the route or segment when prompted.
[<img src="https://github.com/user-attachments/assets/b0cc320a-083e-46a7-a9f8-ca775bbe5604">](https://github.com/user-attachments/assets/b0cc320a-083e-46a7-a9f8-ca775bbe5604)

## 2. Attach LLDB

* When prompted, pick the running **`replay_drive` process**.
* ⚠️ Attach quickly, or `replay_drive` will start consuming messages.

> [!TIP]
> Add a Python breakpoint at the start of `replay_drive.py` to pause execution and give yourself time to attach LLDB.

## 3. Set breakpoints in VS Code
Breakpoints can be set directly in `modes/xxx.h` (or any C file).
No extra LLDB commands are required — just place breakpoints in the editor.

## 4. Resume execution
Once attached, you can step through both Python (on the replay) and C safety code as CAN logs are replayed.

> [!NOTE]
> * Use short routes for quicker iteration.
> * Pause `replay_drive` early to avoid wasting log messages.

## Video

View a demo of this workflow on the PR that added it: https://github.com/commaai/openpilot/pull/36055#issue-3352911578