import SwiftUI

private let dextID = "org.tinygrad.tinygpu.edriver"

@main
struct TinyGPUApp: App {
  private static var runner: TinyGPUCLIRunner?  // prevent dealloc before callback
  @State private var text = ""
  @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

  init() {
    guard CommandLine.arguments.count > 1 else { return }
    Self.runner = TinyGPUCLIRunner(dextID)
    Self.runner?.run(args: CommandLine.arguments) { exit($0.rawValue) }
    dispatchMain()
  }

  var body: some Scene {
    WindowGroup("TinyGPU") {
      ScrollView {
        Text(text).font(.custom("Menlo", size: 11)).frame(maxWidth: .infinity, alignment: .leading).padding(8)
      }
      .frame(width: 500, height: 300).padding()
      .onAppear { setup() }
    }
    .commands { CommandGroup(replacing: .newItem) {} }
  }

  func setup() {
    let bundlePath = Bundle.main.bundlePath
    guard bundlePath.hasPrefix("/Applications/") else {
      var error: NSDictionary?
      NSAppleScript(source: "do shell script \"mv '\(bundlePath)' '/Applications/'\" with administrator privileges")?.executeAndReturnError(&error)
      text = error == nil ? "Moved! Please reopen from /Applications/\n" : "Move TinyGPU to /Applications first.\n"
      return
    }
    let state = TinyGPUCLIRunner.queryDextState(dextID)
    if state == .unloaded || state == .activating {
      Self.runner = TinyGPUCLIRunner(dextID)
      Self.runner?.run(args: ["", "install"]) { _ in }
    }
    text = "TinyGPU - Remote PCI Device Server\n\n" + TinyGPUCLIRunner.statusText(state)
  }
}

class AppDelegate: NSObject, NSApplicationDelegate {
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
