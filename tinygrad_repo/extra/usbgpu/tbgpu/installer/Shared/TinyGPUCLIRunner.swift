import Foundation
import SystemExtensions

enum TinyGPUCLIExit: Int32 { case ok = 0, usage = 2, failed = 3, needsApproval = 4 }
enum DextState { case unloaded, activating, needsApproval, activated }

final class TinyGPUCLIRunner: NSObject, OSSystemExtensionRequestDelegate {
  private let dextID: String
  private var done: ((TinyGPUCLIExit) -> Void)?
  private var isInstall = true

  init(_ dextID: String) { self.dextID = dextID }

  static func queryDextState(_ bundleID: String) -> DextState {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/systemextensionsctl")
    p.arguments = ["list"]
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = Pipe()
    guard (try? p.run()) != nil else { return .unloaded }
    p.waitUntilExit()

    guard let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8),
          let line = output.split(separator: "\n").first(where: { $0.contains(bundleID) }) else { return .unloaded }
    if line.contains("[activated enabled]") { return .activated }
    if line.contains("[activated waiting for user]") { return .needsApproval }
    return line.contains("terminated waiting to uninstall") ? .unloaded : .activating
  }

  private static let approvalHelp = """
    Please go to System Settings > Privacy & Security and allow the extension.

    If previously disabled: System Settings > General > Login Items & Extensions > Driver Extensions > Toggle TinyGPU ON

    """

  static func statusText(_ state: DextState) -> String {
    switch state {
    case .unloaded: return "Driver extension not installed.\n\n"
    case .activating: return "Extension is activating...\n\n"
    case .needsApproval: return "Extension awaiting approval.\n\n" + approvalHelp
    case .activated: return "Extension is ready! Run tinygrad to use your eGPU.\n\n"
    }
  }

  func run(args: [String], done: @escaping (TinyGPUCLIExit) -> Void) {
    self.done = done
    guard args.count > 1 else { return usage() }

    switch args[1] {
    case "status":
      print(Self.statusText(Self.queryDextState(dextID)))
      done(.ok)
    case "install":
      if Self.queryDextState(dextID) == .needsApproval { print(Self.statusText(.needsApproval)); return done(.needsApproval) }
      print("Installing TinyGPU driver extension...\nYou may need to approve in System Settings.\n")
      submitRequest(activate: true)
    case "uninstall":
      guard Self.queryDextState(dextID) != .unloaded else { print("Not installed.\n"); return done(.ok) }
      print("Uninstalling TinyGPU driver extension...\n")
      isInstall = false
      submitRequest(activate: false)
    case "server":
      guard args.count > 2 else { print("Error: server requires socket path\n"); return usage() }
      done(run_server(args[2]) == 0 ? .ok : .failed)
    case "help", "-h", "--help":
      usage(); done(.ok)
    default:
      print("Unknown command: \(args[1])\n"); usage()
    }
  }

  private func usage() {
    print("""
      Usage: TinyGPU <command>
        status     Show extension status
        install    Install the driver extension
        uninstall  Remove the driver extension
        server <path>  Start server on Unix socket
      """)
    done?(.usage)
  }

  private func submitRequest(activate: Bool) {
    let req = activate
      ? OSSystemExtensionRequest.activationRequest(forExtensionWithIdentifier: dextID, queue: .main)
      : OSSystemExtensionRequest.deactivationRequest(forExtensionWithIdentifier: dextID, queue: .main)
    req.delegate = self
    OSSystemExtensionManager.shared.submitRequest(req)
  }

  // MARK: - OSSystemExtensionRequestDelegate
  func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
    print("\nUser approval required!\n\n\(Self.approvalHelp)After approval, connect the gpu and use it with tinygrad.\n")
    done?(.needsApproval)
  }

  func request(_ request: OSSystemExtensionRequest, didFinishWithResult result: OSSystemExtensionRequest.Result) {
    switch result {
    case .completed: print("Driver extension \(isInstall ? "installed" : "uninstalled") successfully!\n")
    case .willCompleteAfterReboot: print("Will complete after reboot.\n")
    @unknown default: print("Completed: \(result)\n")
    }
    done?(.ok)
  }

  func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
    print("\nError: \(error.localizedDescription)\n")
    let code = (error as NSError).code
    if code == 4 { print("Missing entitlements. Rebuild with proper signing.\n") }
    else if code == 8 { print("Extension not found in app bundle.\n") }
    else if code == 9 { print("Extension disabled by user.\n\n\(Self.approvalHelp)") }
    done?(.failed)
  }

  func request(_ request: OSSystemExtensionRequest, actionForReplacingExtension existing: OSSystemExtensionProperties,
               withExtension ext: OSSystemExtensionProperties) -> OSSystemExtensionRequest.ReplacementAction {
    print("Updating v\(existing.bundleShortVersion) -> v\(ext.bundleShortVersion)...\n")
    return .replace
  }
}
