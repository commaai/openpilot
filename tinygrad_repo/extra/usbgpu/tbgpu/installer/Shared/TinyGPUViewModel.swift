import Foundation
import os.log
import SystemExtensions

class TinyGPUDriverLoadingStateMachine {
  enum State { case unloaded, activating, needsApproval, activated, activationError }
}

class TinyGPUViewModel: NSObject {

	@Published private var state: TinyGPUDriverLoadingStateMachine.State = .unloaded
	
	override init() {
		super.init()
		refreshInitialDextState()
	}
	
	private func refreshInitialDextState() {
#if os(macOS)
		Task.detached { [dextIdentifier] in
		  let newState = Self.queryDextState(bundleID: dextIdentifier)
		  await MainActor.run { self.state = newState }
		}
#endif
	}

#if os(macOS)
	private static func queryDextState(bundleID: String) -> TinyGPUDriverLoadingStateMachine.State {
		let tool = "/usr/bin/systemextensionsctl"
		let p = Process()
		p.executableURL = URL(fileURLWithPath: tool)
		p.arguments = ["list"]

		let pipe = Pipe()
		p.standardOutput = pipe
		p.standardError = Pipe()

		do {
			try p.run()
			p.waitUntilExit()
			let data = pipe.fileHandleForReading.readDataToEndOfFile()
			guard let output = String(data: data, encoding: .utf8) else { return .unloaded }

			// Look for our bundle id line
			if let line = output.split(separator: "\n").first(where: { $0.contains(bundleID) }) {
				if line.contains("[activated enabled]") { return .activated }
				if line.contains("[activated waiting for user]") { return .needsApproval }
				if line.contains("terminated waiting to uninstall") { return .unloaded }
				return .activating
			} else {
				return .unloaded
			}
		} catch {
			return .unloaded
		}
	}
#endif

	private let dextIdentifier: String = "org.tinygrad.tinygpu.edriver"

	public var dextLoadingState: String {
		switch state {
		case .unloaded:
			return "TinyGPUDriver isn't loaded."
		case .activating:
			return "Activating TinyGPUDriver, please wait."
		case .needsApproval:
			return "Please follow the prompt to approve TinyGPUDriver."
		case .activated:
			return "TinyGPUDriver has been activated and is ready to use. You can close the installer."
		case .activationError:
			return "TinyGPUDriver has experienced an error during activation.\nPlease check the logs to find the error."
		}
	}
}

extension TinyGPUViewModel: ObservableObject {

#if os(macOS)
	func activateMyDext() {
		activateExtension(dextIdentifier)
	}
	
	func deactivateMyDext() {
		deactivateExtension(dextIdentifier)
	}

	func activateExtension(_ dextIdentifier: String) {

		let request = OSSystemExtensionRequest
			.activationRequest(forExtensionWithIdentifier: dextIdentifier,
							   queue: .main)
		request.delegate = self
		OSSystemExtensionManager.shared.submitRequest(request)

		self.state = .activating
	}
	
	func deactivateExtension(_ dextIdentifier: String) {

		let request = OSSystemExtensionRequest.deactivationRequest(forExtensionWithIdentifier: dextIdentifier, queue: .main)
		request.delegate = self
		OSSystemExtensionManager.shared.submitRequest(request)

		self.state = .unloaded
	}
#endif
}

#if os(macOS)
extension TinyGPUViewModel: OSSystemExtensionRequestDelegate {

	func request(
		_ request: OSSystemExtensionRequest,
		actionForReplacingExtension existing: OSSystemExtensionProperties,
		withExtension ext: OSSystemExtensionProperties) -> OSSystemExtensionRequest.ReplacementAction {

		var replacementAction: OSSystemExtensionRequest.ReplacementAction

		os_log("sysex actionForReplacingExtension: %@ %@", existing, ext)

		replacementAction = .replace
		self.state = .activating
		return replacementAction
	}

	func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
		os_log("sysex requestNeedsUserApproval")
		self.state = .needsApproval
	}

	func request(_ request: OSSystemExtensionRequest, didFinishWithResult result: OSSystemExtensionRequest.Result) {
		os_log("sysex didFinishWithResult: %d", result.rawValue)
		self.state = .activated
	}

	func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
		os_log("sysex didFailWithError: %@", error.localizedDescription)
		self.state = .activationError
	}
}
#endif
