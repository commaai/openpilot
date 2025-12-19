import SwiftUI

struct TinyGPUView: View {
	@ObservedObject var viewModel = TinyGPUViewModel()

	var body: some View {
#if os(macOS)
		VStack(alignment: .center) {
			Text("TinyGPU Intsaller")
				.padding()
				.font(.title)
			Text(self.viewModel.dextLoadingState)
				.multilineTextAlignment(.center)
			HStack {
				Button(
					action: {
						self.viewModel.activateMyDext()
					}, label: {
						Text("Install extension")
					}
				)
			}
		}
		.frame(width: 500, height: 200, alignment: .center)
#endif
	}
}

struct TinyGPUView_Previews: PreviewProvider {
    static var previews: some View {
		TinyGPUView()
    }
}
