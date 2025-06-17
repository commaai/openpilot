import CoreML

// ANE?
let config = MLModelConfiguration()
config.computeUnits = .all

// CPU?
let opts = MLPredictionOptions()
opts.usesCPUOnly = false

class MNISTInput : MLFeatureProvider {
  var featureNames: Set<String> {
    get {
      return ["image", "image2"]
    }
  }
  func featureValue(for featureName: String) -> MLFeatureValue? {
    if (featureName == "image") {
      let tokenIDMultiArray = try? MLMultiArray(shape: [64], dataType: MLMultiArrayDataType.float32)
      tokenIDMultiArray?[0] = NSNumber(value: 1337)
      return MLFeatureValue(multiArray: tokenIDMultiArray!)
    }
    if (featureName == "image2") {
      let tokenIDMultiArray = try? MLMultiArray(shape: [64], dataType: MLMultiArrayDataType.float32)
      tokenIDMultiArray?[0] = NSNumber(value: 1337)
      return MLFeatureValue(multiArray: tokenIDMultiArray!)
    }
    return nil
  }
}

let compiledUrl = try MLModel.compileModel(at: URL(string: "test.mlmodel")!)
let model = try MLModel(contentsOf: compiledUrl, configuration: config)
let out = try model.prediction(from: MNISTInput(), options: opts)

print(out.featureValue(for: "probs") as Any)
