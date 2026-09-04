const path = require("path");

module.exports = {
  mode: "production",
  entry: "./tiktoken-export.js",
  output: {
    filename: "tiktoken.js",
    path: path.resolve(__dirname, "dist"),
    library: {
      type: "module"
    }
  },
  experiments: {
    outputModule: true,
    asyncWebAssembly: true
  },
  module: {
    rules: [
      {
        test: /\.wasm$/,
        type: "asset/resource",
      }
    ]
  }
};