import onnx
import pickle
import codecs

# Preprocessing: load the ONNX model
model_path = "models/driving_vision.onnx"
onnx_model = onnx.load(model_path)

# Print the entire model structure (can be verbose for large models)
# print(f"The model is:\n{onnx_model}")

# Check the model for validity (optional, but good practice)
try:
    onnx.checker.check_model(onnx_model)
    print("The model is valid!")
except onnx.checker.ValidationError as e:
    print(f"The model is invalid: {e}")

# Inspect inputs and outputs
print("\nModel Inputs:")
for input in onnx_model.graph.input:
    print(f"Name: {input.name}")
    # Get shape information
    shape = [dim.dim_value if dim.dim_value else "?" for dim in input.type.tensor_type.shape.dim]
    print(f"Shape: {shape}")
    # Get data type
    dtype = onnx.helper.tensor_dtype_to_string(input.type.tensor_type.elem_type)
    print(f"Data type: {dtype}\n")

print(f"metadata_props:")
for prop in onnx_model.metadata_props:
    print(f"[{prop.key}, {prop.value}]")
    if prop.key == "output_slices":
        output_slices = pickle.loads(codecs.decode(prop.value.encode(), "base64"))
        print(f"Unpickling output_slices:\n{output_slices}")

# You can similarly iterate over `model.graph.output` and `model.graph.node`
# to examine the outputs and individual operations/layers.
