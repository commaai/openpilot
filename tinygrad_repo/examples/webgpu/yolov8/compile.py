from pathlib import Path
from examples.yolov8 import YOLOv8, get_weights_location
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from extra.export_model import export_model
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict

if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"
    yolo_variant = 'n'
    yolo_infer = YOLOv8(w=0.25, r=2.0, d=0.33, num_classes=80)
    state_dict = safe_load(get_weights_location(yolo_variant))
    load_state_dict(yolo_infer, state_dict)
    prg, inp_sizes, out_sizes, state = export_model(yolo_infer, Device.DEFAULT.lower(), Tensor.randn(1,3,416,416), model_name="yolov8")
    dirname = Path(__file__).parent
    safe_save(state, (dirname / "net.safetensors").as_posix())
    with open(dirname / f"net.js", "w") as text_file:
       text_file.write(prg)
