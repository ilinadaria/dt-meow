import os
from typing import Tuple

import numpy as np

import torch
import subprocess
from ultralytics import YOLO

from dt_data_api import DataClient
from solution.integration_activity import MODEL_NAME, DT_TOKEN

from dt_device_utils import DeviceHardwareBrand, get_device_hardware_brand

from .constants import IMAGE_SIZE, ASSETS_DIR

JETSON_FP16 = True



def run(input, exception_on_failure=False):
    print(input)
    try:
        import subprocess

        program_output = subprocess.check_output(
            f"{input}", shell=True, universal_newlines=True, stderr=subprocess.STDOUT
        )
    except Exception as e:
        if exception_on_failure:
            print(e.output)
            raise e
        program_output = e.output
    print(program_output)
    return program_output.strip()


class Wrapper:
    def __init__(self, aido_eval=False):
        model_name = MODEL_NAME()

        models_path = os.path.join(ASSETS_DIR, "nn_models")
        weight_file_path = os.path.join(models_path, f"{model_name}.pt")

        print(f"Looking for local model at: {weight_file_path}")

        if not os.path.exists(weight_file_path):
            raise FileNotFoundError(
                f"Local model not found at: {weight_file_path}"
            )

        self.model = Model(weight_file_path)

        print("Loaded YOLO classes:")
        for i, name in self.model.model.names.items():
            print(i, ":", name)

    def predict(self, image: np.ndarray) -> Tuple[list, list, list]:
        return self.model.infer(image)

class Model:
    def __init__(self, weight_file_path: str):
        print("HELLO")
        super().__init__()

        print("HIIII")

        if not os.path.exists("/yolov5/hubconf.py"):
            print("Cloning YOLOv5...")
            subprocess.check_call("git clone -b v6.2 https://github.com/ultralytics/yolov5.git /yolov5", shell=True)

        model = torch.hub.load("/yolov5", "custom", path=weight_file_path, source="local")
        #model = YOLO(weight_file_path)
        model.eval()

        print(model)


        use_fp16 = (JETSON_FP16 and torch.cuda.is_available() and get_device_hardware_brand() == DeviceHardwareBrand.JETSON_NANO)

        if use_fp16:
            print("Using FP16")
            model = model.half()
        else:
            print("Using FP32")


        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()

        del model

    def infer(self, image: np.ndarray) -> Tuple[list, list, list]:
        print("HIII")
        det = self.model(image, size=IMAGE_SIZE)

        xyxy = det.xyxy[0]  # grabs det of first image (aka the only image we sent to the net)

        if xyxy.shape[0] > 0:
            conf = xyxy[:, -2]
            clas = xyxy[:, -1]
            xyxy = xyxy[:, :-2]

            return xyxy.tolist(), clas.tolist(), conf.tolist()
        return [], [], []