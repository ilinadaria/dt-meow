import os
from typing import Tuple

import numpy as np
from collections import deque
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
        
        self.sign_filter = SignCVaRFilter(IMAGE_SIZE)

    def predict(self, image: np.ndarray) -> Tuple[list, list, list]:
        return self.model.infer(image)
    
    def predict_and_filter(self, image: np.ndarray):
        bboxes, classes, scores = self.model.infer(image)
        trusted_classes, trusted_bboxes, trusted_scores = self.sign_filter.update(bboxes, classes, scores)

        return trusted_bboxes, trusted_classes, trusted_scores

class SignCVaRFilter:
    def __init__(self, image_size):
        self.image_size = image_size
        self.sign_classes = [0, 1, 3, 4, 5]

        self.history_len = 5
        self.alpha = 0.80
        self.min_history = 3

        self.histories = {
            cls: deque(maxlen=self.history_len)
            for cls in self.sign_classes
        }

        self.score_thresholds = {
            0: 0.60,
            1: 0.60,
            3: 0.45,
            4: 0.60,
            5: 0.50,
        }

        self.cvar_thresholds = {
            0: 0.25,
            1: 0.25,
            3: 0.15,
            4: 0.25,
            5: 0.15,
        }

    def _bbox_center_x(self, bbox):
        x1, _, x2, _ = bbox
        return 0.5 * (float(x1) + float(x2))

    def sign_strength(self, bbox, score):
        x1, y1, x2, y2 = bbox
        area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        norm_area = area / float(self.image_size * self.image_size)

        center_x = self._bbox_center_x(bbox)
        image_center_x = self.image_size / 2.0
        center_offset = abs(center_x - image_center_x) / max(image_center_x, 1.0)
        center_weight = max(0.0, 1.0 - center_offset)

        return float(score) * norm_area * center_weight * 10.0

    def compute_var_cvar(self, samples, alpha):
        if len(samples) == 0:
            return 0.0, 0.0

        arr = np.asarray(samples, dtype=np.float32)

        # lower-tail VaR: cutoff for the weakest (1 - alpha) fraction
        var_alpha = float(np.quantile(arr, 1.0 - alpha))
        tail = arr[arr <= var_alpha]

        if tail.size == 0:
            cvar_alpha = var_alpha
        else:
            cvar_alpha = float(np.mean(tail))

        return var_alpha, cvar_alpha

    def update(self, bboxes, classes, scores):
        frame_strength = {cls: 0.0 for cls in self.sign_classes}
        frame_bbox = {cls: None for cls in self.sign_classes}
        frame_score = {cls: 0.0 for cls in self.sign_classes}

        for cls, bbox, score in zip(classes, bboxes, scores):
            cls = int(cls)

            if score < self.score_thresholds[cls]:
                continue

            strength = self.sign_strength(bbox, score)
            if strength > frame_strength[cls]:
                frame_strength[cls] = strength
                frame_bbox[cls] = bbox
                frame_score[cls] = score

        for cls in self.sign_classes:
            self.histories[cls].append(frame_strength[cls])

        trusted_classes = []
        trusted_bboxes = []
        trusted_scores = []
        stats = {}

        for cls in self.sign_classes:
            history = list(self.histories[cls])
            var_alpha, cvar_alpha = self.compute_var_cvar(history, self.alpha)

            enough_history = len(history) >= self.min_history
            trusted = cvar_alpha > self.cvar_thresholds[cls]

            if enough_history and trusted:
                trusted_classes.append(cls)
                trusted_bboxes.append(frame_bbox[cls])
                trusted_scores.append(frame_score[cls])

        return trusted_classes, trusted_bboxes, trusted_scores

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