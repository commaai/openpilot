# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Imports
import numpy as np
from openwakeword.utils import AudioFeatures, re_arg

import onnxruntime as ort
import os
import functools
from collections import deque, defaultdict
from functools import partial
import time
from typing import List, DefaultDict, Dict


# Define main model class
class Model():
    """
    The main model class for openWakeWord. Creates a model object with the shared audio pre-processer
    and for arbitrarily many custom wake word/wake phrase models.
    """
    @re_arg({"wakeword_model_paths": "wakeword_models"})  # temporary handling of keyword argument change
    def __init__(
            self,
            wakeword_models: List[str] = [],
            class_mapping_dicts: List[dict] = [],
            enable_speex_noise_suppression: bool = False,
            **kwargs
            ):
        """Initialize the openWakeWord model object.

        Args:
            wakeword_models (List[str]): A list of paths of ONNX models to load into the openWakeWord model object.
                                              If not provided, will load all of the pre-trained models. Alternatively,
                                              just the names of pre-trained models can be provided to select a subset of models.
            class_mapping_dicts (List[dict]): A list of dictionaries with integer to string class mappings for
                                              each model in the `wakeword_models` arguments
                                              (e.g., {"0": "class_1", "1": "class_2"})
            enable_speex_noise_suppression (bool): Whether to use the noise suppresion from the SpeexDSP
                                                   library to pre-process all incoming audio. May increase
                                                   model performance when reasonably stationary background noise
                                                   is present in the environment where openWakeWord will be used.
                                                   It is very lightweight, so enabling it doesn't significantly
                                                   impact efficiency.
            inference_framework (str): The inference framework to use when for model prediction. Options are
                                       "tflite" or "onnx".
            kwargs (dict): Any other keyword arguments to pass the the preprocessor instance
        """
        wakeword_model_names = []
        if len(wakeword_models) >= 1:
            for ndx, i in enumerate(wakeword_models):
                if os.path.exists(i):
                    wakeword_model_names.append(os.path.splitext(os.path.basename(i))[0])

        # Create attributes to store models and metadata
        self.models = {}
        self.model_inputs = {}
        self.model_outputs = {}
        self.model_prediction_function = {}
        self.class_mapping = {}

        # Do imports for inference framework
        try:
            

            def onnx_predict(onnx_model, x):
                return onnx_model.run(None, {onnx_model.get_inputs()[0].name: x})

        except ImportError:
            raise ValueError("Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")

        for mdl_path, mdl_name in zip(wakeword_models, wakeword_model_names):
            # Load openwakeword models
            sessionOptions = ort.SessionOptions()
            sessionOptions.inter_op_num_threads = 1
            sessionOptions.intra_op_num_threads = 1

            self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions,
                                                            providers=["CPUExecutionProvider"])

            self.model_inputs[mdl_name] = self.models[mdl_name].get_inputs()[0].shape[1]
            self.model_outputs[mdl_name] = self.models[mdl_name].get_outputs()[0].shape[1]
            pred_function = functools.partial(onnx_predict, self.models[mdl_name])
            self.model_prediction_function[mdl_name] = pred_function

            if class_mapping_dicts and class_mapping_dicts[wakeword_models.index(mdl_path)].get(mdl_name, None):
                self.class_mapping[mdl_name] = class_mapping_dicts[wakeword_models.index(mdl_path)]
            else:
                self.class_mapping[mdl_name] = {str(i): str(i) for i in range(0, self.model_outputs[mdl_name])}

        # Create buffer to store frame predictions
        self.prediction_buffer: DefaultDict[str, deque] = defaultdict(partial(deque, maxlen=30))

        # Initialize SpeexDSP noise canceller
        if enable_speex_noise_suppression:
            from speexdsp_ns import NoiseSuppression
            self.speex_ns = NoiseSuppression.create(160, 16000)
        else:
            self.speex_ns = None

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(**kwargs)

    def get_parent_model_from_label(self, label):
        """Gets the parent model associated with a given prediction label"""
        parent_model = ""
        for mdl in self.class_mapping.keys():
            if label in self.class_mapping[mdl].values():
                parent_model = mdl
            elif label in self.class_mapping.keys() and label == mdl:
                parent_model = mdl

        return parent_model

    def reset(self):
        """Reset the prediction buffer"""
        self.prediction_buffer = defaultdict(partial(deque, maxlen=30))

    def predict(self, x: np.ndarray, patience: dict = {}, threshold: dict = {}, timing: bool = False):
        """Predict with all of the wakeword models on the input audio frames

        Args:
            x (ndarray): The input audio data to predict on with the models. Ideally should be multiples of 80 ms
                                (1280 samples), with longer lengths reducing overall CPU usage
                                but decreasing detection latency. Input audio with durations greater than or less
                                than 80 ms is also supported, though this will add a detection delay of up to 80 ms
                                as the appropriate number of samples are accumulated.
            patience (dict): How many consecutive frames (of 1280 samples or 80 ms) above the threshold that must
                             be observed before the current frame will be returned as non-zero.
                             Must be provided as an a dictionary where the keys are the
                             model names and the values are the number of frames. Can reduce false-positive
                             detections at the cost of a lower true-positive rate.
                             By default, this behavior is disabled.
            threshold (dict): The threshold values to use when the `patience` behavior is enabled.
                              Must be provided as an a dictionary where the keys are the
                              model names and the values are the thresholds.
            timing (bool): Whether to return timing information of the models. Can be useful to debug and
                           assess how efficiently models are running on the current hardware.

        Returns:
            dict: A dictionary of scores between 0 and 1 for each model, where 0 indicates no
                  wake-word/wake-phrase detected. If the `timing` argument is true, returns a
                  tuple of dicts containing model predictions and timing information, respectively.
        """
        # Check input data type
        if not isinstance(x, np.ndarray):
            raise ValueError(f"The input audio data (x) must by a Numpy array, instead received an object of type {type(x)}.")

        # Setup timing dict
        if timing:
            timing_dict: Dict[str, Dict] = {}
            timing_dict["models"] = {}
            feature_start = time.time()

        # Get audio features (optionally with Speex noise suppression)
        if self.speex_ns:
            n_prepared_samples = self.preprocessor(self._suppress_noise_with_speex(x))
        else:
            n_prepared_samples = self.preprocessor(x)

        if timing:
            timing_dict["models"]["preprocessor"] = time.time() - feature_start

        # Get predictions from model(s)
        predictions = {}
        for mdl in self.models.keys():
            if timing:
                model_start = time.time()

            # Run model to get predictions
            if n_prepared_samples > 1280:
                group_predictions = []
                for i in np.arange(n_prepared_samples//1280-1, -1, -1):
                    group_predictions.extend(
                        self.model_prediction_function[mdl](
                            self.preprocessor.get_features(
                                    self.model_inputs[mdl],
                                    start_ndx=-self.model_inputs[mdl] - i
                            )
                        )
                    )
                prediction = np.array(group_predictions).max(axis=0)[None, ]
            elif n_prepared_samples == 1280:
                prediction = self.model_prediction_function[mdl](
                    self.preprocessor.get_features(self.model_inputs[mdl])
                )
            elif n_prepared_samples < 1280:  # get previous prediction if there aren't enough samples
                if self.model_outputs[mdl] == 1:
                    if len(self.prediction_buffer[mdl]) > 0:
                        prediction = [[[self.prediction_buffer[mdl][-1]]]]
                    else:
                        prediction = [[[0]]]
                elif self.model_outputs[mdl] != 1:
                    n_classes = max([int(i) for i in self.class_mapping[mdl].keys()])
                    prediction = [[[0]*(n_classes+1)]]

            if self.model_outputs[mdl] == 1:
                predictions[mdl] = prediction[0][0][0]
            else:
                for int_label, cls in self.class_mapping[mdl].items():
                    predictions[cls] = prediction[0][0][int(int_label)]


            # Update prediction buffer, and zero predictions for first 5 frames during model initialization
            for cls in predictions.keys():
                if len(self.prediction_buffer[cls]) < 5:
                    predictions[cls] = 0.0
                self.prediction_buffer[cls].append(predictions[cls])

            # Get timing information
            if timing:
                timing_dict["models"][mdl] = time.time() - model_start

        if timing:
            return predictions, timing_dict
        else:
            return predictions

    def _suppress_noise_with_speex(self, x: np.ndarray, frame_size: int = 160):
        """
        Runs the input audio through the SpeexDSP noise suppression algorithm.
        Note that this function updates the state of the existing Speex noise
        suppression object, and isn't intended to be called externally.

        Args:
            x (ndarray): The 16-bit, 16khz audio to process. Must always be an
                         integer multiple of `frame_size`.
            frame_size (int): The frame size to use for the Speex Noise suppressor.
                              Must match the frame size specified during the
                              initialization of the noise suppressor.

        Returns:
            ndarray: The input audio with noise suppression applied
        """
        cleaned = []
        for i in range(0, x.shape[0], frame_size):
            chunk = x[i:i+frame_size]
            cleaned.append(self.speex_ns.process(chunk.tobytes()))

        cleaned_bytestring = b''.join(cleaned)
        cleaned_array = np.frombuffer(cleaned_bytestring, np.int16)
        return cleaned_array
