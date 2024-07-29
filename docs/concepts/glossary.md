# Glossary

## A

### ACC
**Adaptive Cruise Control:** Adjusts the speed of the vehicle to maintain a safe distance from vehicles ahead.

## B

### big model
**Big Model:** A new paradigm in model development that takes a bigger input frame. Full frame is 1164x874, little model is a 512x256 crop, big model is a 1024x512 crop, 4x bigger than little. Make box bigger, drive better. Useful for signs and lights.

## D

### Driving Model
**Driving Model (model):** The resulting neural network after Comma trains on driving data on their supercomputer. This file lives on the device, and processes inputs to give outputs relevant to driving. Usually takes the form of an ONNX file, or a THNEED file after compilation on device. This file does not change or get trained on device, only processes inputs and outputs. See the list of driving models for names and details of models over time.

## E

### End to end
**End to end (e2e):** End to end means the model reacts like a human would. It assesses the whole picture and acts accordingly. Unlike other approaches where things must be labeled by hand, end to end learns all the nuances of driving. A model is basically trained on what human drivers would do in a certain situation and attempts to reproduce that behavior.

## L

### longitudinal
**Longitudinal (long):** Refers to gas and brake control.

### lateral
**Lateral (lat):** Refers to steering control.

### lead
**Lead:** Selected radar point from your car's radar by the driving model of openpilot using the camera. Used for longitudinal MPC. Usual attributes: distance, speed, and acceleration.

## M

### Model predictive control
**Model Predictive Control (mpc):** An advanced method of process control that is used to control a process while satisfying a set of constraints. Used for longitudinal and lateral control.

<!-- Add more terms and definitions as needed -->
