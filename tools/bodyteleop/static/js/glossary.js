document.addEventListener('DOMContentLoaded', function() {
  const terms = {
    "big model": "A new paradigm in model development that takes a bigger input frame. Full frame is 1164x874, little model is a 512x256 crop, big model is a 1024x512 crop, 4x bigger than little. Make box bigger, drive better. Useful for signs and lights.",
    "Driving Model": "The resulting neural network after Comma trains on driving data on their supercomputer. This file lives on the device, and processes inputs to give outputs relevant to driving. Usually takes the form of an ONNX file, or a THNEED file after compilation on device. This file does not change or get trained on device, only processes inputs and outputs. See the list of driving models for names and details of models over time.",
    "End to end": "End to end means the model reacts like a human would. It assesses the whole picture and acts accordingly. Unlike other approaches where things must be labeled by hand, end to end learns all the nuances of driving. A model is basically trained on what human drivers would do in a certain situation and attempts to reproduce that behavior.",
    "longitudinal": "Refers to gas and brake control.",
    "lateral": "Refers to steering control.",
    "Model predictive control": "An advanced method of process control that is used to control a process while satisfying a set of constraints. Used for longitudinal and lateral control.",
    "lead": "Selected radar point from your car's radar by the driving model of openpilot using the camera. Used for longitudinal MPC. Usual attributes: distance, speed, and acceleration."
  };

  document.querySelectorAll('.glossary-term').forEach(function(termElement) {
    termElement.addEventListener('mouseenter', function() {
      const term = termElement.getAttribute('data-term');
      const definition = terms[term];
      let tooltip = document.createElement('div');
      tooltip.classList.add('tooltip');
      tooltip.innerText = definition;
      document.body.appendChild(tooltip);
      let rect = termElement.getBoundingClientRect();
      tooltip.style.left = `${rect.left}px`;
      tooltip.style.top = `${rect.bottom}px`;
    });

    termElement.addEventListener('mouseleave', function() {
      document.querySelector('.tooltip').remove();
    });
  });
});
