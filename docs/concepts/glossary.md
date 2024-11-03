# openpilot glossary

* **issue**:  To formally report a situation. Before opening a new issue, check if one already exists for your problem. If so, add your case to it to increase priority and visibility.
* **route**: A file recording a driving session’s data, including the path, speed, GPS position, and driving assistance events. This information aids in analysis and system improvement.
* **routeID**:  The unique identifier for your route in comma connect (e.g.,`a0e5673b89222047/00000070--604b06efd9`).
* **segment**: Routes are divided into one-minute segments. For example, the third minute of the route mentioned above is:`a0e5673b89222047/00000070--604b06efd9/2`.
* **onroad**: The state openpilot enters when ignition is detected, displaying the camera feed on the screen.
* **offroad**:  The state openpilot stays in when no ignition is detected, showing the main page on the screen.
* **bookmark**: When reporting an issue, include the routeID and the segment where the problem occurred. To mark events, use the flag in the lower left sidebar in onroad mode; this will add a yellow marker in comma connect for easy reference.
* **panda**: A component that communicates using CAN, the standard electronic language for vehicle control. It can be embedded internally in devices like comma 2/3/3X or used externally on development devices like EONs.
* **comma 3X**: A device for running openpilot software, equipped with an internal Panda CAN-FD. Maintenance for openpilot software primarily targets this official device sold by comma.ai.
