---
name: Tuning
about: For openpilot tuning changes
title: ''
labels: 'tuning'
assignees: ''
---

**Description**

<!-- A description of what is wrong with the current tuning and how the PR addresses this. -->

**Verification**

<!-- To verify tuning, capture the following scenarios (broadly, not exactly), with current tune and this tune.
Use the PlotJuggler tuning layout to compare planned versus actual behavior.

Run ./juggle.py <route> --layout layouts/tuning.xml , screenshot the full tab of interest, and paste into this PR.

Longitudinal:
* Maintaining speed at 25, 40, 65mph
* Driving up and down hills
* Accelerating from a stop
* Decelerating to a stop
* Following large changes in set speed
* Coming to a stop behind a lead car

Lateral:
* Straight driving at ~25, ~45 and ~65mph
* Turns driving at ~25, ~45 and ~65mph
-->