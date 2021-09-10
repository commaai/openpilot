---
name: Tuning
about: For openpilot tuning changes
title: ''
labels: 'tuning'
assignees: ''
---

**Description**

<!-- A description of what is wrong with the current tuning and how the PR adresses this. -->

**Verification**

<!-- To verify a longitudinal tuning PRs we need the plotjuggler plots of several scenarios of the desired speed/accel (first value from longitudinalPlan message)
vs measured speed/accel (from carState message), we need these plots for before and after the changes in the PR. For lateral tuning PRs we need plots of the 
desired steeringAngle vs actual steeringAngle. These scenarios don't need to be exactly followed, but do need to be broadly captured.

For longitudinal we need plots of before and after for all of the following scenarios:
* Maintaining speed at 25, 40, 65mph
* Driving up and down hills
* Accelerating from a stop
* Decelerating to a stop
* Following large changes in set speed
* Coming to a stop behind a lead car

For lateral we need plots of before and after for all of the following scenarios:
* Straight driving at ~25, ~45 and ~65mph
* Turns driving at ~25, ~45 and ~65mph



-->
