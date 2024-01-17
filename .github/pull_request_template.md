<!-- Please copy and paste the relevant template -->

<!-- ***** Template: Fingerprint *****

**Car**
Which car (make, model, year) this fingerprint is for

**Route**
A route with the fingerprint

-->

<!-- ***** Template: Car Bugfix *****

**Description**

A description of the bug and the fix. Also link the issue if it exists. 

**Verification**

Explain how you tested this bug fix. 

**Route**

Route: [a route with the bug fix]


-->

<!-- ***** Template: Bugfix *****

**Description**

A description of the bug and the fix. Also link the issue if it exists. 

**Verification**

Explain how you tested this bug fix. 


-->

<!-- ***** Template: Car Port *****

**Checklist**

- [ ] added entry to CarInfo in selfdrive/car/*/values.py and ran `selfdrive/car/docs.py` to generate new docs
- [ ] test route added to [routes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/car/tests/routes.py)
- [ ] route with openpilot:
- [ ] route with stock system:
- [ ] car harness used (if comma doesn't sell it, put N/A):


-->

<!-- ***** Template: Refactor *****

**Description**

A description of the refactor, including the goals it accomplishes. 

**Verification**

Explain how you tested the refactor for regressions. 


-->

<!-- ***** Template: Tuning *****

**Description**

A description of what is wrong with the current tuning and how the PR addresses this. 

**Verification**

To verify tuning, capture the following scenarios (broadly, not exactly), with current tune and this tune.
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

