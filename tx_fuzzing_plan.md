# openpilot #32425 - TX Fuzzing Implementation Plan

## Task
Add fuzz testing for tx messages to detect mismatches between openpilot and panda tx logic.

## Current State
- RX fuzzing exists: `test_panda_safety_carstate_fuzzy` in test_models.py
- Tests compare openpilot CarState with panda safety state
- Fuzzes CAN RX messages

## Implementation Plan

### 1. Create `test_panda_safety_tx_fuzzy` method
```python
def test_panda_safety_tx_fuzzy(self, data):
    """
    Fuzz openpilot's tx messages and check for panda safety mismatches.
    Focus on controlsAllowed transitions and tx safety hooks.
    """
```

### 2. Fuzzing Strategy
- Fuzz CarControl parameters (steer, gas, brake)
- Test controlsAllowed state transitions (enabled -> disabled, vice versa)
- Compare openpilot apply() output with panda safety_tx_hook
- Catch bugs like panda#1948

### 3. Test Cases
- Fuzz steer torque/angle values
- Fuzz gas/brake values
- Fuzz cruise control commands (resume, cancel)
- Test edge cases in controlsAllowed logic

### 4. Performance
- Use hypothesis for fuzzing
- MAX_EXAMPLES = 300 (same as rx fuzzing)
- Should complete in similar time

## Next Steps
1. Implement test_panda_safety_tx_fuzzy in test_models.py
2. Test with existing car models
3. Validate it catches known bugs
4. Submit PR

## Expected Bounty
$100-200
