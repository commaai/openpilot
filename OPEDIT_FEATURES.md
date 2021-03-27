# opEdit Features:
- You can misspell parameter names and opEdit should be able to figure out which parameter you want. Ex. `cmra off` would be parsed as: `camera_offset`
  - You can also still enter the corresponding parameter index while choosing parameters to edit
- Type `l` to toggle live tuning only mode, which only shows params that update within a second
- Shows a detailed description for each parameter once you choose it
- Parameter value type restriction. Ensures a user cannot save an unsupported value type for any parameters, breaking the fork
- Remembers which mode you were last in and initializes opEdit with that mode (live tuning or not)
- Case-insensitive boolean and NoneType entrance. Type `faLsE` to save `False (bool)`, etc

## ❗NEW❗:
- **Now, any parameter that isn't marked with `(static)` can update while driving and onroad. Only the parameters in the "live" view update within a second, the rest update within 10 seconds. If you enter a parameter, it will tell you its update behavior**
- As before, static parameters will need a reboot of the device, or ignition in some cases.

## To run the opEdit parameter manager:
```python
cd /data/openpilot
python op_edit.py
```

**OR**

```python
cd /data/openpilot
./op_edit.py
```
