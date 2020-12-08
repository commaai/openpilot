# opEdit Features:
- You can misspell parameter names and opEdit should be able to figure out which parameter you want. Ex. `cmra off` would be parsed as: `camera_offset`
  - You can also still enter the corresponding parameter index while choosing parameters to edit
- Type `a` to add a parameter, `d` to delete a parameter, or `l` to toggle live tuning only mode
- Shows a detailed description for each parameter once you choose it
- Parameter value type restriction. Ensures a user cannot save an unsupported value type for any parameters, breaking the fork
- Remembers which mode you were last in and initializes opEdit with that mode (live tuning or not)
- Case insensitive boolean and NoneType entrance. Type `faLsE` to save `False (bool)`, etc
- **Parameters marked with `(live!)` will have updates take affect within 3 seconds while driving! All other params will require a reboot of your EON/C2 to take effect.**

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
