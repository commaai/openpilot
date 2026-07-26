# Docs development

The `docs/` tree is the source for [docs.comma.ai](https://docs.comma.ai).
The site is updated on pushes to master by this [workflow](../.github/workflows/docs.yaml).

Those commands must be run in the root directory of openpilot, **not /docs**

**1. Install the docs dependencies**
``` bash
uv pip install .[docs]
```

**2. Build the new site**
``` bash
python scripts/docs.py --build
```

**3. Run the new site locally**
``` bash
python scripts/docs.py
```
