# Docs development

The `docs/` tree is the source for [docs.comma.ai](https://docs.comma.ai).
The site is updated on pushes to master by this [workflow](../.github/workflows/docs.yaml).

Those commands must be run in the root directory of openpilot, **not /docs**.
No extra dependencies — `docs/serve.py` is stdlib-only.

**1. Build the site**
``` bash
python docs/serve.py --build
```

**2. Run the site locally** (rebuilds on change)
``` bash
python docs/serve.py
```
