# Docs development

The `docs/` tree is the source for [docs.comma.ai](https://docs.comma.ai).
The site is updated on pushes to master by this [workflow](../.github/workflows/docs.yaml).

**1. Build the site**
``` bash
op docs --build
```

**2. Run the site locally** (rebuilds on change)
``` bash
op docs
```
