# openpilot docs

This is the source for [docs.comma.ai](https://docs.comma.ai).
The site is updated on pushes to master by this [workflow](../.github/workflows/docs.yaml).

## Development

**1. Install the docs dependencies**
``` bash
pip install .[docs]
```

**2. Build the site**
``` bash
mkdocs build
```

**3. Host the new site locally**
``` bash
mkdocs serve
```

References:
* https://www.mkdocs.org/getting-started/
* https://github.com/ntno/mkdocs-terminal
