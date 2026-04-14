# openpilot docs

This is the source for [docs.comma.ai](https://docs.comma.ai).
The site is updated on pushes to master by this [workflow](../.github/workflows/docs.yaml).

## Development
NOTE: Those commands must be run in the root directory of openpilot, **not /docs**

**1. Install the docs tools**
``` bash
uv pip install --group docs
```

**2. Build the new site**
``` bash
mdbook build
```

**3. Run the new site locally**
``` bash
mdbook serve
```

References:
* https://rust-lang.github.io/mdBook/
* https://github.com/rust-lang/mdBook
