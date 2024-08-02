# openpilot docs

This is the source for [docs.comma.ai](https://docs.comma.ai).
The site is updated on pushes to master by this [workflow](../.github/workflows/docs.yaml).

## Development

> [!NOTE]
> Run the following commands in the `docs/` directory, NOT root

Custom docs theme using `mkdocs` and `tailwindcss` with `daisyui`.

**1. Install the docs dependencies**
``` bash
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

**2. Run dev site locally**
``` bash
# tailwind if adjusting theme
tailwindcss-extra -i styles.css -o theme/assets/stylesheets/output.css --watch
# different terminal
mkdocs serve --dev-addr=0.0.0.0:8000
```

**3. Run prod site locally**
``` bash
tailwindcss-extra -i styles.css -o theme/assets/stylesheets/output.css --minify
mkdocs build
# install caddy https://caddyserver.com/docs/install
caddy file-server --root site/ --listen :8080
# open http://localhost:8080
```

References:
* https://www.mkdocs.org/getting-started/
* https://www.mkdocs.org/dev-guide/themes/
* https://tailwindcss.com/
* https://daisyui.com/
