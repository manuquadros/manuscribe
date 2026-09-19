# manuscribe

PDF parser used to convert PDFs for the D3 Annotation Hub

## Installation

```bash
pdm install
```

## Development

```bash
# Install all dependencies (including dev group)
pdm install

# Install pre-commit hooks (runs ruff + exports requirements.txt on commit)
pre-commit install

# Run tests
pdm run test

# Run tests with coverage
pdm run test-cov

# Lint
pdm run lint

# Format
pdm run fmt

# Type-check
pdm run typecheck

# Build docs
pdm run docs-build

# Live-reload docs server
pdm run docs-serve
```

## Documentation

Published at <https://manuquadros.github.io/manuscribe/> on every push to `main`
(`.github/workflows/docs.yml`). Locally, `pdm run docs-build` writes the same site
to `docs/_build/html/`; the build fails on any unresolved cross-reference.

## License

GPL-3.0-or-later
