"""Enable ``python -m e2m`` as an alias for the ``e2m`` console script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
