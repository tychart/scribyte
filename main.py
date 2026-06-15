from app.main import app as app
from app.main import create_app as create_app
from app.main import main as _main

__all__ = ["app", "create_app", "main"]


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
