"""Cross-platform production entrypoint using waitress.

Usage:
    python -m thoraxvision.serve

For gunicorn on Linux/macOS:
    gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 thoraxvision.app:app
"""

import os

from waitress import serve

from .app import app


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    threads = int(os.environ.get("WEB_THREADS", "4"))
    print(f"Serving on http://{host}:{port} (threads={threads})")
    serve(app, host=host, port=port, threads=threads)


if __name__ == "__main__":
    main()
