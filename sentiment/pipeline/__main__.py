"""
Allow ``python -m pipeline`` to start the application.

Modes
-----
- ``python -m pipeline``         → Desktop app (native window via pywebview)
- ``python -m pipeline --web``   → Browser-based Flask app (original mode)
"""
import sys

if "--web" in sys.argv:
    sys.argv.remove("--web")
    from pipeline.client.local_app import main
else:
    from pipeline.desktop_app import main

main()
