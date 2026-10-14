#!/bin/sh
cd "/home/dutch/ai/Strata"
exec "/home/dutch/ai/Strata/.venv/bin/python" "/home/dutch/ai/Strata/serve/server.py" "--engine" "strata" "--config" "/home/dutch/ai/Strata/strata-iq3_s.json" "--port" "8080"
