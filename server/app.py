from __future__ import annotations

import os

import uvicorn

from app import app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
