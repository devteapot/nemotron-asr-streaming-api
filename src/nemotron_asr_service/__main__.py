from __future__ import annotations

import uvicorn

from nemotron_asr_service.app import create_app
from nemotron_asr_service.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
