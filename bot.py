import asyncio
import os
import logging
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from db import init_db
from handlers import router as root_router

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"ENV {name} is required")
    return val

async def main():
    # Basic logging to see polling and errors in console
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    load_dotenv()

    bot_token = require_env("BOT_TOKEN")
    files_dir = os.getenv("FILES_DIR", "./storage")

    await init_db()

    os.makedirs(files_dir, exist_ok=True)

    bot = Bot(
        token=bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    dp.include_router(root_router)

    logging.getLogger(__name__).info("Starting bot polling...")
    await dp.start_polling(bot, allowed_updates=["message", "edited_message"])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
