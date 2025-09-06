from aiogram import Router, types
from aiogram.filters import Command

router = Router()

@router.message(Command("start", "help"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! 👋\n"
        "Отправьте файл резюме (PDF/DOCX/TXT/DOC/RTF).\n"
        "Я проверю соответствие требованиям вакансии и дам ответ."
    )