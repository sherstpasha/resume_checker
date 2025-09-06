from aiogram import Router
from .start import router as start_router
from .documents import router as documents_router

router = Router()
router.include_router(start_router)
router.include_router(documents_router)