"""Direct notification endpoints (protected by X-API-Key and rate-limited)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import rate_limit_notifications, require_api_key
from app.integrations import IntegrationError, send_email, send_telegram
from app.models.schemas import NotificationResponse, SendEmailRequest, SendTelegramRequest

router = APIRouter(
    tags=["notifications"],
    dependencies=[Depends(require_api_key), Depends(rate_limit_notifications)],
)


@router.post("/send-email", response_model=NotificationResponse)
async def send_email_endpoint(req: SendEmailRequest):
    try:
        return await send_email(req.to, req.subject, req.body)
    except IntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from None


@router.post("/send-telegram", response_model=NotificationResponse)
async def send_telegram_endpoint(req: SendTelegramRequest):
    try:
        return await send_telegram(req.message)
    except IntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from None
