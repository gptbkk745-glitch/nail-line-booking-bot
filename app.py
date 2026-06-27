import json
import logging
import os
from datetime import datetime, timedelta
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from google import genai
from google.genai import types
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    MessagingApi,
    Configuration,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhook import WebhookParser
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from pydantic import BaseModel, Field, ValidationError


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nail-line-booking-bot")

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

BUSINESS_TZ = os.getenv("BUSINESS_TZ", "Asia/Bangkok")
WORK_START = os.getenv("WORK_START", "10:00")
WORK_END = os.getenv("WORK_END", "20:00")

TIMEZONE = ZoneInfo(BUSINESS_TZ)

SERVICES = {
    "gel_polish": {
        "display_name": "Gel Polish",
        "price_thb": 200,
        "duration_minutes": 60,
    },
    "acrylic": {
        "display_name": "Acrylic Nails",
        "price_thb": 500,
        "duration_minutes": 120,
    },
    "manicure": {
        "display_name": "Manicure",
        "price_thb": 150,
        "duration_minutes": 45,
    },
}

app = FastAPI(title="Nail Stall AI Receptionist Bot")

line_ready = (
    LINE_CHANNEL_SECRET
    and LINE_CHANNEL_ACCESS_TOKEN
    and LINE_CHANNEL_SECRET != "your_line_channel_secret"
    and LINE_CHANNEL_ACCESS_TOKEN != "your_line_channel_access_token"
)

gemini_ready = GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key"

if line_ready:
    line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    line_api_client = ApiClient(line_config)
    line_bot_api = MessagingApi(line_api_client)
    line_parser = WebhookParser(LINE_CHANNEL_SECRET)
else:
    line_bot_api = None
    line_parser = None

if gemini_ready:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None


SYSTEM_PROMPT = """
You are the AI receptionist for a small nail stall in Chinatown, Chiang Mai, Thailand.

Business identity:
- Business type: Nail stall / nail salon
- Location: Chinatown, Chiang Mai
- Customers: Thai locals, English-speaking tourists, and Chinese-speaking tourists
- Tone: warm, polite, clear, friendly, and concise
- The business has only 1 seat, so only one appointment can happen at a time.

Business hours:
- Open daily from 10:00 to 20:00 Thailand time.

Services and prices:
- Gel Polish: 200 THB
- Acrylic Nails: 500 THB
- Manicure: 150 THB

Your tasks:
1. Detect the user language automatically:
   - Thai => "th"
   - English => "en"
   - Chinese => "zh"
2. Understand the user intent:
   - "faq": price, location, service, opening hours, payment, etc.
   - "booking_request": user wants to book.
   - "reschedule_request": user wants to change an appointment.
   - "cancel_request": user wants to cancel.
   - "unknown": unclear.
3. Extract booking details if present.
4. Always respond in the same language as the user.
5. Do not confirm that an appointment is booked. The Python backend checks Google Calendar first.
6. customer_name and phone are optional. Do not ask for name or phone before checking availability.
7. If service, booking_date, or booking_time is missing, politely ask only for the missing required information.
8. missing_fields must contain ONLY these required fields when missing: service, booking_date, booking_time.
9. Never include customer_name or phone in missing_fields.
10. If service, booking_date, and booking_time are present, set missing_fields to [] and say you are checking availability.
11. Convert relative dates using the current Thailand date provided by backend.
12. Use 24-hour time format.
13. Output valid JSON only. No markdown. No extra text.

Return exactly this JSON structure:
{
  "intent": "faq | booking_request | reschedule_request | cancel_request | unknown",
  "language": "th | en | zh",
  "reply_message": "Message to send to the customer",
  "service": "gel_polish | acrylic | manicure | unknown | null",
  "service_display_name": "Human readable service name or null",
  "booking_date": "YYYY-MM-DD or null",
  "booking_time": "HH:MM or null",
  "duration_minutes": 60,
  "customer_name": "Customer name or null",
  "phone": "Phone number or null",
  "missing_fields": [],
  "confidence": 0.0
}
"""


class GeminiBookingResult(BaseModel):
    intent: Literal[
        "faq",
        "booking_request",
        "reschedule_request",
        "cancel_request",
        "unknown",
    ]
    language: Literal["th", "en", "zh"]
    reply_message: str

    service: Optional[Literal["gel_polish", "acrylic", "manicure", "unknown"]] = None
    service_display_name: Optional[str] = None
    booking_date: Optional[str] = None
    booking_time: Optional[str] = None
    duration_minutes: Optional[int] = None

    customer_name: Optional[str] = None
    phone: Optional[str] = None
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = 0.0


def now_thailand() -> datetime:
    return datetime.now(TIMEZONE)


def parse_hhmm(value: str):
    return datetime.strptime(value, "%H:%M").time()


def is_within_working_hours(start_dt: datetime, duration_minutes: int) -> bool:
    work_start = parse_hhmm(WORK_START)
    work_end = parse_hhmm(WORK_END)
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    return start_dt.time() >= work_start and end_dt.time() <= work_end


def build_start_datetime(booking_date: str, booking_time: str) -> datetime:
    naive_dt = datetime.strptime(f"{booking_date} {booking_time}", "%Y-%m-%d %H:%M")
    return naive_dt.replace(tzinfo=TIMEZONE)


def safe_json_loads(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


async def analyze_message_with_gemini(user_message: str, line_user_id: str) -> GeminiBookingResult:
    if not gemini_ready or gemini_client is None:
        return GeminiBookingResult(
            intent="unknown",
            language="en",
            reply_message="Gemini API key is not added yet. Please update GEMINI_API_KEY in the .env file.",
            service=None,
            service_display_name=None,
            booking_date=None,
            booking_time=None,
            duration_minutes=None,
            missing_fields=[],
            confidence=0.0,
        )

    today = now_thailand().strftime("%Y-%m-%d")

    user_prompt = f"""
Current Thailand date: {today}
Business timezone: {BUSINESS_TZ}
Working hours: {WORK_START}-{WORK_END}
LINE user ID: {line_user_id}

Customer message:
{user_message}
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=GeminiBookingResult,
            temperature=0.2,
        ),
    )

    try:
        if getattr(response, "parsed", None):
            return response.parsed

        data = safe_json_loads(response.text)
        return GeminiBookingResult.model_validate(data)

    except (json.JSONDecodeError, ValidationError) as exc:
        logger.exception("Gemini JSON parsing failed: %s", exc)

        return GeminiBookingResult(
            intent="unknown",
            language="en",
            reply_message="Sorry, I could not understand clearly. Could you please send your request again?",
            service=None,
            service_display_name=None,
            booking_date=None,
            booking_time=None,
            duration_minutes=None,
            missing_fields=[],
            confidence=0.0,
        )


def localized_outside_hours_reply(language: str) -> str:
    if language == "th":
        return "ขออภัยค่ะ เวลานี้อยู่นอกเวลาทำการของร้าน ร้านเปิด 10:00–20:00 ค่ะ กรุณาเลือกเวลาใหม่ได้ไหมคะ"
    if language == "zh":
        return "不好意思，这个时间不在营业时间内。我们每天 10:00–20:00 营业。请您选择其他时间。"
    return "Sorry, that time is outside our opening hours. We are open daily from 10:00 to 20:00. Please choose another time."


def localized_success_reply(result: GeminiBookingResult, start_dt: datetime) -> str:
    service_name = result.service_display_name or SERVICES[result.service]["display_name"]
    date_text = start_dt.strftime("%Y-%m-%d")
    time_text = start_dt.strftime("%H:%M")

    if result.language == "th":
        return f"จองสำเร็จค่ะ ✅\nบริการ: {service_name}\nวันที่: {date_text}\nเวลา: {time_text}\nแล้วพบกันนะคะ"
    if result.language == "zh":
        return f"预约成功 ✅\n服务：{service_name}\n日期：{date_text}\n时间：{time_text}\n期待见到您。"
    return f"Booking confirmed ✅\nService: {service_name}\nDate: {date_text}\nTime: {time_text}\nSee you soon!"


def localized_occupied_reply(language: str, alternatives: list[str]) -> str:
    alt_text = ", ".join(alternatives)

    if language == "th":
        return f"ขออภัยค่ะ เวลานั้นถูกจองแล้ว 🙏\nเวลาที่ว่างใกล้เคียงคือ: {alt_text}\nต้องการจองเวลาไหนคะ"
    if language == "zh":
        return f"不好意思，那个时间已经被预约了 🙏\n附近可预约时间：{alt_text}\n您想选择哪个时间？"
    return f"Sorry, that time is already booked 🙏\nNearby available times: {alt_text}\nWhich time would you prefer?"


async def check_calendar_availability(start_dt: datetime, duration_minutes: int) -> bool:
    """
    Dummy Google Calendar availability check.
    For testing, 14:00 is always occupied.
    """

    if start_dt.hour == 14:
        return False

    return True


async def create_calendar_booking(
    start_dt: datetime,
    duration_minutes: int,
    service: str,
    customer_name: Optional[str],
    phone: Optional[str],
    line_user_id: str,
) -> str:
    """
    Dummy Google Calendar booking creation.
    Later we will replace this with real Google Calendar API.
    """

    end_dt = start_dt + timedelta(minutes=duration_minutes)

    logger.info(
        "Dummy booking created: %s to %s, service=%s, customer=%s, phone=%s, user=%s",
        start_dt,
        end_dt,
        service,
        customer_name,
        phone,
        line_user_id,
    )

    return "dummy_calendar_event_id"


async def suggest_alternative_times(start_dt: datetime, duration_minutes: int) -> list[str]:
    alternatives = []
    candidate = start_dt + timedelta(minutes=30)

    while len(alternatives) < 3:
        if is_within_working_hours(candidate, duration_minutes):
            available = await check_calendar_availability(candidate, duration_minutes)
            if available:
                alternatives.append(candidate.strftime("%H:%M"))

        candidate += timedelta(minutes=30)

        if candidate.date() != start_dt.date():
            break

    return alternatives or ["10:00", "11:00", "12:00"]


async def process_customer_message(user_message: str, line_user_id: str) -> str:
    ai_result = await analyze_message_with_gemini(user_message, line_user_id)

    logger.info("AI result: %s", ai_result.model_dump())

    if ai_result.intent != "booking_request":
        return ai_result.reply_message

    required_missing = [
        field for field in ai_result.missing_fields
        if field in {"service", "booking_date", "booking_time"}
    ]

    if required_missing:
        return ai_result.reply_message

    if not ai_result.service or ai_result.service == "unknown":
        return ai_result.reply_message

    if ai_result.service not in SERVICES:
        return ai_result.reply_message

    if not ai_result.booking_date or not ai_result.booking_time:
        return ai_result.reply_message

    duration = SERVICES[ai_result.service]["duration_minutes"]

    try:
        start_dt = build_start_datetime(ai_result.booking_date, ai_result.booking_time)
    except ValueError:
        return ai_result.reply_message

    if not is_within_working_hours(start_dt, duration):
        return localized_outside_hours_reply(ai_result.language)

    available = await check_calendar_availability(start_dt, duration)

    if available:
        await create_calendar_booking(
            start_dt=start_dt,
            duration_minutes=duration,
            service=ai_result.service,
            customer_name=ai_result.customer_name,
            phone=ai_result.phone,
            line_user_id=line_user_id,
        )
        return localized_success_reply(ai_result, start_dt)

    alternatives = await suggest_alternative_times(start_dt, duration)
    return localized_occupied_reply(ai_result.language, alternatives)


@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "service": "Nail Stall AI Receptionist Bot",
        "timezone": BUSINESS_TZ,
        "working_hours": f"{WORK_START}-{WORK_END}",
        "line_ready": bool(line_ready),
        "gemini_ready": bool(gemini_ready),
    }


@app.post("/test-chat")
async def test_chat(payload: dict):
    message = payload.get("message", "")

    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    reply = await process_customer_message(message, "local_test_user")

    return {
        "input": message,
        "reply": reply,
    }


@app.post("/callback")
async def line_webhook(request: Request):
    if not line_ready or line_parser is None or line_bot_api is None:
        raise HTTPException(
            status_code=500,
            detail="LINE credentials are not configured. Please update .env file.",
        )

    signature = request.headers.get("X-Line-Signature")

    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    body_bytes = await request.body()
    body = body_bytes.decode("utf-8")

    try:
        events = line_parser.parse(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid LINE signature")
        raise HTTPException(status_code=400, detail="Invalid LINE signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue

        if not isinstance(event.message, TextMessageContent):
            continue

        user_message = event.message.text
        line_user_id = event.source.user_id if event.source and event.source.user_id else "unknown"

        try:
            reply_text = await process_customer_message(user_message, line_user_id)
        except Exception as exc:
            logger.exception("Error processing message: %s", exc)
            reply_text = "Sorry, something went wrong. Please try again."

        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )

    return {"status": "ok"}
