import base64
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhook import WebhookParser
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from pydantic import BaseModel, Field, ValidationError


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-tenant-receptionist-bot")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TENANT_SLUG = os.getenv("DEFAULT_TENANT_SLUG", "nail-salon")
TENANTS_FILE = os.getenv("TENANTS_FILE", "tenants.json")

GOOGLE_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]


app = FastAPI(title="Multi-Tenant AI Receptionist Booking Bot")


gemini_ready = GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key"

if gemini_ready:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None


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

    service: Optional[str] = None
    service_display_name: Optional[str] = None
    booking_date: Optional[str] = None
    booking_time: Optional[str] = None
    duration_minutes: Optional[int] = None

    customer_name: Optional[str] = None
    phone: Optional[str] = None
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = 0.0


def load_tenants() -> dict[str, Any]:
    path = Path(TENANTS_FILE)

    if not path.exists():
        raise RuntimeError(f"Tenants file not found: {TENANTS_FILE}")

    return json.loads(path.read_text(encoding="utf-8-sig"))


TENANTS = load_tenants()


def get_tenant(tenant_slug: str) -> dict[str, Any]:
    tenant = TENANTS.get(tenant_slug)

    if not tenant:
        raise HTTPException(status_code=404, detail=f"Unknown tenant: {tenant_slug}")

    tenant = dict(tenant)
    tenant["slug"] = tenant_slug
    return tenant


def tenant_timezone(tenant: dict[str, Any]) -> str:
    return tenant.get("timezone", "Asia/Bangkok")


def tenant_zoneinfo(tenant: dict[str, Any]) -> ZoneInfo:
    return ZoneInfo(tenant_timezone(tenant))


def tenant_work_start(tenant: dict[str, Any]) -> str:
    return tenant.get("working_hours", {}).get("start", "10:00")


def tenant_work_end(tenant: dict[str, Any]) -> str:
    return tenant.get("working_hours", {}).get("end", "20:00")


def tenant_services(tenant: dict[str, Any]) -> dict[str, Any]:
    return tenant.get("services", {})


def env_from_tenant(tenant: dict[str, Any], env_key_name: str) -> str:
    env_name = tenant.get(env_key_name, "")
    if not env_name:
        return ""
    return os.getenv(env_name, "")


def tenant_line_ready(tenant: dict[str, Any]) -> bool:
    secret = env_from_tenant(tenant, "line_channel_secret_env")
    token = env_from_tenant(tenant, "line_channel_access_token_env")

    return bool(
        secret
        and token
        and secret != "your_line_channel_secret"
        and token != "your_line_channel_access_token"
    )


def tenant_calendar_id(tenant: dict[str, Any]) -> str:
    return env_from_tenant(tenant, "google_calendar_id_env")


def tenant_calendar_ready(tenant: dict[str, Any]) -> bool:
    calendar_id = tenant_calendar_id(tenant)
    b64_creds = env_from_tenant(tenant, "google_service_account_b64_env")
    json_creds = env_from_tenant(tenant, "google_service_account_json_env")

    return bool(calendar_id and (b64_creds or json_creds))


def get_line_clients(tenant: dict[str, Any]):
    if not tenant_line_ready(tenant):
        return None, None

    secret = env_from_tenant(tenant, "line_channel_secret_env")
    token = env_from_tenant(tenant, "line_channel_access_token_env")

    line_config = Configuration(access_token=token)
    line_api_client = ApiClient(line_config)
    line_bot_api = MessagingApi(line_api_client)
    line_parser = WebhookParser(secret)

    return line_parser, line_bot_api


def load_google_service_account_info(tenant: dict[str, Any]) -> dict:
    b64_creds = env_from_tenant(tenant, "google_service_account_b64_env")
    json_creds = env_from_tenant(tenant, "google_service_account_json_env")

    if b64_creds:
        decoded = base64.b64decode(b64_creds).decode("utf-8")
        return json.loads(decoded)

    if json_creds:
        return json.loads(json_creds)

    raise RuntimeError("Google Calendar service account credentials not configured.")


def get_calendar_service(tenant: dict[str, Any]):
    info = load_google_service_account_info(tenant)

    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=GOOGLE_CALENDAR_SCOPES,
    )

    return build(
        "calendar",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def format_services_for_prompt(tenant: dict[str, Any]) -> str:
    lines = []

    for key, service in tenant_services(tenant).items():
        display_name = service.get("display_name", key)
        duration = service.get("duration_minutes", 60)
        price = service.get("price_thb")

        if price is None:
            price_text = "price not listed"
        else:
            price_text = f"{price} THB"

        lines.append(
            f"- {key}: {display_name}, {price_text}, duration {duration} minutes"
        )

    return "\n".join(lines)


def build_system_prompt(tenant: dict[str, Any]) -> str:
    service_keys = list(tenant_services(tenant).keys())
    service_key_text = " | ".join(service_keys)

    business_rules = "\n".join(
        f"- {rule}" for rule in tenant.get("business_rules", [])
    )

    return f"""
You are the AI receptionist for {tenant.get("business_name", "the business")}.

Business identity:
- Business name: {tenant.get("business_name", "Business")}
- Business type: {tenant.get("business_type", "service business")}
- Location: {tenant.get("location", "not specified")}
- Tone: warm, polite, clear, friendly, and concise

Business hours:
- Open daily from {tenant_work_start(tenant)} to {tenant_work_end(tenant)} local time.
- Business timezone: {tenant_timezone(tenant)}

Services:
{format_services_for_prompt(tenant)}

Business rules:
{business_rules}

Your tasks:
1. Detect the user language automatically:
   - Thai => "th"
   - English => "en"
   - Chinese => "zh"
2. Understand the user intent:
   - "faq": price, location, service, opening hours, membership, rules, general information, etc.
   - "booking_request": user wants to book an appointment, room, service, or slot.
   - "reschedule_request": user wants to change an existing booking.
   - "cancel_request": user wants to cancel an existing booking.
   - "unknown": unclear.
3. Extract booking details if present.
4. Always respond in the same language as the user.
5. Do not confirm that a booking is complete. The Python backend checks Google Calendar first.
6. customer_name and phone are optional. Do not ask for name or phone before checking availability.
7. If service, booking_date, or booking_time is missing for a booking_request, politely ask only for the missing required information.
8. missing_fields must contain ONLY these required fields when missing: service, booking_date, booking_time.
9. Never include customer_name or phone in missing_fields.
10. If service, booking_date, and booking_time are present, set missing_fields to [] and say you are checking availability.
11. For cancel_request, extract booking_date and booking_time if the user gives them. Service is optional for cancellation.
12. Convert relative dates using the current local date provided by backend.
13. Use 24-hour time format.
14. Output valid JSON only. No markdown. No extra text.
15. For service, use one of these service keys when possible: {service_key_text}. Use "unknown" if unclear.

Return exactly this JSON structure:
{{
  "intent": "faq | booking_request | reschedule_request | cancel_request | unknown",
  "language": "th | en | zh",
  "reply_message": "Message to send to the customer",
  "service": "{service_key_text} | unknown | null",
  "service_display_name": "Human readable service name or null",
  "booking_date": "YYYY-MM-DD or null",
  "booking_time": "HH:MM or null",
  "duration_minutes": 60,
  "customer_name": "Customer name or null",
  "phone": "Phone number or null",
  "missing_fields": [],
  "confidence": 0.0
}}
"""


def now_for_tenant(tenant: dict[str, Any]) -> datetime:
    return datetime.now(tenant_zoneinfo(tenant))


def parse_hhmm(value: str):
    return datetime.strptime(value, "%H:%M").time()


def is_within_working_hours(
    start_dt: datetime,
    duration_minutes: int,
    tenant: dict[str, Any],
) -> bool:
    work_start = parse_hhmm(tenant_work_start(tenant))
    work_end = parse_hhmm(tenant_work_end(tenant))
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    return start_dt.time() >= work_start and end_dt.time() <= work_end


def build_start_datetime(
    booking_date: str,
    booking_time: str,
    tenant: dict[str, Any],
) -> datetime:
    naive_dt = datetime.strptime(f"{booking_date} {booking_time}", "%Y-%m-%d %H:%M")
    return naive_dt.replace(tzinfo=tenant_zoneinfo(tenant))


def safe_json_loads(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


async def analyze_message_with_gemini(
    user_message: str,
    line_user_id: str,
    tenant: dict[str, Any],
) -> GeminiBookingResult:
    if not gemini_ready or gemini_client is None:
        return GeminiBookingResult(
            intent="unknown",
            language="en",
            reply_message="Gemini API key is not added yet. Please update GEMINI_API_KEY.",
            service=None,
            service_display_name=None,
            booking_date=None,
            booking_time=None,
            duration_minutes=None,
            missing_fields=[],
            confidence=0.0,
        )

    today = now_for_tenant(tenant).strftime("%Y-%m-%d")

    user_prompt = f"""
Current local date: {today}
Business timezone: {tenant_timezone(tenant)}
Working hours: {tenant_work_start(tenant)}-{tenant_work_end(tenant)}
Tenant slug: {tenant["slug"]}
LINE user ID: {line_user_id}

Customer message:
{user_message}
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(tenant),
                response_mime_type="application/json",
                response_schema=GeminiBookingResult,
                temperature=0.2,
            ),
        )

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

    except Exception as exc:
        logger.exception("Gemini API request failed: %s", exc)

        # Simple language fallback without calling Gemini again
        if any("\u0E00" <= ch <= "\u0E7F" for ch in user_message):
            language = "th"
            reply = "ขออภัยค่ะ ระบบ AI กำลังใช้งานหนาแน่นชั่วคราว กรุณาลองใหม่อีกครั้งในสักครู่นะคะ"
        elif any("\u4E00" <= ch <= "\u9FFF" for ch in user_message):
            language = "zh"
            reply = "不好意思，AI 系统暂时繁忙。请稍后再试。"
        else:
            language = "en"
            reply = "Sorry, the AI service is temporarily busy. Please try again in a minute."

        return GeminiBookingResult(
            intent="unknown",
            language=language,
            reply_message=reply,
            service=None,
            service_display_name=None,
            booking_date=None,
            booking_time=None,
            duration_minutes=None,
            missing_fields=[],
            confidence=0.0,
        )


def localized_outside_hours_reply(language: str, tenant: dict[str, Any]) -> str:
    start = tenant_work_start(tenant)
    end = tenant_work_end(tenant)

    if language == "th":
        return f"ขออภัยค่ะ เวลานี้อยู่นอกเวลาทำการของเรา เปิดให้บริการ {start}–{end} ค่ะ กรุณาเลือกเวลาใหม่ได้ไหมคะ"
    if language == "zh":
        return f"不好意思，这个时间不在营业时间内。我们每天 {start}–{end} 营业。请您选择其他时间。"
    return f"Sorry, that time is outside our opening hours. We are open daily from {start} to {end}. Please choose another time."


def localized_success_reply(
    result: GeminiBookingResult,
    start_dt: datetime,
    tenant: dict[str, Any],
) -> str:
    services = tenant_services(tenant)
    service_data = services.get(result.service or "", {})
    service_name = result.service_display_name or service_data.get("display_name", result.service)

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


def localized_cancel_missing_reply(language: str) -> str:
    if language == "th":
        return "ได้ค่ะ กรุณาบอกวันที่และเวลาที่ต้องการยกเลิกการจองด้วยนะคะ"
    if language == "zh":
        return "可以的，请告诉我您想取消预约的日期和时间。"
    return "Sure. Please tell me the booking date and time you want to cancel."


def localized_cancel_success_reply(language: str, date_text: str, time_text: str) -> str:
    if language == "th":
        return f"ยกเลิกการจองเรียบร้อยแล้วค่ะ ✅\nวันที่: {date_text}\nเวลา: {time_text}"
    if language == "zh":
        return f"预约已取消 ✅\n日期：{date_text}\n时间：{time_text}"
    return f"Your booking has been cancelled ✅\nDate: {date_text}\nTime: {time_text}"


def localized_cancel_not_found_reply(language: str) -> str:
    if language == "th":
        return "ขออภัยค่ะ ไม่พบการจองของคุณในวันและเวลานี้ กรุณาตรวจสอบวันเวลาอีกครั้งนะคะ"
    if language == "zh":
        return "不好意思，没有找到您在这个日期和时间的预约。请再确认一下日期和时间。"
    return "Sorry, I could not find your booking at that date and time. Please check the date and time again."


async def check_calendar_availability(
    start_dt: datetime,
    duration_minutes: int,
    tenant: dict[str, Any],
) -> bool:
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    if not tenant_calendar_ready(tenant):
        logger.warning("Google Calendar not configured for tenant %s. Using dummy availability logic.", tenant["slug"])
        if start_dt.hour == 14:
            return False
        return True

    calendar_service = get_calendar_service(tenant)
    calendar_id = tenant_calendar_id(tenant)

    body = {
        "timeMin": start_dt.isoformat(),
        "timeMax": end_dt.isoformat(),
        "timeZone": tenant_timezone(tenant),
        "items": [{"id": calendar_id}],
    }

    result = calendar_service.freebusy().query(body=body).execute()

    busy_slots = (
        result
        .get("calendars", {})
        .get(calendar_id, {})
        .get("busy", [])
    )

    return len(busy_slots) == 0


async def create_calendar_booking(
    start_dt: datetime,
    duration_minutes: int,
    service: str,
    customer_name: Optional[str],
    phone: Optional[str],
    line_user_id: str,
    tenant: dict[str, Any],
) -> str:
    end_dt = start_dt + timedelta(minutes=duration_minutes)
    services = tenant_services(tenant)
    service_display = services[service]["display_name"]

    if not tenant_calendar_ready(tenant):
        logger.warning("Google Calendar not configured for tenant %s. Dummy booking created.", tenant["slug"])
        logger.info(
            "Dummy booking created: tenant=%s, %s to %s, service=%s, customer=%s, phone=%s, user=%s",
            tenant["slug"],
            start_dt,
            end_dt,
            service,
            customer_name,
            phone,
            line_user_id,
        )
        return "dummy_calendar_event_id"

    calendar_service = get_calendar_service(tenant)
    calendar_id = tenant_calendar_id(tenant)

    event_description = (
        f"Tenant: {tenant['slug']}\n"
        f"Business: {tenant.get('business_name', '')}\n"
        f"Service: {service_display}\n"
        f"Customer name: {customer_name or 'Not provided'}\n"
        f"Phone: {phone or 'Not provided'}\n"
        f"LINE user ID: {line_user_id}\n"
        "Created by Multi-Tenant AI Receptionist Bot"
    )

    event_body = {
        "summary": f"{tenant.get('business_name', 'Booking')} - {service_display}",
        "description": event_description,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": tenant_timezone(tenant),
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": tenant_timezone(tenant),
        },
    }

    created_event = calendar_service.events().insert(
        calendarId=calendar_id,
        body=event_body,
    ).execute()

    event_id = created_event.get("id", "")
    logger.info("Google Calendar booking created for tenant %s: %s", tenant["slug"], event_id)

    return event_id


async def cancel_calendar_booking(
    start_dt: datetime,
    line_user_id: str,
    tenant: dict[str, Any],
) -> bool:
    if not tenant_calendar_ready(tenant):
        logger.warning("Google Calendar not configured for tenant %s. Cannot cancel real booking.", tenant["slug"])
        return False

    calendar_service = get_calendar_service(tenant)
    calendar_id = tenant_calendar_id(tenant)

    day_start = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    events_result = calendar_service.events().list(
        calendarId=calendar_id,
        timeMin=day_start.isoformat(),
        timeMax=day_end.isoformat(),
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    target_prefix = start_dt.strftime("%Y-%m-%dT%H:%M")

    for event in events_result.get("items", []):
        event_start = event.get("start", {}).get("dateTime", "")
        description = event.get("description", "")

        if event_start.startswith(target_prefix) and line_user_id in description:
            event_id = event.get("id")
            calendar_service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
            ).execute()

            logger.info("Google Calendar booking cancelled for tenant %s: %s", tenant["slug"], event_id)
            return True

    return False


async def process_cancel_request(
    ai_result: GeminiBookingResult,
    line_user_id: str,
    tenant: dict[str, Any],
) -> str:
    if not ai_result.booking_date or not ai_result.booking_time:
        return localized_cancel_missing_reply(ai_result.language)

    try:
        start_dt = build_start_datetime(ai_result.booking_date, ai_result.booking_time, tenant)
    except ValueError:
        return localized_cancel_missing_reply(ai_result.language)

    cancelled = await cancel_calendar_booking(start_dt, line_user_id, tenant)

    if cancelled:
        return localized_cancel_success_reply(
            ai_result.language,
            start_dt.strftime("%Y-%m-%d"),
            start_dt.strftime("%H:%M"),
        )

    return localized_cancel_not_found_reply(ai_result.language)


async def suggest_alternative_times(
    start_dt: datetime,
    duration_minutes: int,
    tenant: dict[str, Any],
) -> list[str]:
    alternatives = []
    candidate = start_dt + timedelta(minutes=30)

    while len(alternatives) < 3:
        if is_within_working_hours(candidate, duration_minutes, tenant):
            available = await check_calendar_availability(candidate, duration_minutes, tenant)
            if available:
                alternatives.append(candidate.strftime("%H:%M"))

        candidate += timedelta(minutes=30)

        if candidate.date() != start_dt.date():
            break

    return alternatives or [tenant_work_start(tenant)]




def detect_simple_language(message: str) -> str:
    if any("\u0E00" <= ch <= "\u0E7F" for ch in message):
        return "th"
    if any("\u4E00" <= ch <= "\u9FFF" for ch in message):
        return "zh"
    return "en"


def direct_tenant_faq_reply(user_message: str, tenant: dict[str, Any]) -> Optional[str]:
    """
    Fast rule-based FAQ reply.
    This saves Gemini quota for common questions.
    """

    msg = user_message.lower().strip()
    language = detect_simple_language(user_message)
    services = tenant_services(tenant)

    # If message looks like an actual booking request, do not answer from FAQ.
    # Let Gemini extract service/date/time.
    booking_words = ["book", "booking", "reserve", "reservation", "appointment", "slot"]
    date_time_words = [
        "today", "tomorrow", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "am", "pm", ":"
    ]

    looks_like_booking = (
        any(word in msg for word in booking_words)
        and any(word in msg for word in date_time_words)
    )

    if looks_like_booking:
        return None

    # Tenant-specific FAQs from tenants.json
    for faq in tenant.get("faqs", []):
        keywords = faq.get("keywords", [])
        if any(keyword.lower() in msg for keyword in keywords):
            if language == "th" and faq.get("answer_th"):
                return faq["answer_th"]
            if language == "zh" and faq.get("answer_zh"):
                return faq["answer_zh"]
            return faq.get("answer_en")

    # Opening hours
    if any(word in msg for word in ["open", "opening", "hours", "time", "close", "closing"]):
        start = tenant_work_start(tenant)
        end = tenant_work_end(tenant)

        if language == "th":
            return f"เราเปิดให้บริการทุกวัน เวลา {start}–{end} ค่ะ"
        if language == "zh":
            return f"我们每天营业时间是 {start}–{end}。"
        return f"We are open daily from {start} to {end}."

    # Location
    if any(word in msg for word in ["location", "where", "address", "located"]):
        location = tenant.get("location", "not specified")

        if language == "th":
            return f"เราอยู่ที่ {location} ค่ะ"
        if language == "zh":
            return f"我们的位置是：{location}"
        return f"We are located at {location}."

    # Services list
    if any(word in msg for word in ["service", "services", "what do you offer", "menu"]):
        service_lines = []
        for service in services.values():
            name = service.get("display_name", "Service")
            duration = service.get("duration_minutes", 60)
            price = service.get("price_thb")

            if price is None:
                service_lines.append(f"- {name}: {duration} minutes")
            else:
                service_lines.append(f"- {name}: {price} THB, {duration} minutes")

        if language == "th":
            return "บริการของเรามี:\n" + "\n".join(service_lines)
        if language == "zh":
            return "我们的服务包括：\n" + "\n".join(service_lines)
        return "Our services are:\n" + "\n".join(service_lines)

    # Price for specific service
    if any(word in msg for word in ["price", "cost", "how much", "fee"]):
        for service in services.values():
            name = service.get("display_name", "")
            name_lower = name.lower()

            if name_lower and any(part in msg for part in name_lower.split()):
                price = service.get("price_thb")
                duration = service.get("duration_minutes", 60)

                if price is None:
                    if language == "th":
                        return f"{name} ใช้เวลาประมาณ {duration} นาทีค่ะ"
                    if language == "zh":
                        return f"{name} 大约需要 {duration} 分钟。"
                    return f"{name} takes about {duration} minutes."

                if language == "th":
                    return f"{name} ราคา {price} THB ใช้เวลาประมาณ {duration} นาทีค่ะ"
                if language == "zh":
                    return f"{name} 价格是 {price} THB，大约需要 {duration} 分钟。"
                return f"{name} costs {price} THB and takes about {duration} minutes."

        # General price list
        price_lines = []
        for service in services.values():
            name = service.get("display_name", "Service")
            price = service.get("price_thb")
            duration = service.get("duration_minutes", 60)

            if price is None:
                price_lines.append(f"- {name}: {duration} minutes")
            else:
                price_lines.append(f"- {name}: {price} THB, {duration} minutes")

        if language == "th":
            return "ราคาบริการ:\n" + "\n".join(price_lines)
        if language == "zh":
            return "价格如下：\n" + "\n".join(price_lines)
        return "Here are our prices:\n" + "\n".join(price_lines)

    return None



async def process_customer_message(
    user_message: str,
    line_user_id: str,
    tenant_slug: str,
) -> str:
    tenant = get_tenant(tenant_slug)
    services = tenant_services(tenant)

    direct_reply = direct_tenant_faq_reply(user_message, tenant)
    if direct_reply:
        return direct_reply

    ai_result = await analyze_message_with_gemini(user_message, line_user_id, tenant)

    logger.info("AI result for tenant %s: %s", tenant_slug, ai_result.model_dump())

    if ai_result.intent == "cancel_request":
        return await process_cancel_request(ai_result, line_user_id, tenant)

    if ai_result.intent != "booking_request":
        return ai_result.reply_message

    required_missing = [
        field for field in ai_result.missing_fields
        if field in {"service", "booking_date", "booking_time"}
    ]

    if required_missing:
        return ai_result.reply_message

    service_key = ai_result.service

    if not service_key or service_key == "unknown":
        return ai_result.reply_message

    if service_key not in services:
        return ai_result.reply_message

    if not services[service_key].get("bookable", True):
        return ai_result.reply_message

    if not ai_result.booking_date or not ai_result.booking_time:
        return ai_result.reply_message

    duration = int(services[service_key].get("duration_minutes", 60))

    try:
        start_dt = build_start_datetime(ai_result.booking_date, ai_result.booking_time, tenant)
    except ValueError:
        return ai_result.reply_message

    if not is_within_working_hours(start_dt, duration, tenant):
        return localized_outside_hours_reply(ai_result.language, tenant)

    available = await check_calendar_availability(start_dt, duration, tenant)

    if available:
        await create_calendar_booking(
            start_dt=start_dt,
            duration_minutes=duration,
            service=service_key,
            customer_name=ai_result.customer_name,
            phone=ai_result.phone,
            line_user_id=line_user_id,
            tenant=tenant,
        )
        return localized_success_reply(ai_result, start_dt, tenant)

    alternatives = await suggest_alternative_times(start_dt, duration, tenant)
    return localized_occupied_reply(ai_result.language, alternatives)


def tenant_public_status(tenant_slug: str) -> dict[str, Any]:
    tenant = get_tenant(tenant_slug)

    return {
        "tenant": tenant_slug,
        "business_name": tenant.get("business_name"),
        "business_type": tenant.get("business_type"),
        "timezone": tenant_timezone(tenant),
        "working_hours": f"{tenant_work_start(tenant)}-{tenant_work_end(tenant)}",
        "line_ready": tenant_line_ready(tenant),
        "calendar_ready": tenant_calendar_ready(tenant),
        "services": list(tenant_services(tenant).keys()),
    }


@app.get("/")
async def health_check():
    default_tenant = get_tenant(DEFAULT_TENANT_SLUG)

    return {
        "status": "ok",
        "service": "Multi-Tenant AI Receptionist Booking Bot",
        "mode": "multi-tenant",
        "default_tenant": DEFAULT_TENANT_SLUG,
        "tenants": list(TENANTS.keys()),
        "gemini_ready": bool(gemini_ready),
        "line_ready": tenant_line_ready(default_tenant),
        "calendar_ready": tenant_calendar_ready(default_tenant),
        "default_tenant_status": tenant_public_status(DEFAULT_TENANT_SLUG),
    }


@app.get("/tenants")
async def list_tenants():
    return {
        "default_tenant": DEFAULT_TENANT_SLUG,
        "gemini_ready": bool(gemini_ready),
        "tenants": [
            tenant_public_status(tenant_slug)
            for tenant_slug in TENANTS.keys()
        ],
    }


@app.post("/test-chat")
async def test_chat_default(payload: dict):
    return await test_chat_for_tenant(DEFAULT_TENANT_SLUG, payload)


@app.post("/test-chat/{tenant_slug}")
async def test_chat_for_tenant(tenant_slug: str, payload: dict):
    message = payload.get("message", "")

    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    reply = await process_customer_message(message, "local_test_user", tenant_slug)

    return {
        "tenant": tenant_slug,
        "input": message,
        "reply": reply,
    }


@app.post("/callback")
async def line_webhook_default(request: Request):
    return await line_webhook_for_tenant(DEFAULT_TENANT_SLUG, request)


@app.post("/callback/{tenant_slug}")
async def line_webhook_for_tenant(tenant_slug: str, request: Request):
    tenant = get_tenant(tenant_slug)
    line_parser, line_bot_api = get_line_clients(tenant)

    if not line_parser or not line_bot_api:
        raise HTTPException(
            status_code=500,
            detail=f"LINE credentials are not configured for tenant: {tenant_slug}",
        )

    signature = request.headers.get("X-Line-Signature")

    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    body_bytes = await request.body()
    body = body_bytes.decode("utf-8")

    try:
        events = line_parser.parse(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid LINE signature for tenant %s", tenant_slug)
        raise HTTPException(status_code=400, detail="Invalid LINE signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue

        if not isinstance(event.message, TextMessageContent):
            continue

        user_message = event.message.text
        line_user_id = event.source.user_id if event.source and event.source.user_id else "unknown"

        try:
            reply_text = await process_customer_message(user_message, line_user_id, tenant_slug)
        except Exception as exc:
            logger.exception("Error processing message for tenant %s: %s", tenant_slug, exc)
            reply_text = "Sorry, something went wrong. Please try again."

        try:
            line_bot_api.reply_message(
                reply_message_request=ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)],
                )
            )
        except Exception as exc:
            logger.exception("LINE reply_message failed for tenant %s, trying push fallback: %s", tenant_slug, exc)

            if line_user_id != "unknown":
                try:
                    line_bot_api.push_message(
                        push_message_request=PushMessageRequest(
                            to=line_user_id,
                            messages=[TextMessage(text=reply_text)],
                        )
                    )
                except Exception as push_exc:
                    logger.exception("LINE push_message fallback also failed for tenant %s: %s", tenant_slug, push_exc)

    return {
        "status": "ok",
        "tenant": tenant_slug,
    }\n\n# --- CORS for 3D web frontend ---
from fastapi.middleware.cors import CORSMiddleware

if not any(getattr(m.cls, "__name__", "") == "CORSMiddleware" for m in app.user_middleware):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
