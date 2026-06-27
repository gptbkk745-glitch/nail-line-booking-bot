# Nail LINE Booking Bot

AI Receptionist and Booking Bot for Nail Stall in Chinatown, Chiang Mai.

Environment name:

nailbot_env

Run server:

.\nailbot_env\Scripts\python.exe -m uvicorn app:app --reload --port 8000

Run ngrok in another PowerShell window:

ngrok http 8000

LINE webhook URL:

https://YOUR-NGROK-URL/callback
