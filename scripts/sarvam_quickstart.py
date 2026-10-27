"""One Sarvam AI chat completion call. Reads SARVAM_API_KEY from the environment / .env (never printed)."""
import os

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()
api_key = os.environ.get("SARVAM_API_KEY")
if not api_key:
    raise SystemExit("SARVAM_API_KEY is not set — add it to .env")

client = SarvamAI(api_subscription_key=api_key)
response = client.chat.completions(
    model="sarvam-105b-conversations",
    messages=[{"role": "user", "content": "In one sentence, what is a waitlisted train ticket?"}],
)
print(response.choices[0].message.content)
