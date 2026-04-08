import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    HTTP_BASE_URL: str = os.environ["HTTP_BASE_URL"]  # e.g. http://localhost:8002 or https://api.dalgo.org
    WEBSOCKET_BASE_URL: str = os.environ["WEBSOCKET_BASE_URL"]  # e.g. ws://localhost:8002 or wss://api.dalgo.org
    USERNAME: str = os.environ["USERNAME"]
    PASSWORD: str = os.environ["PASSWORD"]
    ORG_SLUG: str = os.environ["ORG_SLUG"]
    DASHBOARD_ID: int = int(os.environ["DASHBOARD_ID"])
    OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")

    @property
    def login_url(self) -> str:
        return f"{self.HTTP_BASE_URL}/api/v2/login/"

    @property
    def ws_url(self) -> str:
        return f"{self.WEBSOCKET_BASE_URL}/wss/dashboards/{self.DASHBOARD_ID}/chat/"
