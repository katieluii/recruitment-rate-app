from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ct_api_base: str = "https://clinicaltrials.gov/api/v2"
    ct_api_page_size: int = 1000
    ct_api_max_pages: int = 10

    # "api" or "postgres"
    data_source: str = "api"

    db_user: str = ""
    db_pass: str = ""
    db_host: str = "aact-db.ctti-clinicaltrials.org"
    db_port: int = 5432
    db_name: str = "aact"

    models_dir: Path = Path(__file__).parent.parent / "models" / "artifacts"
    frontend_dir: Path = Path(__file__).parent.parent / "frontend"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
