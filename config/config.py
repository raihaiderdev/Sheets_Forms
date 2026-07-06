import os
from datetime import timedelta


def _normalize_postgres_url(url: str) -> str:
    """Railway/Heroku-style URLs sometimes start with postgres:// —
    SQLAlchemy 1.4+ requires the postgresql:// scheme."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url


def _require_postgres_url(env_var: str) -> str:
    url = os.environ.get(env_var)
    if not url:
        raise RuntimeError(
            f"{env_var} is not set. This project requires PostgreSQL — "
            f"set {env_var} to a postgresql:// connection string in your .env file."
        )
    url = _normalize_postgres_url(url)
    if not url.startswith("postgresql://"):
        raise RuntimeError(
            f"{env_var} must be a postgresql:// connection string, got: {url.split('://')[0]}://..."
        )
    return url


class BaseConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-in-production")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)

    # Fixed office list — kept server-side so it can be validated on signup
    OFFICE_CHOICES = [
        "CEO",
        "DEO-SE",
        "DEO-M-EE",
        "DEO-W-EE",
        "DY.DEO-M-BWN",
        "DY.DEO-M-CTN",
        "DY.DEO-M-HND",
        "DY.DEO-M-MND",
        "DY.DEO-M-FTS",
        "DY.DEO-M-BWN-FEMALE",
        "DY.DEO-M-CTN-FEMALE",
        "DY.DEO-M-HND-FEMALE",
        "DY.DEO-M-MND-FEMALE",
        "DY.DEO-M-FTS-FEMALE",
    ]

    @classmethod
    def init_db_uri(cls):
        """Resolved lazily in create_app(), only for the config actually selected —
        so a missing TEST_DATABASE_URL doesn't break a normal production/development run."""
        raise NotImplementedError


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    SESSION_COOKIE_SECURE = False

    @classmethod
    def init_db_uri(cls):
        return _require_postgres_url("DATABASE_URL")


class ProductionConfig(BaseConfig):
    DEBUG = False
    SESSION_COOKIE_SECURE = True

    @classmethod
    def init_db_uri(cls):
        return _require_postgres_url("DATABASE_URL")


class TestingConfig(BaseConfig):
    TESTING = True
    WTF_CSRF_ENABLED = False

    @classmethod
    def init_db_uri(cls):
        return _require_postgres_url("TEST_DATABASE_URL")


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
