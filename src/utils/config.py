import json
from pathlib import Path


class EnvConfig:
    """
    Loads and validates environment variables from a .env file, but i am thinking of switch to decouple.
    """

    def __init__(self, env_path: Path) -> None:
        self.env_path = env_path
        self.values = {}

        self._load_env()
        self._parse_types()

    def _load_env(self):
        if not self.env_path.exists():
            return

        with self.env_path.open("r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue  # ignore malformed

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                self.values[key] = value

    def _parse_types(self):
        """
        Converts strings from .env into real Python types:
        - bool
        - int
        - float
        """
        for k, v in self.values.items():
            # BOOL
            if v.lower() in ["true", "false"]:
                self.values[k] = v.lower() == "true"
                continue

            # INT
            if v.isdigit():
                self.values[k] = int(v)
                continue

            # FLOAT
            try:
                float_v = float(v)
                self.values[k] = float_v
                continue
            except ValueError:
                pass

            # FALLBACK = string
            self.values[k] = v

    def get(self, key: str, default=None):
        """
        Safe getter: returns default if missing.
        """
        return self.values.get(key, default)

    def require(self, key: str):
        """
        Strong getter: raises error if variable is missing.
        """
        if key not in self.values:
            raise KeyError(f"Missing required environment variable: {key}")
        return self.values[key]

    def apply_to(self, config_obj):
        """
        Injects values into a config class (AppConfig, MonitorConfig, etc.)
        Only overrides attributes that already exist.
        """
        for key, value in self.values.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)


class AppConfig:
    def __init__(self) -> None:
        self.LOG_LEVELS = ["INFO, DEBUG, WARNING, ERROR"]
        self.FALLBACK = ["SAFE", "STRICT", "PERMISSIVE"]

        self.LOG_LEVEL = self.LOG_LEVELS[0]
        self.FALLBACK_MODE = self.FALLBACK[0]

        self.MAX_REQUESTS_PER_MINUTE = 60
        self.GLOBAL_TIMEOUT = 10  # sec

        self.DEBUG = False
        self.ENABLE_RATE_LIMIT = True
        self.ENABLE_AI_FEATURES = True


class PathConfig:
    """
    class created to set up and centrilize all the paths for this project
    """

    def __init__(self) -> None:
        "root dir"
        self.BASE_DIR = Path(__file__).resolve().parent.parent

        "internal dirs"
        self.CONFIG_DIR = self.BASE_DIR / "config"
        self.LOG_DIR = self.BASE_DIR / "logs"
        self.CACHE_DIR = self.BASE_DIR / "cache"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODEL_DIR = self.BASE_DIR / "models"
        self.TEMP_DIR = self.BASE_DIR / "temp"

        self.ENV_FILES = self.BASE_DIR / ".env"
        self.SETTINGS_FILE = self.CONFIG_DIR / "settings.json"

    def __ensure_directories(self):
        dirs = [
            self.CONFIG_DIR,
            self.LOG_DIR,
            self.CACHE_DIR,
            self.DATA_DIR,
            self.MODEL_DIR,
            self.TEMP_DIR,
        ]

        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)


class MonitorConfig:
    """
    handles the behaviour of the sys. checking from time to time if its working
    """

    def __init__(self) -> None:
        self.SCAN_INTERVAL = 30
        self.URL_TIMEOUT = 5
        self.MAX_CONCURRENT_SCANS = 10

        self.RETRY_ATTEMPTS = 3
        self.RETRY_BACKOFF = 1.5

        self.ENABLE_MONITORING = True
        self.STRICT_MODE = False


class ClassifierConfig:
    """
    class that deals with ALL AI related stuff.
    """

    def __init__(self, paths: PathConfig) -> None:
        self.MODEL_DIR = paths.MODEL_DIR
        self.MODEL_NAME = "threat_classifier_v1"
        self.MODEL_VERSION = "1.0.0"

        "model path"
        self.MODEL_PATH = self.MODEL_DIR / f"{self.MODEL_NAME}.bin"

        "interference config"
        self.CONFIDENCE_THRESHOLD = 0.65
        self.CRITICAL_THRESHOLD = 0.85

        "pre processors"
        self.NORMALIZE_INPUT = True
        self.MAX_INPUT_LENGTH = 4096

        "performance tunning (sec)"
        self.BATCH_SIZE = 16
        self.MAX_CONCURRENT_INFERENCES = 4
        self.INFERENCE_TIMEOUT = 3

        "extra stuff"
        self.ENABLE_EXPLAINABILITY = False
        self.ENABLE_MODEL_CACHE = True


class AutomationConfig:
    """
    define and handles when the actions will be executed, and the values for them.
    """

    def __init__(self) -> None:
        self.ENABLE_AUTOMATION = False
        self.DRY_RUN = True

        "Thresholds"
        self.THREAT_SCORE_THRESHOLD = 0.70
        self.CRITICAL_SCORE_THRESHOLD = 0.90

        "auto actions"
        self.AUTO_BLOCK_ENABLED = False
        self.AUTO_REPORT_ENABLED = True
        self.AUTO_LOG_DETAIL = True

        "limit"
        self.MAX_AUTOMATION_ACTIONS_PER_HOUR = 20

        "Notifications (sec)"
        self.ENABLE_ALERTS = True
        self.ALERT_COOLDOWN = 180


class SecurityConfig:
    """
    handles security, at a basic level (well, the best that i can do)
    """

    def __init__(self) -> None:
        "hashing"
        self.HASH_ALGO = "sha256"
        self.SALT_ROUNDS = 12

        "tokens (sec)"
        self.ENABLE_TOKENS = True
        self.TOKEN_EXPIRATION = 3600
        self.REFRESH_TOKEN_EXPIRATION = 86400

        "Rate limiting (sec)"
        self.RATE_LIMIT_ENABLED = True
        self.RATE_LIMIT_REQUESTS = 100
        self.RATE_LIMIT_WINDOW = 60

        self.ALLOW_UNVERIFIED_SOURCES = False
        self.BLOCK_ON_CRITICAL = True

        self.ENABLE_AUDIT_LOGS = True
        self.AUDIT_LOG_RETENTION_DAYS = 30

        "extra stuff (sec)"
        self.MAX_LOGIN_ATTEMPTS = 5
        self.COOLDOWN_ON_FAILURE = 300


class Settings:
    "initialize and glue everything"

    def __init__(self) -> None:
        from .config import (
            AppConfig,
            AutomationConfig,
            ClassifierConfig,
            MonitorConfig,
            PathConfig,
            SecurityConfig,
        )

        self.paths = PathConfig()
        self.app = AppConfig()
        self.monitor = MonitorConfig()
        self.automation = AutomationConfig()
        self.security = SecurityConfig()

        self.classifiers = ClassifierConfig(paths=self.paths)


###################################################################################################


def load_settings() -> Settings:
    settings = Settings()

    # --- Load .env securely ---
    env = EnvConfig(settings.paths.ENV_FILES)

    env.apply_to(settings.app)
    env.apply_to(settings.monitor)
    env.apply_to(settings.automation)
    env.apply_to(settings.security)

    # --- Load settings.json ---
    config_path = settings.paths.SETTINGS_FILE
    if config_path.exists():
        with config_path.open() as f:
            raw = json.load(f)
    else:
        raw = {}

    # Apply JSON overrides
    for section_name, section_obj in [
        ("app", settings.app),
        ("monitor", settings.monitor),
        ("automation", settings.automation),
        ("security", settings.security),
    ]:
        if section_name in raw:
            for key, value in raw[section_name].items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    return settings
