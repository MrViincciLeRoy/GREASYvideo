import json
import os
import re
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
import time


@dataclass
class APIKeyStatus:
    key: str
    provider: str

    requests_remaining: Optional[int] = None
    requests_limit: Optional[int] = None
    tokens_remaining: Optional[int] = None
    tokens_limit: Optional[int] = None
    reset_time: Optional[str] = None

    requests_made: int = 0
    daily_limit: int = 14400
    last_reset: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True
    error_count: int = 0

    def update_from_headers(self, headers: Dict):
        if 'x-ratelimit-remaining-requests' in headers:
            self.requests_remaining = int(headers['x-ratelimit-remaining-requests'])
        if 'x-ratelimit-limit-requests' in headers:
            self.requests_limit = int(headers['x-ratelimit-limit-requests'])
        if 'x-ratelimit-remaining-tokens' in headers:
            self.tokens_remaining = int(headers['x-ratelimit-remaining-tokens'])
        if 'x-ratelimit-limit-tokens' in headers:
            self.tokens_limit = int(headers['x-ratelimit-limit-tokens'])
        if 'x-ratelimit-reset-requests' in headers:
            self.reset_time = headers['x-ratelimit-reset-requests']
        self.last_used = datetime.now().isoformat()

    def is_exhausted(self) -> bool:
        if self.requests_remaining is not None:
            return self.requests_remaining <= 0
        return self.requests_made >= self.daily_limit

    def get_usage_percent(self) -> float:
        if self.requests_limit and self.requests_remaining is not None:
            used = self.requests_limit - self.requests_remaining
            return (used / self.requests_limit * 100) if self.requests_limit > 0 else 0
        return (self.requests_made / self.daily_limit * 100) if self.daily_limit > 0 else 0

    def get_remaining_requests(self) -> int:
        if self.requests_remaining is not None:
            return self.requests_remaining
        return max(0, self.daily_limit - self.requests_made)

    def should_reset(self) -> bool:
        return datetime.now() - datetime.fromisoformat(self.last_reset) > timedelta(days=1)

    def reset_if_needed(self):
        if self.should_reset():
            self.requests_made = 0
            self.last_reset = datetime.now().isoformat()
            self.is_active = True

    def increment_usage(self):
        self.requests_made += 1
        self.last_used = datetime.now().isoformat()

    def record_error(self):
        self.error_count += 1
        if self.error_count >= 5:
            self.is_active = False


def _parse_retry_wait(error_str: str) -> float:
    """
    Pull the suggested wait time from a Groq 429 error message.
    e.g. 'Please try again in 6.164s' -> 6.164
    Falls back to 10s if not found.
    """
    match = re.search(r'try again in ([\d.]+)s', error_str)
    if match:
        return float(match.group(1)) + 1.0  # add 1s buffer
    return 10.0


@dataclass
class APIKeyManager:
    groq_keys: List[str] = field(default_factory=list)
    huggingface_token: Optional[str] = None

    groq_daily_limit: int = 14400
    groq_rpm_limit: int = 30

    key_statuses: Dict[str, APIKeyStatus] = field(default_factory=dict)
    current_provider: str = "groq"
    current_key_index: int = 0

    last_request_time: float = 0.0
    min_request_interval: float = 2.0

    state_file: str = "api_key_state.json"

    def __post_init__(self):
        self._lock = threading.Lock()

        if os.path.exists(self.state_file):
            self.load_state()
        else:
            for key in self.groq_keys:
                if key not in self.key_statuses:
                    self.key_statuses[key] = APIKeyStatus(
                        key=key, provider="groq", daily_limit=self.groq_daily_limit
                    )
            if self.huggingface_token and self.huggingface_token not in self.key_statuses:
                self.key_statuses[self.huggingface_token] = APIKeyStatus(
                    key=self.huggingface_token, provider="huggingface", daily_limit=999999
                )

        for status in self.key_statuses.values():
            status.reset_if_needed()

    def get_active_key(self) -> Tuple[str, str]:
        with self._lock:
            self._rate_limit_unsafe()

            groq_statuses = [s for s in self.key_statuses.values()
                             if s.provider == "groq" and s.is_active]

            if self.groq_keys and not groq_statuses:
                for key in self.groq_keys:
                    if key not in self.key_statuses:
                        self.key_statuses[key] = APIKeyStatus(
                            key=key, provider="groq", daily_limit=self.groq_daily_limit
                        )
                groq_statuses = [s for s in self.key_statuses.values()
                                 if s.provider == "groq" and s.is_active]

            if groq_statuses:
                for _ in range(len(groq_statuses)):
                    current_status = groq_statuses[self.current_key_index % len(groq_statuses)]
                    current_status.reset_if_needed()

                    if not current_status.is_exhausted():
                        self.current_provider = "groq"
                        remaining = current_status.get_remaining_requests()
                        print(f"✓ Groq key {self.current_key_index + 1}: "
                              f"{current_status.key[:12]}... ({remaining:,} remaining)")
                        return current_status.key, "groq"

                    self.current_key_index += 1

            if self.huggingface_token:
                hf_status = self.key_statuses.get(self.huggingface_token)
                if hf_status and hf_status.is_active and not hf_status.is_exhausted():
                    self.current_provider = "huggingface"
                    print("\n⚠️  All Groq keys exhausted — switching to HuggingFace")
                    return hf_status.key, "huggingface"

            raise RuntimeError(
                "All API keys exhausted!\n"
                f"Groq keys: {len(groq_statuses)} all at daily limit\n"
                f"HuggingFace: {'Not configured' if not self.huggingface_token else 'Also exhausted'}\n"
                "Please wait until tomorrow or add more keys."
            )

    def _rate_limit_unsafe(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def record_success(self, api_key: str, response_headers: Dict = None):
        with self._lock:
            if api_key in self.key_statuses:
                status = self.key_statuses[api_key]
                if response_headers:
                    status.update_from_headers(response_headers)
                status.increment_usage()
                self.save_state()

    def record_error(self, api_key: str, error: Exception):
        with self._lock:
            if api_key in self.key_statuses:
                status = self.key_statuses[api_key]
                status.record_error()
                error_str = str(error)
                if "429" in error_str or "rate limit" in error_str.lower():
                    # Only mark daily-exhausted if it's a requests limit, not TPM
                    if "tokens per minute" not in error_str.lower():
                        status.requests_remaining = 0
                        print(f"⚠️  Key {api_key[:12]}... hit request limit, rotating")
                    # TPM errors: don't mark exhausted, just rotate temporarily
                self.save_state()

    def get_status_summary(self) -> Dict:
        with self._lock:
            summary = {"current_provider": self.current_provider, "groq_keys": [],
                       "huggingface": None, "total_remaining": 0}

            for status in self.key_statuses.values():
                if status.provider == "groq":
                    remaining = status.get_remaining_requests()
                    limit = status.requests_limit if status.requests_limit else status.daily_limit
                    summary["groq_keys"].append({
                        "key_preview": status.key[:12] + "...",
                        "requests_remaining": remaining,
                        "requests_limit": limit,
                        "exhausted": status.is_exhausted(),
                        "active": status.is_active,
                        "usage_percent": status.get_usage_percent(),
                        "tokens_remaining": status.tokens_remaining,
                        "reset_time": status.reset_time
                    })
                    summary["total_remaining"] += remaining
                elif status.provider == "huggingface":
                    summary["huggingface"] = {
                        "configured": True,
                        "requests_used": status.requests_made,
                        "active": status.is_active
                    }
            return summary

    def print_status(self):
        summary = self.get_status_summary()
        print(f"\n{'='*60}")
        print(f"Provider: {summary['current_provider'].upper()} | "
              f"Total remaining: {summary['total_remaining']:,}")
        for i, k in enumerate(summary['groq_keys'], 1):
            icon = "🟢" if not k['exhausted'] else "🔴"
            print(f"  {icon} Key {i} ({k['key_preview']}): "
                  f"{k['requests_remaining']:,}/{k['requests_limit']:,} "
                  f"({k['usage_percent']:.1f}%)")
        if summary["huggingface"]:
            print(f"  🟢 HuggingFace: {summary['huggingface']['requests_used']} used")
        print(f"{'='*60}\n")

    def save_state(self):
        state = {
            "key_statuses": {k: asdict(v) for k, v in self.key_statuses.items()},
            "current_key_index": self.current_key_index,
            "current_provider": self.current_provider,
            "last_updated": datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.key_statuses = {k: APIKeyStatus(**v) for k, v in state['key_statuses'].items()}
            self.current_key_index = state.get('current_key_index', 0)
            self.current_provider = state.get('current_provider', 'groq')
            print(f"✓ Loaded API key state from {self.state_file}")
        except Exception as e:
            print(f"⚠️  Could not load state: {e}")

    def reset_all(self):
        with self._lock:
            for status in self.key_statuses.values():
                status.requests_made = 0
                status.last_reset = datetime.now().isoformat()
                status.is_active = True
                status.error_count = 0
            self.save_state()
        print("✓ Reset all API key counters")


class ManagedGroqClient:

    def __init__(self, key_manager: APIKeyManager):
        self.key_manager = key_manager
        self._local = threading.local()
        self._init_client()

    def _init_client(self):
        key, provider = self.key_manager.get_active_key()
        self._local.current_key = key
        self._local.current_provider = provider

        if provider == "groq":
            from groq import Groq
            self._local.groq_client = Groq(api_key=key)
            self._local.hf_client = None
        else:
            from huggingface_hub import InferenceClient
            self._local.hf_client = InferenceClient(token=key)
            self._local.groq_client = None

    def _ensure_client(self):
        if not hasattr(self._local, 'current_key'):
            self._init_client()

    def _strip_to_text_only(self, messages: List[Dict]) -> List[Dict]:
        """Strip image_url parts — HF free tier is text-only."""
        text_only = []
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, list):
                text_parts = [
                    part['text'] for part in content
                    if isinstance(part, dict) and part.get('type') == 'text'
                ]
                content = '\n'.join(text_parts)
            text_only.append({'role': msg['role'], 'content': content})
        return text_only

    def chat_completions_create(self, messages: List[Dict], model: str, **kwargs):
        self._ensure_client()
        num_keys = max(len(self.key_manager.groq_keys), 1)
        max_retries = num_keys * 2 + 1  # enough to cycle all keys twice

        for attempt in range(max_retries):
            try:
                if self._local.current_provider == "groq":
                    response = self._local.groq_client.chat.completions.create(
                        messages=messages, model=model, **kwargs
                    )
                    headers = {}
                    if hasattr(response, '_raw_response') and hasattr(response._raw_response, 'headers'):
                        headers = dict(response._raw_response.headers)
                    self.key_manager.record_success(self._local.current_key, headers)
                    return response

                else:
                    hf_model = "meta-llama/Llama-3.3-70B-Instruct"
                    print(f"  🤗 HuggingFace chat_completion ({hf_model})...")
                    text_messages = self._strip_to_text_only(messages)
                    response = self._local.hf_client.chat_completion(
                        model=hf_model,
                        messages=text_messages,
                        max_tokens=kwargs.get('max_tokens', 1000),
                        temperature=kwargs.get('temperature', 0.7)
                    )
                    self.key_manager.record_success(self._local.current_key)
                    return response

            except Exception as e:
                error_str = str(e)
                is_tpm = "tokens per minute" in error_str.lower()
                is_429 = "429" in error_str or "rate limit" in error_str.lower()

                if is_429 and is_tpm:
                    # TPM hit — parse wait time from error, sleep then rotate key
                    wait = _parse_retry_wait(error_str)
                    print(f"⚠️  TPM limit on key {self._local.current_key[:12]}... "
                          f"waiting {wait:.1f}s then rotating (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    # Rotate to next key without marking this one exhausted
                    self.key_manager.current_key_index += 1
                    try:
                        self._init_client()
                    except RuntimeError:
                        if attempt == max_retries - 1:
                            raise
                    continue

                print(f"⚠️  Error attempt {attempt + 1}/{max_retries}: {e}")
                self.key_manager.record_error(self._local.current_key, e)
                try:
                    self._init_client()
                except RuntimeError:
                    if attempt == max_retries - 1:
                        raise
                    continue

        raise RuntimeError("All API keys failed after maximum retries")