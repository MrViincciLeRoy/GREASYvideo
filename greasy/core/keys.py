
# enhanced_api_key_manager.py
"""
Enhanced API Key Manager with Real-time Groq Quota Tracking
Reads actual remaining quota from Groq API headers
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time

@dataclass
class APIKeyStatus:
    """Track status of a single API key with real-time data"""
    key: str
    provider: str  # "groq" or "huggingface"

    # Groq provides these in response headers
    requests_remaining: Optional[int] = None  # x-ratelimit-remaining-requests
    requests_limit: Optional[int] = None      # x-ratelimit-limit-requests
    tokens_remaining: Optional[int] = None    # x-ratelimit-remaining-tokens
    tokens_limit: Optional[int] = None        # x-ratelimit-limit-tokens
    reset_time: Optional[str] = None          # x-ratelimit-reset-requests

    # Local tracking (fallback)
    requests_made: int = 0
    daily_limit: int = 14400
    last_reset: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True
    error_count: int = 0

    def update_from_headers(self, headers: Dict):
        """Update status from Groq API response headers"""
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
        """Check if key has hit daily limit (using real-time data if available)"""
        if self.requests_remaining is not None:
            # Use real-time data from Groq
            return self.requests_remaining <= 0
        else:
            # Fallback to local tracking
            return self.requests_made >= self.daily_limit

    def get_usage_percent(self) -> float:
        """Get usage percentage"""
        if self.requests_limit and self.requests_remaining is not None:
            used = self.requests_limit - self.requests_remaining
            return (used / self.requests_limit * 100) if self.requests_limit > 0 else 0
        else:
            return (self.requests_made / self.daily_limit * 100) if self.daily_limit > 0 else 0

    def get_remaining_requests(self) -> int:
        """Get remaining requests (prefer real-time data)"""
        if self.requests_remaining is not None:
            return self.requests_remaining
        else:
            return max(0, self.daily_limit - self.requests_made)

    def should_reset(self) -> bool:
        """Check if we should reset the counter (new day)"""
        last_reset = datetime.fromisoformat(self.last_reset)
        return datetime.now() - last_reset > timedelta(days=1)

    def reset_if_needed(self):
        """Reset counter if it's a new day"""
        if self.should_reset():
            self.requests_made = 0
            self.last_reset = datetime.now().isoformat()
            self.is_active = True

    def increment_usage(self):
        """Record a successful API call"""
        self.requests_made += 1
        self.last_used = datetime.now().isoformat()

    def record_error(self):
        """Record an error with this key"""
        self.error_count += 1
        if self.error_count >= 5:
            self.is_active = False

@dataclass
class APIKeyManager:
    """Enhanced manager with real-time Groq quota tracking"""
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
        """Initialize key statuses"""
        if os.path.exists(self.state_file):
            self.load_state()
        else:
            for key in self.groq_keys:
                if key not in self.key_statuses:
                    self.key_statuses[key] = APIKeyStatus(
                        key=key,
                        provider="groq",
                        daily_limit=self.groq_daily_limit
                    )

            if self.huggingface_token and self.huggingface_token not in self.key_statuses:
                self.key_statuses[self.huggingface_token] = APIKeyStatus(
                    key=self.huggingface_token,
                    provider="huggingface",
                    daily_limit=999999
                )

        for status in self.key_statuses.values():
            status.reset_if_needed()

    def get_active_key(self) -> Tuple[str, str]:
        """Get an active API key with available quota"""
        self._rate_limit()

        groq_statuses = [s for s in self.key_statuses.values()
                        if s.provider == "groq" and s.is_active]

        if len(self.groq_keys) > 0 and len(groq_statuses) == 0:
            print("⚠️  Re-initializing Groq key statuses...")
            for key in self.groq_keys:
                if key not in self.key_statuses:
                    self.key_statuses[key] = APIKeyStatus(
                        key=key,
                        provider="groq",
                        daily_limit=self.groq_daily_limit
                    )
            groq_statuses = [s for s in self.key_statuses.values()
                            if s.provider == "groq" and s.is_active]

        if len(groq_statuses) > 0:
            for _ in range(len(groq_statuses)):
                current_status = groq_statuses[self.current_key_index % len(groq_statuses)]
                current_status.reset_if_needed()

                if not current_status.is_exhausted():
                    self.current_provider = "groq"
                    remaining = current_status.get_remaining_requests()
                    print(f"✓ Using Groq key {self.current_key_index + 1}: "
                          f"{current_status.key[:12]}... ({remaining:,} requests remaining)")
                    return current_status.key, "groq"

                self.current_key_index += 1

        if self.huggingface_token:
            hf_status = self.key_statuses.get(self.huggingface_token)
            if hf_status and hf_status.is_active and not hf_status.is_exhausted():
                self.current_provider = "huggingface"
                print("\n⚠️  All Groq keys exhausted, switching to HuggingFace")
                return hf_status.key, "huggingface"

        raise RuntimeError(
            "All API keys exhausted!\n"
            f"Groq keys: {len(groq_statuses)} all at daily limit\n"
            f"HuggingFace: {'Not configured' if not self.huggingface_token else 'Also exhausted'}\n"
            "Please wait until tomorrow or add more keys."
        )

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def record_success(self, api_key: str, response_headers: Dict = None):
        """Record successful API call with real-time quota update"""
        if api_key in self.key_statuses:
            status = self.key_statuses[api_key]

            # Update from response headers (real-time data from Groq)
            if response_headers:
                status.update_from_headers(response_headers)

            # Also increment local counter
            status.increment_usage()
            self.save_state()

    def record_error(self, api_key: str, error: Exception):
        """Record failed API call"""
        if api_key in self.key_statuses:
            status = self.key_statuses[api_key]
            status.record_error()

            error_str = str(error).lower()
            if "rate limit" in error_str or "429" in error_str:
                status.requests_remaining = 0  # Mark as exhausted
                print(f"⚠️  Key {api_key[:12]}... hit rate limit, rotating")

            self.save_state()

    def get_status_summary(self) -> Dict:
        """Get current status with real-time data"""
        summary = {
            "current_provider": self.current_provider,
            "groq_keys": [],
            "huggingface": None,
            "total_remaining": 0
        }

        for status in self.key_statuses.values():
            if status.provider == "groq":
                remaining = status.get_remaining_requests()
                limit = status.requests_limit if status.requests_limit else status.daily_limit

                key_info = {
                    "key_preview": status.key[:12] + "...",
                    "requests_remaining": remaining,
                    "requests_limit": limit,
                    "exhausted": status.is_exhausted(),
                    "active": status.is_active,
                    "usage_percent": status.get_usage_percent(),
                    "tokens_remaining": status.tokens_remaining,
                    "reset_time": status.reset_time
                }
                summary["groq_keys"].append(key_info)
                summary["total_remaining"] += remaining

            elif status.provider == "huggingface":
                summary["huggingface"] = {
                    "configured": True,
                    "requests_used": status.requests_made,
                    "active": status.is_active
                }

        return summary

    def print_status(self):
        """Print detailed status with real-time quota"""
        print("\n" + "="*80)
        print("API KEY MANAGER STATUS (Real-time from Groq)")
        print("="*80)

        summary = self.get_status_summary()

        print(f"\nCurrent Provider: {summary['current_provider'].upper()}")
        print(f"Total Remaining Requests: {summary['total_remaining']:,}")

        print(f"\nGroq Keys ({len(summary['groq_keys'])} total):")
        print("-"*80)
        for i, key_info in enumerate(summary['groq_keys'], 1):
            status_icon = "✓" if not key_info['exhausted'] else "✗"
            active_icon = "🟢" if key_info['active'] else "🔴"

            print(f"  {active_icon} Key {i} ({key_info['key_preview']}): "
                  f"{key_info['requests_remaining']:,}/{key_info['requests_limit']:,} requests "
                  f"({key_info['usage_percent']:.1f}%) {status_icon}")

            if key_info['tokens_remaining']:
                print(f"      Tokens: {key_info['tokens_remaining']:,} remaining")

            if key_info['reset_time']:
                print(f"      Resets: {key_info['reset_time']}")

        if summary["huggingface"]:
            print(f"\nHuggingFace:")
            print("-"*80)
            active_icon = "🟢" if summary['huggingface']['active'] else "🔴"
            print(f"  {active_icon} Configured (used: {summary['huggingface']['requests_used']} requests)")
        else:
            print(f"\nHuggingFace: ❌ Not configured")

        print("="*80 + "\n")

    def save_state(self):
        """Save state to file"""
        state = {
            "key_statuses": {k: asdict(v) for k, v in self.key_statuses.items()},
            "current_key_index": self.current_key_index,
            "current_provider": self.current_provider,
            "last_updated": datetime.now().isoformat()
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load state from file"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.key_statuses = {}
            for key, status_dict in state['key_statuses'].items():
                self.key_statuses[key] = APIKeyStatus(**status_dict)

            self.current_key_index = state.get('current_key_index', 0)
            self.current_provider = state.get('current_provider', 'groq')

            print(f"✓ Loaded API key state from {self.state_file}")
        except Exception as e:
            print(f"⚠️  Could not load state: {e}")

    def reset_all(self):
        """Reset all counters"""
        for status in self.key_statuses.values():
            status.requests_made = 0
            status.last_reset = datetime.now().isoformat()
            status.is_active = True
            status.error_count = 0
        self.save_state()
        print("✓ Reset all API key counters")


class ManagedGroqClient:
    """Enhanced client that tracks real-time quota from Groq"""

    def __init__(self, key_manager: APIKeyManager):
        self.key_manager = key_manager
        self.groq_client = None
        self.hf_client = None
        self.current_key = None
        self.current_provider = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize or rotate to a new client"""
        try:
            self.current_key, self.current_provider = self.key_manager.get_active_key()

            if self.current_provider == "groq":
                from groq import Groq
                self.groq_client = Groq(api_key=self.current_key)
            elif self.current_provider == "huggingface":
                from huggingface_hub import InferenceClient
                self.hf_client = InferenceClient(token=self.current_key)
        except RuntimeError as e:
            print(f"\n❌ {e}")
            raise

    def chat_completions_create(self, messages: List[Dict], model: str, **kwargs):
        """Create chat completion with real-time quota tracking"""
        max_retries = len(self.key_manager.groq_keys) + 1

        for attempt in range(max_retries):
            try:
                if self.current_provider == "groq":
                    response = self.groq_client.chat.completions.create(
                        messages=messages,
                        model=model,
                        **kwargs
                    )

                    # Extract rate limit headers from response
                    # Note: response._raw_response.headers contains the headers
                    headers = {}
                    if hasattr(response, '_raw_response') and hasattr(response._raw_response, 'headers'):
                        headers = dict(response._raw_response.headers)

                    # Record success with real-time quota data
                    self.key_manager.record_success(self.current_key, headers)
                    return response

                elif self.current_provider == "huggingface":
                    hf_model = "meta-llama/Llama-3.3-70B-Instruct"
                    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

                    response_text = self.hf_client.text_generation(
                        prompt,
                        model=hf_model,
                        max_new_tokens=kwargs.get('max_tokens', 1000),
                        temperature=kwargs.get('temperature', 0.7)
                    )

                    class HFResponse:
                        def __init__(self, text):
                            self.choices = [type('obj', (object,), {
                                'message': type('obj', (object,), {
                                    'content': text
                                })()
                            })()]

                    self.key_manager.record_success(self.current_key)
                    return HFResponse(response_text)

            except Exception as e:
                print(f"⚠️  Error with {self.current_provider} (attempt {attempt + 1}/{max_retries}): {e}")
                self.key_manager.record_error(self.current_key, e)

                try:
                    self._initialize_client()
                except RuntimeError:
                    if attempt == max_retries - 1:
                        raise
                    continue

        raise RuntimeError("All API keys failed after maximum retries")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║      ENHANCED API KEY MANAGER WITH REAL-TIME TRACKING        ║
║                                                              ║
║  • Reads actual quota from Groq API response headers        ║
║  • Shows exact remaining requests                           ║
║  • Tracks token usage                                       ║
║  • Displays reset times                                     ║
╚══════════════════════════════════════════════════════════════╝
""")

    manager = APIKeyManager(
        groq_keys=[ os.getenv('GROQ_KEY'), 
        os.getenv('GROQ_KEY_2'), 
       os.getenv('GROQ_KEY_3'), 
       os.getenv('GROQ_KEY_4')], 
        huggingface_token=None,
        state_file="api_keys.json"
    )

    manager.print_status()

    try:
        client = ManagedGroqClient(manager)
        print("\n✓ Client initialized!")

        print("\n🧪 Making test API call to get real-time quota...")
        response = client.chat_completions_create(
            messages=[{"role": "user", "content": "Say 'Hello' in 1 word"}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=10
        )
        print(f"✓ Response: {response.choices[0].message.content}")

        print("\n📊 Updated status with real-time data from Groq:")
        manager.print_status()

    except Exception as e:
        print(f"\n❌ Error: {e}")