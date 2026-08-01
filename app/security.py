import re

# Common prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"bypass system prompt",
    r"you are now DAN",
    r"reveal system prompt",
    r"disregard all prior rules"
]

def sanitize_and_validate_prompt(user_input: str) -> str:
    """Validates user input against prompt injection and scrubs sensitive PII."""
    # 1. Prompt Injection Check
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            raise ValueError("Potential security policy violation detected in prompt.")
            
    # 2. PII Masking (Email & Phone Numbers)
    sanitized = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', user_input)
    sanitized = re.sub(r'\+?\d{10,13}', '[PHONE_REDACTED]', sanitized)
    
    return sanitized