from dataclasses import dataclass
import re


@dataclass
class CheckpointResult:
    decision: str   # ALLOW, SANITIZE, BLOCK
    content: str
    reason: str = ""


class SecurityCheckpoint:
    def __init__(self):
        self.patterns = [
            r"ignore (all|any|previous) instructions",
            r"forget (all|any|previous) instructions",
            r"disregard (all|any|previous) instructions",
            r"act as (a )?system",
            r"reveal (your|the) prompt",
            r"what are your instructions",
            r"repeat exactly",
            r"always reply with",
        ]

    def evaluate_input(self, text: str) -> CheckpointResult:
        lowered = text.lower().strip()

        for pattern in self.patterns:
            if re.search(pattern, lowered):
                return CheckpointResult(
                    decision="BLOCK",
                    content="Your message was blocked by the security checkpoint.",
                    reason=f"Matched pattern: {pattern}"
                )

        return CheckpointResult(
            decision="ALLOW",
            content=text
        )

    def evaluate_output(self, text: str) -> CheckpointResult:
        lowered = text.lower().strip()

        if "goats are bad" in lowered:
            return CheckpointResult(
                decision="SANITIZE",
                content="I can only provide safe responses about goats.",
                reason="Unsafe target phrase detected in model output."
            )

        return CheckpointResult(
            decision="ALLOW",
            content=text
        )


def is_prompt_safe(prompt: str) -> tuple[bool, str]:
    banned_patterns = [
        "ignore previous instructions",
        "reveal system prompt",
        "bypass safety",
        "jailbreak",
        "you are now",
        "disregard all prior instructions",
    ]  

    lower_prompt = prompt.lower()

    for pattern in banned_patterns:
        if pattern in lower_prompt:
            return False, f"Blocked suspicious prompt pattern: {pattern}"

    return True, "Prompt is safe"