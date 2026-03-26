from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}
EmailCategory = Literal["spam", "urgent", "normal", "escalation"]
ActionType = Literal["classify", "respond", "escalate", "ignore", "wait"]
ResponseTemplate = Literal[
    "acknowledge",
    "resolve",
    "request_info",
    "escalate_notice",
    "none",
]


class VersionedModel(BaseModel):
    schema_version: str = Field(default="1.0.0")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema version: {value}")
        return value


class EmailView(VersionedModel):
    email_id: str
    sender: str
    subject: str
    body: str
    thread_id: str
    age: int = Field(ge=0)
    priority_hint: Literal["low", "medium", "high", "critical"]
    noise_score: float = Field(ge=0.0, le=1.0)
    seen: bool = False


class ActionTrace(VersionedModel):
    step_index: int = Field(ge=0)
    email_id: Optional[str] = None
    action_type: ActionType
    summary: str


class Observation(VersionedModel):
    task_name: str
    step_index: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    remaining_steps: int = Field(ge=0)
    seed: int
    inbox: list[EmailView]
    completed_email_ids: list[str]
    action_history: list[ActionTrace]


class Action(VersionedModel):
    action_type: ActionType
    email_id: Optional[str] = None
    category: Optional[EmailCategory] = None
    response_template: Optional[ResponseTemplate] = None
    priority: Optional[Literal["low", "medium", "high", "critical"]] = None

    @model_validator(mode="after")
    def validate_action(self) -> "Action":
        if self.action_type == "wait":
            if any(
                value is not None
                for value in (self.email_id, self.category, self.response_template, self.priority)
            ):
                raise ValueError("wait action cannot target an email or include extra fields")
            return self

        if self.email_id is None:
            raise ValueError("email_id is required for non-wait actions")

        if self.action_type == "classify" and self.category is None:
            raise ValueError("classify action requires category")
        if self.action_type != "classify" and self.category is not None:
            raise ValueError("category is only valid for classify actions")

        if self.action_type == "respond":
            if self.response_template is None or self.response_template == "none":
                raise ValueError("respond action requires a non-empty response_template")
        elif self.response_template is not None:
            raise ValueError("response_template is only valid for respond actions")

        if self.action_type in {"respond", "escalate"} and self.priority is None:
            raise ValueError("respond and escalate actions require priority")
        if self.action_type not in {"respond", "escalate"} and self.priority is not None:
            raise ValueError("priority is only valid for respond/escalate actions")
        return self


class Reward(VersionedModel):
    total: float
    components: dict[str, float] = Field(default_factory=dict)
    reason: str = ""

    @field_validator("components")
    @classmethod
    def ensure_json_safe_components(cls, value: dict[str, float]) -> dict[str, float]:
        return {str(key): float(amount) for key, amount in value.items()}


class EmailSpec(VersionedModel):
    email_id: str
    sender: str
    subject: str
    body: str
    thread_id: str
    arrival_step: int = Field(ge=0)
    priority_hint: Literal["low", "medium", "high", "critical"]
    noise_score: float = Field(ge=0.0, le=1.0)
    true_category: EmailCategory
    response_template: ResponseTemplate = "none"
    requires_response: bool = False
    requires_escalation: bool = False
    escalation_trigger_step: Optional[int] = Field(default=None, ge=0)
    classification_deadline: int = Field(ge=0)
    response_deadline: int = Field(ge=0)
    escalation_deadline: int = Field(ge=0)

    def to_view(self, current_step: int, seen: bool) -> EmailView:
        return EmailView(
            email_id=self.email_id,
            sender=self.sender,
            subject=self.subject,
            body=self.body,
            thread_id=self.thread_id,
            age=max(current_step - self.arrival_step, 0),
            priority_hint=self.priority_hint,
            noise_score=self.noise_score,
            seen=seen,
        )


class StepRecord(VersionedModel):
    step_index: int = Field(ge=0)
    action: Action
    reward: Reward
    observation: Observation
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)

