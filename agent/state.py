from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class Step(BaseModel):
    id: int
    action: str
    target: str = ""
    value: str = ""
    reason: str = ""
    status: str = "PENDING"
    result: Optional[dict] = None


class Task(BaseModel):
    id: str
    user_prompt: str
    status: str = "CREATED"
    steps: list[Step] = Field(default_factory=list)
    current_step: int = 0
    observation: str = ""
    last_error: Optional[str] = None
    history: list[dict] = Field(default_factory=list)
    intervention: Optional[dict] = None
    stop_requested: bool = False
    visited_urls: list[str] = Field(default_factory=list)

    @classmethod
    def create(cls, prompt: str):
        return cls(id=str(uuid4()), user_prompt=prompt)
