from typing import Optional
from pydantic import BaseModel, HttpUrl, model_validator

class DecriptoRequest(BaseModel):
    modelo: str
    type: str
    content: Optional[str] = None
    url:     Optional[HttpUrl] = None

    @model_validator(mode="before")
    def check_content_or_url(cls, values):
        if not values.get("content") and not values.get("url"):
            raise ValueError("Você deve informar content ou url")
        return values
