from typing import List

from pydantic import BaseModel


class User(BaseModel):
    id: int
    is_active: bool
    wx_infos: List[str] = []

    class Config:
        from_attributes = True
