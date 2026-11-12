
from dataclasses import dataclass


@dataclass
class Ctx:
    session_id: str
    user_id: str
    token: str
