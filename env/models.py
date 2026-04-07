from pydantic import BaseModel
from typing import List, Dict, Optional


# What agent sees
class Observation(BaseModel):
    documents: List[Dict]   # list of docs {id, text}
    question: str           # question to answer
    history: List[str]      # past actions (can be empty)


# What agent does
class Action(BaseModel):
    answer: str             # generated answer
    source: Optional[int]   # document id used


# What agent gets
class Reward(BaseModel):
    score: float            # 0.0 to 1.0
    feedback: str           # explanation of reward