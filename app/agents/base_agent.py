from abc import ABC, abstractmethod
from app.llm.client import llm
from app.utils.logger import logger

class BaseAgent(ABC):
    name: str = "base"

    def __init__(self):
        self.llm = llm

    @abstractmethod
    def run(self, state: dict) -> dict:
        ...

    def log(self, msg: str):
        logger.info(f"[{self.name}] {msg}")