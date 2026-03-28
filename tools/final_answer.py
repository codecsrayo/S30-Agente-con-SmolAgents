from typing import Any
from smolagents.tools import Tool


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # bug fix: llamar al padre
        self.is_initialized = True

    def forward(self, answer: Any) -> Any:
        return answer
