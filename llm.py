import prompts

from pydantic import BaseModel, Field, ConfigDict
from openai import OpenAI

from typing import List, Dict

class LLMEngine(BaseModel):
    user_game_description: str
    context: List[Dict[str, str]] = Field(default_factory=list)
    client: OpenAI = Field(default_factory=OpenAI)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, user_game_description: str, **kwargs):
        super().__init__(user_game_description=user_game_description, **kwargs)

        self.context = [
            {
                "role": "system",
                "content": prompts.INITIAL_PROMPT_TEMPLATE.replace("{{&USER_GAME_DESCRIPTION}}", user_game_description)
            },
        ]

    def get_completion(self) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.context,
        ).choices[0].message.content
        self.context.append({"role": "assistant", "content": completion})
        return completion

    def ingest_command(self, command: str) -> None:
        self.context.append({"role": "user", "content": command})

    def step(self, input_text: str=None) -> str:
        if input_text is not None:
            self.ingest_command(input_text)
        return self.get_completion()
