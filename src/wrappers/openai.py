# src/wrappers/openai.py
from openai import OpenAI
from typing import Generator
from .base import BaseWrapper

class OpenAIWrapper(BaseWrapper):
    def _setup_client(self):
        return OpenAI(api_key=self.api_key)

    def _stream_api(self, prompt: str, system_instruction: str) -> Generator[str, None, None]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True  # ENABLE STREAMING
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content