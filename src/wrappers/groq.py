# src/wrappers/groq.py
import os
from groq import Groq
from typing import Generator
from .base import BaseWrapper

class GroqWrapper(BaseWrapper):
    def _setup_client(self):
        return Groq(api_key=self.api_key)

    def _stream_api(self, prompt: str, system_instruction: str) -> Generator[str, None, None]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            # UPDATED: Using Llama 3.3 70B Versatile as requested
            stream = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error calling Groq: {str(e)}"