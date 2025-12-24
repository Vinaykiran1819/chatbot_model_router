import anthropic
from typing import Generator
from .base import BaseWrapper

class AnthropicWrapper(BaseWrapper):
    def _setup_client(self):
        return anthropic.Anthropic(api_key=self.api_key)

    def _stream_api(self, prompt: str, system_instruction: str) -> Generator[str, None, None]:
        # Anthropic requires system prompt to be a top-level parameter, not in messages
        try:
            stream = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1024,
                system=system_instruction if system_instruction else "",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            for event in stream:
                if event.type == "content_block_delta":
                    yield event.delta.text
                    
        except Exception as e:
            yield f"Error calling Anthropic: {str(e)}"