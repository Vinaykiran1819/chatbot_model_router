# src/wrappers/gemini.py
from google import genai
from typing import Generator
from .base import BaseWrapper

class GeminiWrapper(BaseWrapper):
    def _setup_client(self):
        # Initialize the new Google GenAI Client
        return genai.Client(api_key=self.api_key)

    def _stream_api(self, prompt: str, system_instruction: str) -> Generator[str, None, None]:
        try:
            # Configure system instructions if provided
            config = None
            if system_instruction:
                config = {"system_instruction": system_instruction}

            # Call the API using the new v1 SDK method
            response = self.client.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=prompt,
                config=config
            )
            
            # Yield chunks of text as they arrive
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            yield f"Error calling Gemini: {str(e)}"