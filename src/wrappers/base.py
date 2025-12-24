# src/wrappers/base.py
from abc import ABC, abstractmethod
from typing import Generator, Dict, Any
import time

class BaseWrapper(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = self._setup_client()

    @abstractmethod
    def _setup_client(self):
        pass

    @abstractmethod
    def _stream_api(self, prompt: str, system_instruction: str) -> Generator[str, None, None]:
        """Internal method to handle the specific API streaming logic."""
        pass

    def generate_stream(self, prompt: str, system_instruction: str = None) -> Generator[Dict[str, Any], None, None]:
        """
        Yields chunks of text AND metrics.
        The first yield is the TTFT (Time to First Token).
        Subsequent yields are text chunks.
        The final yield is the total metrics.
        """
        start_time = time.time()
        first_token_received = False
        token_count = 0
        
        # Call the child class's specific API stream implementation
        stream = self._stream_api(prompt, system_instruction)
        
        for chunk in stream:
            current_time = time.time()
            
            # Capture TTFT on the very first chunk
            if not first_token_received:
                ttft = current_time - start_time
                yield {"type": "metric", "key": "ttft", "value": ttft}
                first_token_received = True
            
            # Yield the actual text content
            if chunk:
                token_count += 1 # Approximate token count
                yield {"type": "content", "value": chunk}

        total_time = time.time() - start_time
        yield {
            "type": "final_metrics", 
            "total_time": total_time,
            "token_count": token_count,
            "tokens_per_second": token_count / total_time if total_time > 0 else 0
        }