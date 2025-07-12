from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Per user request, using the new google.genai structure
from google import genai
from google.genai import types
from google.genai.types import HarmCategory, HarmBlockThreshold

from .env_config import Config
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class GeminiClient:
    """Simple wrapper around Google Gemini API using the Client interface."""

    config: Config
    client: Optional[genai.client.Client] = None
    # Token usage tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0

    def __post_init__(self) -> None:
        # Configure the API key directly in the client constructor
        self.client = genai.Client(api_key=self.config.GEMINI_API_KEY)

    def _clean_response(self, response: str) -> str:
        """Clean response from markdown formatting and other artifacts.
        
        Gemini often wraps responses (especially code or JSON) in markdown blocks like:
        ```python
        print("Hello, World!")
        ```
        
        This method extracts just the content within the block.
        """
        if not response or not response.strip():
            return response
            
        # Remove leading/trailing whitespace
        cleaned = response.strip()
        
        # Pattern to match content wrapped in markdown code blocks.
        # It handles optional language identifiers (like python, json, etc.).
        code_block_pattern = r'```(?:[a-zA-Z]*)?\s*\n?(.*?)\n?```'
        match = re.search(code_block_pattern, cleaned, re.DOTALL | re.IGNORECASE)
        
        if match:
            # Extract content from the code block
            content = match.group(1).strip()
            logger.debug("Extracted content from markdown code block")
            return content
        
        # If not in a markdown block, return the cleaned string as-is.
        return cleaned

    def _extract_response_content(self, response: types.GenerateContentResponse) -> str:
        """Extract content from Gemini response, handling errors."""
        try:
            # The most direct way to get text content
            return response.text
        except Exception as e:
            logger.warning("Could not access response.text, trying parts: %s", e)
            # Fallback for complex responses or errors
            if response.candidates:
                content_parts = [part.text for part in response.candidates[0].content.parts if part.text]
                return "".join(content_parts)
            raise RuntimeError("Failed to extract content from Gemini response.") from e

    def _convert_messages_to_contents(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Converts an OpenAI-style message list to a Gemini-style contents list."""
        contents = []
        # Gemini uses 'model' for the assistant role, and 'user' for the user role.
        # System messages are handled as the first part of the user's first message.
        system_prompt = ""
        user_messages = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                user_messages.append(message)

        is_first_user_message = True
        for message in user_messages:
            role = "model" if message.get("role") == "assistant" else "user"
            content = message.get("content", "")
            
            if role == "user" and is_first_user_message and system_prompt:
                content = f"{system_prompt}\n\n{content}"
                is_first_user_message = False

            contents.append({'role': role, 'parts': [{'text': content}]})

        # Ensure the conversation starts with a user message if it doesn't already
        if contents and contents[0]['role'] != 'user':
            contents.insert(0, {'role': 'user', 'parts': [{'text': '...'}]})

        return contents

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        # use_thinking semantics:
        #   • False (default) ➜ thinking disabled (budget=0)
        #   • True           ➜ dynamic thinking (budget=-1)
        #   • int  > 0      ➜ fixed thinking budget in tokens
        use_thinking: bool | int = False,
        clean_response: bool = True,
        **kwargs: Any,
    ) -> str:
        """Call the Gemini API using the Client interface and return content string."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized.")

        logger.debug("Sending prompt with %d messages to Gemini", len(messages))
        
        # CORRECTED: Convert to Gemini's structured 'contents'
        contents = self._convert_messages_to_contents(messages)
        
        # Build the generation configuration
        gen_config_params = {"temperature": temperature}
        if max_tokens is not None:
            gen_config_params["max_output_tokens"] = max_tokens

        # ------------------------------------------------------------------
        # Thinking configuration – disable by default for cost savings
        # ------------------------------------------------------------------
        thinking_conf = None
        try:
            if use_thinking is False:
                # Explicitly disable model thinking to avoid extra token costs
                thinking_conf = types.ThinkingConfig(thinking_budget=0)
            elif use_thinking is True:
                # Dynamic thinking (-1) lets the model decide budget
                thinking_conf = types.ThinkingConfig(thinking_budget=-1)
            elif isinstance(use_thinking, int):
                # Caller provided an explicit budget (>=0)
                thinking_conf = types.ThinkingConfig(thinking_budget=use_thinking)
        except Exception as exc:
            logger.warning("Could not create ThinkingConfig: %s", exc)

        # Define safety settings dictionary before using it
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # CORRECTED: Use GenerationConfig and bundle all settings inside it
        generation_config = types.GenerateContentConfig(
            **gen_config_params,
            safety_settings=[
                types.SafetySetting(category=key, threshold=value)
                for key, value in safety_settings.items()
            ],
            thinking_config=thinking_conf,
            **kwargs
        )

        try:
            # CORRECTED: Use 'config' as the parameter name
            response = self.client.models.generate_content(
                model=self.config.GEMINI_MODEL,
                contents=contents,
                config=generation_config,
            )
            
            # Token usage tracking
            if response.usage_metadata:
                self.total_prompt_tokens += response.usage_metadata.prompt_token_count
                self.total_completion_tokens += response.usage_metadata.candidates_token_count
                self.total_tokens += response.usage_metadata.total_token_count
            self.request_count += 1
            
            content = self._extract_response_content(response)
            logger.debug("Received response with %d characters", len(content))
            
            if clean_response:
                content = self._clean_response(content)
            
            return content
            
        except Exception as e:
            logger.error("Gemini API error: %s", str(e))
            raise
    
    def check_version(self) -> None:
        """Prints the google-generativeai library version and the configured model."""
        try:
            # The version is available on the package itself.
            version = getattr(genai, '__version__', 'N/A')
            print("\n" + "="*50)
            print("GEMINI VERSION INFORMATION")
            print("="*50)
            print(f"google.genai library version: {version}")
            print(f"Configured model: {self.config.GEMINI_MODEL}")
            print("="*50)
        except ImportError:
            logger.error("Could not import google.genai to check version.")
        except Exception as e:
            logger.error(f"An error occurred while checking version: {e}")

    def get_token_usage(self) -> Dict[str, int]:
        """Return current token usage statistics."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }
    
    def print_token_summary(self) -> None:
        """Print a formatted summary of token usage."""
        usage = self.get_token_usage()
        print("\n" + "="*50)
        print("GEMINI TOKEN USAGE SUMMARY")
        print("="*50)
        print(f"Total Requests: {usage['request_count']}")
        print(f"Prompt Tokens: {usage['total_prompt_tokens']:,}")
        print(f"Completion Tokens: {usage['total_completion_tokens']:,}")
        print(f"Total Tokens: {usage['total_tokens']:,}")
        
        if usage['total_tokens'] > 0:
            estimated_cost = (usage['total_prompt_tokens'] * 0.3 + usage['total_completion_tokens'] * 2.5) / 1000000
            print(f"💰 Estimated Cost: ~${estimated_cost:.6f}")
        print("="*50)


# For backward compatibility, create an alias
OpenAIClient = GeminiClient 