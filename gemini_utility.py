import json
import os
import time
import tqdm.notebook as tqdm

from concurrent.futures import ThreadPoolExecutor
from google import genai
from pydantic import BaseModel
from typing import Optional, Type


if not os.path.exists('genai_api_key.txt'):
    API_KEY = ''
else:
    with open('genai_api_key.txt', 'r') as f:
        API_KEY = f.read().strip()


class GeminiInstructJsonClient:
    def __init__(
        self,
        llm_model: str, 
        instruction: str,
        resp_model: Type[BaseModel]
    ):
        if not API_KEY:
            raise ValueError("API key not found. Please create a 'genai_api_key.txt' file with your Gemini API key.")
        self._client = genai.Client(api_key=API_KEY)
        self._llm_model = llm_model
        self._instruction = instruction
        self._resp_model = resp_model
        self._resp_config = genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(
                thinking_budget=0,
            ),
            response_mime_type="application/json",
            response_schema=resp_model.model_json_schema()
        )

    def execute_task(self, text: str) -> BaseModel:
        response = self._client.models.generate_content(
            model=self._llm_model,
            contents=self._instruction + text,
            config=self._resp_config
        )
        return self._resp_model.model_validate_json(response.text)

    def safe_execute_task(self, text: str) -> tuple[str, Optional[dict]]:
        for _ in range(5):  # Retry up to 5 times
            try:
                resp = self.execute_task(text)
                return '', resp.model_dump()
            except Exception as e:
                last_exception = e
                time.sleep(.2)  # Backoff before retrying
        return str(last_exception), None
    
    def execute_and_save_cache(
        self, 
        cache_dir: str, 
        texts: list[str], 
        batch_size: int = 1000,
        max_workers: int = 8
    ):
        os.makedirs(cache_dir, exist_ok=True)
        for idx in tqdm.tqdm(list(range(0, len(texts), batch_size))):
            batch_texts = texts[idx:idx + batch_size]
            self._execute_and_save_batch(cache_dir, batch_texts, idx // batch_size, max_workers)

    def _execute_and_save_batch(self, cache_dir: str, texts: list[str], idx: int, max_workers: int) -> list[dict]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.safe_execute_task, texts))
        with open(os.path.join(cache_dir, f'batch_{str(idx).zfill(3)}.jsonl'), 'w') as f:
            for res in results:
                f.write(json.dumps(res) + '\n')
