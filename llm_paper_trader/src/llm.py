import json
import requests
from datetime import datetime, timezone

class LLMPricer:
    def __init__(self, api_key: str = None):
        # API key ignored. Pointing natively to local hardware instance.
        self.url = "http://localhost:11434/api/generate"
        self.model = "llama3.2:3b"

    def get_probability(self, question: str, context: str) -> dict:
        """
        Asks Claude to price the market based on context.
        Returns a dict: {"probability": float, "reasoning": str}
        """
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            system_prompt = f"Today is {today}. You are an elite superforecaster. Return ONLY valid JSON."
            user_prompt = f"""
        Market Question: {question}
        
        Here is the most recent news context:
        {context}
        
        Estimate the true objective probability that this event resolves as YES.
        Think step by step in one sentence of reasoning, then provide the final probability as a float between 0.0 and 1.0.
        
        Respond ONLY with this exact JSON format:
        {{
            "reasoning": "your rationale",
            "probability": 0.65
        }}
        """
        
            payload = {
                "model": self.model,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            }
            
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            text = response.json().get("response", "{}")
            
            result = json.loads(text.strip())
            return {
                "probability": float(result.get("probability", 0.5)),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            print(f"Local LLM API Error: {e}")
            return {"probability": 0.5, "reasoning": f"Error: {str(e)}"}
