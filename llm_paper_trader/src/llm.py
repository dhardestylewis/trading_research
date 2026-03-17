import json
import requests
from datetime import datetime, timezone

class LLMPricer:
    def __init__(self, api_key: str = None):
        import os
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "qwen/qwen-2.5-72b-instruct:free"

    def get_probability(self, question: str, context: str, active_rules: str = "") -> dict:
        """
        Asks OpenRouter (Qwen Series) to price the market based on context.
        Returns a dict: {"probability": float, "reasoning": str}
        """
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            system_prompt = f"Today is {today}. You are an elite quantitative superforecaster. Return ONLY valid JSON."
            if active_rules:
                system_prompt += f"\n\nCRITICAL TRADING RULES (LEARNED FROM PAST RECORD):\n{active_rules}"
                
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
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/dhardestylewis/trading_research",
                "X-Title": "LLM Paper Trader",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            
            result = json.loads(text.strip())
            return {
                "probability": float(result.get("probability", 0.5)),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            print(f"Serverless LLM API Error: {e}")
            return {"probability": 0.5, "reasoning": f"Error: {str(e)}"}
