from openai import OpenAI
from config.configs import Config

client = OpenAI(
    api_key=Config.llm_api_key,
    base_url=Config.llm_base_url
)

class DeepSeekClient:
    def generate_answer(self, system_prompt, user_prompt, model=Config.llm_model):
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
