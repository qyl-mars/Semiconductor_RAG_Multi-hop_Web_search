from openai import OpenAI
from config.configs import Config

client = OpenAI(
    api_key=Config.llm_api_key,
    base_url=Config.llm_base_url
)
#"gpt-5.2-pro"
print("start")
try:
    # 调用一个简单的接口，验证 API Key 是否有效
    response = client.chat.completions.create(
        model=Config.llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, 你是什么模型?"}
        ],
        max_tokens=50
    )
    print("✅ API Key is valid! Here is a sample response:")
    #print(response.choices[0].message["content"])
    print(response.choices[0].message.content.strip())

except Exception as e:
    print(f"⚠️ An error occurred: {e}")
