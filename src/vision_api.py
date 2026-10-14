from openai import OpenAI
from dotenv import load_dotenv
import os
import base64


# 加载API Key
load_dotenv()

api_key = os.getenv(
    "DASHSCOPE_API_KEY"
)


client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)



def image_to_base64(path):

    with open(path, "rb") as f:
        data = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return data



def analyze_image(image_path):

    image_base64 = image_to_base64(
        image_path
    )


    response = client.chat.completions.create(
        model="qwen3.8-omni-flash",

        messages=[
            {
                "role": "user",

                "content":[

                    {
                        "type":"text",
                        "text":
                        "请分析这张图片，描述其中的物体、环境和主要信息。"
                    },

                    {
                        "type":"image_url",

                        "image_url":{
                            "url":
                            f"data:image/jpeg;base64,{image_base64}"
                        }
                    }

                ]
            }
        ]
    )


    return response.choices[0].message.content



if __name__ == "__main__":


    path = input(
        "请输入图片路径:"
    )


    result = analyze_image(path)


    print("\n视觉大模型分析结果:")
    print("----------------")

    print(result)