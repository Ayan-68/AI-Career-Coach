from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import gradio as gr

model_id = "meta-llama/llama-3-2-11b-vision-instruct"

credentials = Credentials(
    url = " ", #Fill your Credentials
)

params = TextChatParameters(
    temperature=0.1,
    max_tokens=512
)

project_id = " " #Fill you Project ID

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)

prompt_txt = "How to be a good Data Scientist?"

def generate_response(prompt_txt):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_txt
            },
        ]
    }
]   

    generated_response = model.chat(messages=messages)
    generated_text = generated_response['choices'][0]['message']['content']

    return generated_text

chat_app = gr.Interface(
    fn=generate_response,
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type Your Question here....."),
    outputs = gr.Textbox(label="output"),
    title = "Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."

)

chat_app.launch()
