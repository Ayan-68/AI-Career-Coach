from ibm_watsonx_ai import Credentials  
from ibm_watsonx_ai import APIClient  
from ibm_watsonx_ai.foundation_models import Model, ModelInference  
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters  
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames  
import gradio as gr  

model_id = "meta-llama/llama-3-2-11b-vision-instruct"  

credentials = Credentials(  
    url=" ", #Your Credentials  
)  

params = TextChatParameters(  
    temperature=0.7,  
    max_tokens=1024  
)  

project_id = " " #Your Project ID 

model = ModelInference(  
    model_id=model_id,  
    credentials=credentials,  
    project_id=project_id,  
    params=params  
)  

def generate_career_advice(position_applied, job_description, resume_content):  
    prompt = f"Considering the job description: {job_description}, and the resume provided: {resume_content}, identify areas for enhancement in the resume. Offer specific suggestions on how to improve these aspects to better match the job requirements and increase the likelihood of being selected for the position of {position_applied}."  

    messages = [  
        {  
            "role": "user",  
            "content": [  
                {  
                    "type": "text",  
                    "text": prompt  
                },  
            ]  
        }  
    ]   
    
    generated_response = model.chat(messages=messages)  
    advice = generated_response['choices'][0]['message']['content']  
    return advice  

career_advice_app = gr.Interface(  
    fn=generate_career_advice,  
    flagging_mode="never",  
    inputs=[  
        gr.Textbox(label="Position Applied For", placeholder="Enter the position you are applying for..."),  
        gr.Textbox(label="Job Description Information", placeholder="Paste the job description here...", lines=10),  
        gr.Textbox(label="Your Resume Content", placeholder="Paste your resume content here...", lines=10),  
    ],  
    outputs=gr.Textbox(label="Advice"),  
    title="Career Advisor",  
    description="Enter the position you're applying for, paste the job description, and your resume content to get advice on what to improve for getting this job."  
)  

career_advice_app.launch()  
