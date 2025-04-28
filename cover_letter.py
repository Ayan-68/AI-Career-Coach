from ibm_watsonx_ai import Credentials  
from ibm_watsonx_ai import APIClient  
from ibm_watsonx_ai.foundation_models import Model, ModelInference  
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters  
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames  
import gradio as gr  

model_id = "meta-llama/llama-3-2-11b-vision-instruct"  
credentials = Credentials(  
    url=" ",  # Your Credentials
)  
params = TextChatParameters(  
    temperature=0.7,  
    max_tokens=512  
)  
project_id = " "  #Your Project ID

model = ModelInference(  
    model_id=model_id,  
    credentials=credentials,  
    project_id=project_id,  
    params=params  
)  

def generate_cover_letter(company_name, position_name, job_description, resume_content):  
    prompt = f"Generate a customized cover letter using the company name: {company_name}, the position applied for: {position_name}, and the job description: {job_description}. Ensure the cover letter highlights my qualifications and experience as detailed in the resume content: {resume_content}. Adapt the content carefully to avoid including experiences not present in my resume but mentioned in the job description. The goal is to emphasize the alignment between my existing skills and the requirements of the role."  
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
    cover_letter = generated_response['choices'][0]['message']['content']  
    return cover_letter  

cover_letter_app = gr.Interface(  
    fn=generate_cover_letter,  
    flagging_mode="never",  
    inputs=[  
        gr.Textbox(label="Company Name", placeholder="Enter the name of the company..."),  
        gr.Textbox(label="Position Name", placeholder="Enter the name of the position..."),  
        gr.Textbox(label="Job Description Information", placeholder="Paste the job description here...", lines=10),  
        gr.Textbox(label="Resume Content", placeholder="Paste your resume content here...", lines=10),  
    ],  
    outputs=gr.Textbox(label="Customized Cover Letter"),  
    title="Customized Cover Letter Generator",  
    description="Generate a customized cover letter by entering the company name, position name, job description and your resume."  
)  

cover_letter_app.launch()  
