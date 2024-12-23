# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI

# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback

import PyPDF2
import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

myOpenAIapiKEY = os.getenv("OPENAI_API_KEY")
print(myOpenAIapiKEY)

# if not myOpenAIapiKEY:
#     logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
# else:
#     logger.info("OpenAI API key loaded successfully.")

llm = ChatOpenAI(openai_api_key=myOpenAIapiKEY, model_name="gpt-3.5-turbo", temperature=0.7)

with open("D:\\Lab Setup\\MCQ Generator\\genAI\\response.json", 'r') as file:
    RESPONSE_JSON = json.load(file)

print(RESPONSE_JSON)

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{RESPONSE_JSON}

"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
    template=TEMPLATE
    )

quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)

review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
                                        output_variables=["quiz", "review"], verbose=True,)

PATH = "D:/Lab Setup/MCQ Generator/genAI/data.txt"

with open(PATH, 'r') as f:
    TEXT = f.read()

print(TEXT)

NUMBER=5
SUBJECT="History"
TONE="simple"

with get_openai_callback() as cb:
    response=generate_evaluate_chain(
        {
            "text": TEXT,
            "number": NUMBER,
            "subject":SUBJECT,
            "tone": TONE,
            "RESPONSE_JSON": json.dumps(RESPONSE_JSON)
        }
    )

print(response)

quiz = response.get("quiz")
quiz=json.loads(quiz)
print(quiz)

quiz_table_data = []
for key, value in quiz.items():
    mcq = value["mcq"]
    options = " | ".join(
        [
            f"{option}: {option_value}"
            for option, option_value in value["options"].items()
            ]
        )
    correct = value["correct"]
    quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

quiz_table = pd.DataFrame(quiz_table_data)

print(quiz_table)