"""
@author: Shubham Kumari

"""


import os
import logging
from logging.handlers import TimedRotatingFileHandler
from langchain_openai import AzureChatOpenAI
import json
import yaml
import openai
#from langchain_openai import AzureChatOpenAI
from firm_case_classifier_api_v8 import process_query
from azure.identity import AzureCliCredential
from azure.keyvault.secrets import SecretClient


# Configure logging with a TimedRotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        TimedRotatingFileHandler(
            filename='app.log',
            when='W0',  # Rotate logs on a weekly basis, starting on Monday
            backupCount=1  # Retain one backup log file (the current week's log)
        )
    ],
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class caseClassifier:
    def __init__(self):
        # Key Vault URL
        key_vault_url = "https://caseratekeyvault.vault.azure.net/"
        credential = AzureCliCredential()
        # Create a SecretClient using the Key Vault URL and credential
        client = SecretClient(vault_url=key_vault_url, credential=credential)

        self.db_path = client.get_secret("pl-db-path").value
        self.OPENAI_API_TYPE = client.get_secret("pl-azure-api-type").value
        self.OPENAI_API_KEY = client.get_secret("pl-open-api-key").value
        #self.OPENAI_DEPLOYMENT_VERSION = client.get_secret("pl-openai-deployment-version").value
        self.OPENAI_DEPLOYMENT_ENDPOINT = client.get_secret("pl-openai-deployment-endpoint").value
        #self.OPENAI_DEPLOYMENT_NAME = client.get_secret("pl-openai-deployment-name").value
        #self.OPENAI_MODEL_NAME = client.get_secret("pl-openai-model").value
        self.OPENAI_DEPLOYMENT_VERSION = client.get_secret("pl-openai-deployment-version-4o-mini").value
        #self.OPENAI_DEPLOYMENT_ENDPOINT = client.get_secret("pl-openai-deployment-endpoint").value
        self.OPENAI_DEPLOYMENT_NAME = client.get_secret("deployment-name-4o-mini").value
        self.OPENAI_MODEL_NAME = client.get_secret("modelname-4o-mini").value
        self.OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = client.get_secret("pl-openai-ada-enbedding-deployment").value
        self.OPENAI_ADA_EMBEDDING_MODEL_NAME = client.get_secret("pl-openai-ada-embedding-model-name").value


        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["OPENAI_DEPLOYMENT_ENDPOINT"] = self.OPENAI_DEPLOYMENT_ENDPOINT
        os.environ["OPENAI_DEPLOYMENT_NAME"] = self.OPENAI_DEPLOYMENT_NAME
        os.environ["OPENAI_MODEL_NAME"] = self.OPENAI_MODEL_NAME
        os.environ["OPENAI_DEPLOYMENT_VERSION"] = self.OPENAI_DEPLOYMENT_VERSION

        self.custom_prompt_template = """
        Please determine if the following description constitutes a legal case: "{question}"
        Your task is to analyze the provided description and indicate whether it resembles a legal case
        Consider elements such as parties involved, legal issues, relevant laws
        court proceedings, or any other factors that typically define a legal case
        If the description aligns with what you understand as a legal case,
        The Output should be strictly in JSON format and the JSON structure must have the following key values:
        "Status" : "YES or NO if description is legal say YES if not than say NO",
        "Explanation" : "Explanation for given response"

        """

    def load_llm(self):
        openai.api_type = "azure"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_version = os.getenv('OPENAI_DEPLOYMENT_VERSION')
        llm = AzureChatOpenAI(
            deployment_name=self.OPENAI_DEPLOYMENT_NAME,
            model_name=self.OPENAI_MODEL_NAME,
            azure_endpoint=self.OPENAI_DEPLOYMENT_ENDPOINT,
            openai_api_version=self.OPENAI_DEPLOYMENT_VERSION,
            openai_api_key=self.OPENAI_API_KEY,
            openai_api_type="azure"
            )

        return llm

    def get_predictions(self, prompt_hf):
        llm = self.load_llm()
        predictions = llm.predict(prompt_hf)
        return predictions

    def analyze_case(self, query):
        try:
            hf_prompt = self.custom_prompt_template.format(question=query)
            predictions = self.get_predictions(hf_prompt)
            return predictions
        except Exception as error:
            print(error)
            return None  # Unable to determine the status from LLM


def flag_check(query):
    case_classifier = caseClassifier()
    app = caseClassifierApp(case_classifier)
    response = app.send(query)
    response = get_string_between_braces(response)
    try:
        #print("*************************************************")
        #print(response)
        #print("*************************************************")
        pred = json.loads(response)
        expl = pred.get("Explanation", "").strip()
        #print(expl)
        try:
            flag = pred.get("Status", "").strip().upper() == "YES"
            if flag is True:
                logging.info("This appears to be a legal case.")
                final_result = process_query(query)
                #return final_result
            elif flag is False:
                #print ("Flag is NO")
                logging.info("This does not appear to be a legal case.")
                final_result = '''{{
                            "PrimaryCaseType": " ",
                            "SecondaryCaseType": " ",
                            "CaseRating": " ",
                            "CaseState" : " ",
                            "IsWorkersCompensation(Yes/No)?": " ",
                            "Confidence(%)": " ",
                            "Explanation": "{e}",
                            "HandlingFirm" : "Unknown"
                        }}'''.format(e=expl)
               # print (final_result)

                #return final_result
            else:
                logging.warning("Unable to determine if it's a legal case due to an unexpected response.")
                final_result = '''
                    {
                        "PrimaryCaseType": " ",
                        "SecondaryCaseType": " ",
                        "CaseRating": " ",
                        "CaseState" : " ",
                        "IsWorkersCompensation(Yes/No)?": " ",
                        "Confidence(%)": " ",
                        "Explanation": "There is some error occured while answering your question, Please try with same case description again.  Sorry for an inconvenience Caused",
                        "HandlingFirm" : "Unknown"
                    }
                    '''
                #return final_result
            return final_result
        except Exception as error:
            logging.exception("An error occurred in flag_check: %s", error)
    except Exception as error:
        logging.exception("An error occurred in flag_check: %s", error)


class caseClassifierApp:
    def __init__(self, case_classifier):
        self.case_classifier = case_classifier

    def send(self, msg):
        result = self.case_classifier.analyze_case(msg)
        return result


def get_string_between_braces(text):
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and start < end:
        return text[start:end+1]
    else:
        return None


if __name__ == "__main__":
    while True:
        query = input('you: ')
        if query == 'q':
            break
        elif query.strip() == "":
            continue
        #response = app.send(query)
        response = flag_check(query)
        logging.info('Final result generated: %s', response)
        print("response", response)
