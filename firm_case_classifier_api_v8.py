"""
@author: Shubham Kumari

"""

import os
import openai
import logging
import json
import yaml
import uuid
import sqlite3
from logging.handlers import TimedRotatingFileHandler
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from Insert_llmdata import data_base

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

        logging.info('Initializing caseClassifier...')

        # Key Vault URL
        key_vault_url = "https://caseratekeyvault.vault.azure.net/"
        # DefaultAzureCredential will handle authentication for managed identity, Azure CLI, and environment variables.
        credential = DefaultAzureCredential()
        # Create a SecretClient using the Key Vault URL and credential
        client = SecretClient(vault_url=key_vault_url, credential=credential)

        self.db_path = client.get_secret("pl-db-path").value
        self.OPENAI_API_TYPE = client.get_secret("pl-azure-api-type").value
        self.OPENAI_API_KEY = client.get_secret("pl-open-api-key").value
        self.OPENAI_DEPLOYMENT_ENDPOINT = client.get_secret("pl-openai-deployment-endpoint").value
        self.OPENAI_DEPLOYMENT_VERSION = client.get_secret("pl-openai-deployment-version-4o-mini").value
        self.OPENAI_DEPLOYMENT_NAME = client.get_secret("deployment-name-4o-mini").value
        self.OPENAI_MODEL_NAME = client.get_secret("modelname-4o-mini").value
        self.OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = client.get_secret("pl-openai-ada-enbedding-deployment").value
        self.OPENAI_ADA_EMBEDDING_MODEL_NAME = client.get_secret("pl-openai-ada-embedding-model-name").value

        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["OPENAI_DEPLOYMENT_ENDPOINT"] = self.OPENAI_DEPLOYMENT_ENDPOINT
        os.environ["OPENAI_DEPLOYMENT_NAME"] = self.OPENAI_DEPLOYMENT_NAME
        os.environ["OPENAI_MODEL_NAME"] = self.OPENAI_MODEL_NAME
        os.environ["OPENAI_DEPLOYMENT_VERSION"] = self.OPENAI_DEPLOYMENT_VERSION 
        os.environ["OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"] = self.OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME
        os.environ["OPENAI_ADA_EMBEDDING_MODEL_NAME"] = self.OPENAI_ADA_EMBEDDING_MODEL_NAME

        self.custom_prompt_template = """
        {context}
        Given the third-party descriptions and matching case types (Primary and Secondary), case ratings, case state and Handling Firm in context.
        Which type of case do you think that the following third-party description  "{question}" indicates and what would be the case rating and case state?
        if you think it indicated more than one case type than provide list of all case type you think is applicable.
        Instruction: In description CL stands for client.Examine the description considering CL as client for answers.
        Select Primary Case Type and Secondary Case Type strictly from below list only, do not make up any other case type:
        Primary Case Types:
            - Employment Law: Case addresses disputes between employers and employees, including issues like discrimination, harassment, wrongful termination, wage disputes, and workplace rights violations.
            - General Injury: Case involves claims for physical or emotional harm caused by another's negligence or intentional actions, covering accidents, slips, falls, or similar incidents.
            - Long-Term Disability: Case involves claims for benefits when an individual is unable to work due to a severe, prolonged medical condition, often against insurers or employers.
            - Mass Tort: Case involves multiple plaintiffs filing claims against a defendant for widespread harm caused by defective products, pharmaceuticals, or large-scale disasters.
            - Nursing Home: Case addresses claims of abuse, neglect, or substandard care within nursing facilities, seeking justice and compensation for harm to residents' well-being.
            - Other
            - Workers Compensation: Case involves claims by employees for benefits due to workplace injuries or illnesses, covering medical costs, lost wages, and rehabilitation without proving employer fault.
            - Workers Compensation Federal: Case involves federal employees seeking benefits for job-related injuries or illnesses, providing medical coverage, wage replacement, and rehabilitation under federal law.
            - Wrongful Death: Case involves claims by surviving family members seeking compensation when a loved one’s death is caused by another’s negligence or intentional actions.
            - Veteran Disability: case involves claims by veterans seeking disability benefits for injuries or conditions resulting from their military service, often filed against the VA (Department of Veterans Affairs).
            - Wills, Estate and Probate: Case involves the validation, execution, distribution, or dispute of a deceased person's estate and will.
            - General Class Action: Case occurs when a group of people collectively sue a defendant for common grievances, usually involving large-scale harm or violations.
            - Criminal: Case involves the prosecution of an individual or entity accused of committing a crime, with penalties such as fines, imprisonment, or other sanctions.
            - Negligent Security: Case arises when a property owner fails to provide adequate security measures, leading to harm or injury to individuals on the premises.
            - Product Liability (auto): Case involves a claim against a manufacturer or seller for injuries or damages caused by a defective or unsafe vehicle or auto part.

        Secondary Case Types:
            - Animal Incident: Case occurs when an individual is injured or harmed due to an animal’s actions, such as bites, attacks, or negligence by the owner.
            - Automobile Accident: Case involves claims for injuries, damages, or liability resulting from a collision or crash between vehicles, often involving negligence or traffic violations.
            - Construction: Case involves disputes or claims related to construction projects, such as contract breaches, negligence, safety violations, or defects in design or workmanship. 
            - Dental Malpractice: Case occurs when a dentist's negligence, improper treatment, or failure to meet the standard of care causes harm or injury to a patient.
            - Medical Malpractice: Case arises when a healthcare provider's negligence, substandard care, or errors cause injury, harm, or death to a patient.
            - Nursing Home: Case addresses claims of abuse, neglect, or substandard care within nursing facilities, seeking justice and compensation for harm to residents' well-being.
            - Police Brutality: Case involves claims of excessive force, abuse, or unlawful actions by law enforcement officers that result in injury, harm, or violation of rights.
            - Product Liability (auto): Case involves a claim against a manufacturer or seller for injuries or damages caused by a defective or unsafe vehicle or auto part.
            - Product Liability (non-auto): Case involves a claim against a manufacturer or seller for injuries or damages caused by a defective or unsafe non-automotive product.
            - Slip and Fall: Case arises when an individual is injured due to hazardous conditions, like wet floors or uneven surfaces, on someone else's property.
            - Unemployment Claims: Case involves disputes over eligibility, denial, or challenges related to an individual’s claim for unemployment benefits from a state or employer.
            - Wage & Hour: Case involves disputes over fair compensation, such as unpaid wages, overtime, misclassification, or violations of labor laws regarding work hours.
            - Veteran Disability: case involves claims by veterans seeking disability benefits for injuries or conditions resulting from their military service, often filed against the VA (Department of Veterans Affairs).
            - Wills, Estate and Probate: Case involves the validation, execution, distribution, or dispute of a deceased person's estate and will.
            - General Class Action: Case occurs when a group of people collectively sue a defendant for common grievances, usually involving large-scale harm or violations.
            - Criminal: Case involves the prosecution of an individual or entity accused of committing a crime, with penalties such as fines, imprisonment, or other sanctions.
            - Negligent Security: Case arises when a property owner fails to provide adequate security measures, leading to harm or injury to individuals on the premises.

        Case Rating is depends on severity of an injury. Tier 5 is severe/major injury while Tier 1 is minor injury.
        Case Rating for various case types is given below, use that information for case ratings:
            For Primary Case Type: "General Injury"/"Workers Compensation"/"Workers Compensation Federal":
                Secondary Case Type: Any
                - Tier 2: Sprain, strain, whiplash, contusions, bruises, medical treatment, medication, physical therapy treatment, tingling, numbing sensations
                - Tier 3: Broken bones etc. with no surgery, Injections, Concussion
                - Tier 4: Surgery or Scheduled surgery, Memory loss
                - Tier 5: Amputation of body parts other than finger or toe, Multiple Surgeries, Crush, Electrocuted, Death, Machine malfunction with severe injuries, Semitruck accident with surgery
                Note: Any accident that involves a semitruck tracks the case up 1 tier

            For Primary Case Type "Nursing Home":
                - Tier 2: Broken bones or any other injury with no surgery, Malnutrition
                - Tier 3: Surgery or Death
                - Tier 4: Stage 3 or 4 Bedsores

            For Primary Case Type: "General Injury"
                Secondary Case Type: "Animal Incident"
                - Tier 2: Bleeding, Swelling, laceration, Puncture wounds on extremities with just an antibiotic shot
                - Tier 3: If Multiple bites mentioned, Severe injuries because bites but no surgeries
                - Tier 4: Surgery Or scheduled surgery
                - Tier 5: Plastic surgery to face

            For Primary Case Type: "General Injury"/"Wrongful Death"
                Secondary Case Type: "Medical Malpractice"/"Dental Malpractice":
                - Tier 3 - Revision surgery is needed
                - Tier 4 - Multiple revision surgeries, Lasting issues as a result of the surgery or misdiagnoses
                - Tier 5 - Unexpected Death as a result of a surgery that wasn't at a high risk of death

        Please ensure that if a state is mentioned in description, it is accurately identified and give state name as per two-character Amarican standard.
        If there is no state mentioned in description,in this type of description case state should be "Unknown without adding extra character or do not make up any case state.
        Case State should be strictly in format like examples given in below list:

            -if NJ in description,given Case State is 'NJ New Jersey'
            -if PA in description,given Case State is 'PA Pennsylvania'
            -if TN in description,given Case State is 'TN Tennessee'
            -if NY in description,given Case State is 'NY New York'
            -if VA in description,given Case State is 'VA Virginia'
            -if DE in description,given Case State is 'DE Delaware'
            -if CA in description,given Case State is 'CA California'
            -if FL in description,given Case State is 'FL Florida'
            -if AL in description,given Case State is 'AL Alabama'
            -if NV in description,given Case State is 'NV Nevada'
            -if SC in description,given Case State is 'SC South Carolina'
            -if GA in description,given Case State is 'GA Georgia'
            -if OH in description,given Case State is 'OH Ohio'
            -if DC in description,given Case State is 'DC District of Columbia'
            -if MD in description,given Case State is 'MD Maryland'
            -if OK in description,given Case State is 'OK Oklahoma'
            -if MO in description,given Case State is 'MO Missouri'
            -if MI in description,given Case State is 'MI Michigan'
            -if NC in description,given Case State is 'NC North Carolina',
            -if MS in description,given Case State is 'MS Mississippi'

        Please answer with Primary Case Type, Secondary Case Type, Case Ratings,Case State and Explain your answer
        The Output should be strictly in correct JSON format without json keyword and the JSON structure must have the following key values:
        "PrimaryCaseType" : "Primary Case Type here",
        "SecondaryCaseType" : "Secondary Case Type here",
        "CaseRating" : "Case Rating here",
        "CaseState" : "Name of State here"
        "IsWorkersCompensation(Yes/No)?" : " 'Yes', If incident happed at client's workplace, else 'No' "
        "Confidence(%)" : "Confidence in %",
        "Explanation" : "Explain your answer here with detail reason behind case, why?"
        """

        self.hf_prompt_template = """
        Given all the details about case in {case_state}, where the case rating is {case_ratings} and case types are {Primary} and {Secondary},
        determine the most suitable handling firm based on the Handling Firm Rules for the given third party description: "{question}" ?

        Assign handling firm strictly according to the Handling Firm Rules provided. Do not create or suggest any other handling firms outside of these rules.

        Rule Priority:

            1. For each rule, check if both the case type and case tier match the conditions exactly.
            2. Specific rules that mention a specific case type (e.g., case type is 'Worker Compensation') take precedence over general rules with 'Any' case type.First, check rules that mention a specific case type. If no specific rule matches, then apply general rules with 'Any' case type.
            3. If a rule specifies a range for the case rating (e.g., 'Tier 1-4'), check if the case rating falls within this range.
            4. Apply the most specific rule that matches the case rating and case type.
            5. Assign the Handling Firm strictly based on the given Handling Firm Rules.
            6. Apply the rules as specified without any modifications.
            7. Check each rule for both case type and case tier, selecting the exact matching rule if available.
            8. If no exact matching rule available, return "SAD" and give proper explanation.

        If multiple handling firms are applicable, provide a list of all applicable firms.If no firm is available for the state, return "SAD".

        The Output should be strictly in JSON format without json keyword. Do not add any extra text in output and the JSON structure must have the following key values:
            "HandlingFirm" : "Recommanded Handling firm from same state for the case and considering the rules given"
            "Assignment Explanation": "Explanation for recommanding handling firm"

        Handling Firm Rules:
        For the state, the handling rules are as follows:
        """

        self.qa_prompt = self.set_custom_prompt(self.custom_prompt_template)
   
    def set_custom_prompt(self, prompt_template):
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        return prompt
    
    def load_llm(self):
        openai.api_type = "azure"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_version = os.getenv('OPENAI_DEPLOYMENT_VERSION')
        llm = AzureChatOpenAI(deployment_name=self.OPENAI_DEPLOYMENT_NAME,
                              model_name=self.OPENAI_MODEL_NAME,
                              azure_endpoint=self.OPENAI_DEPLOYMENT_ENDPOINT,
                              openai_api_version=self.OPENAI_DEPLOYMENT_VERSION,
                              openai_api_key=self.OPENAI_API_KEY,
                              openai_api_type="azure")

        return llm
    
    def qa_bot(self, prompt):
        embeddings = AzureOpenAIEmbeddings(deployment=self.OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                           model=self.OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                           azure_endpoint=self.OPENAI_DEPLOYMENT_ENDPOINT,
                                           openai_api_type="azure",
                                           chunk_size=1,)
                                      
        db = FAISS.load_local(self.db_path, embeddings,allow_dangerous_deserialization=True)
        llm = self.load_llm()
        qa_chain = self.retrieval_qa_chain(llm, prompt, db)
        return qa_chain

    def hf_bot(self, prompt_hf):
        llm = self.load_llm()
        predictions = llm.predict(prompt_hf)
        return predictions
    
    @staticmethod
    def retrieval_qa_chain(llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 10}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
            )
        return qa_chain
   
    def final_result(self, query):
        try:
            logging.info('Generating final result for query: %s', query)
            qa_result = self.qa_bot(self.qa_prompt)
            response = qa_result({'query': query})
            return response["result"]
        except Exception as error:
            logging.error('An error occurred while generating final result: %s', error)
            response = '''
            {
                "PrimaryCaseType": " ",
                "SecondaryCaseType": " ",
                "CaseRating": " ",
                "CaseState" : " ",
                "IsWorkersCompensation(Yes/No)?": " ",
                "Confidence(%)": " ",
                "Explanation": "There is some error occured while answering your question, Please try with same case description again.  Sorry for an inconvenience Caused"
            }
            '''
            return (response)

    def get_hadling_firm(self, query, qa_result):
        
        """
        Retrieves the handling firm recommendation based on the given query and QA result.
        Args:
            query (str): The case description query.
            qa_result (dict): The result of the question answering process containing case details.
        Returns:
            str: A JSON-formatted string representing the handling firm recommendation.
        Raises:
            IOError: If there is an issue reading the firm rules file.
        """
        # Extract case details from the QA result
        primary_case_type = qa_result.get("PrimaryCaseType")
        secondary_case_type = qa_result.get("SecondaryCaseType")
        case_ratings = qa_result.get("CaseRating")
        case_state = qa_result.get("CaseState")
        # Prepare the path to the firm rules JSON file
        path = "firm_rules.json"
        try:
            # Extract the state abbreviation from the case state
            if " " in case_state:
                state_parts = case_state.split(" ", 1)
                if not state_parts[0].isupper():
                    case_state = state_parts[0] + " " + state_parts[1]
                else:
                    case_state = state_parts[1]
            else:
                case_state = case_state

            if case_state == "District of Columbia":
                case_state = "Washington DC"

            # Load firm rules from JSON file
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Check if rules exist for the given state
            if case_state in data["rules"]:
                # Retrieve rules for the given state
                state_rules = data["rules"][case_state]
               
                # Generate the handling firm prompt template based on rules
                try:
                    for rule in state_rules:
                        self.hf_prompt_template += f"  - If the case rating is '{rule['condition']['case_rating']}' and case type is '{rule['condition']['case_type']}', {rule['action']}\n"               
                    # Format the handling firm prompt with case details
                    hf_prompt = self.hf_prompt_template.format(
                        case_state=case_state,
                        case_ratings=case_ratings,
                        Primary=primary_case_type,
                        Secondary=secondary_case_type,
                        question=query
                    )
                    # Get the handling firm recommendation
                    hf_result = self.hf_bot(hf_prompt)
                except IOError as e:
                    # Log an error if there is an issue creating handling firm rules prompt
                    logging.error('An error occurred creating handling firm rules prompt: %s', e)
                    hf_result = '''
                    {
                        "HandlingFirm" : "SAD"
                    }
                    '''
                    #return hf_result
            else:
                # Set case state to unknown if rules are not found
                qa_result["CaseState"] = "Unknown"
                hf_result = '''
                {
                    "HandlingFirm" : "SAD"
                }
                '''
            return hf_result
        except IOError as e:
            # Log an error if there is an issue reading the firm rules file
            logging.error('An error occurred while reading firm rules file: %s', e)
            hf_result = '''
            {
                "HandlingFirm" : "SAD"
            }
            '''
            return hf_result


class caseClassifierApp:
    def __init__(self, case_classifier):
        self.case_classifier = case_classifier

    def send(self, msg):
        result = self.case_classifier.final_result(msg)
        return result

    def hf_send(self, msg, qa_result):
        result = self.case_classifier.get_hadling_firm(msg, qa_result)
        return result


def get_string_between_braces(text):
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and start < end:
        return text[start:end+1]
    else:
        return None


def process_query(query):
    # Initialize dictionaries to store results
    qa_result = {}
    final_result = {}
    # Initialize case classifier and application
    case_classifier = caseClassifier()
    app = caseClassifierApp(case_classifier)
    generated_uuid = uuid.uuid4()
    # Send query to the application and handle response
    response = app.send(query)
#    print(response)
#    response = response.replace('json', '')
#    response = get_string_between_braces(response)
#    print(response)
    try:
        qa_result = json.loads(response)
    except Exception as error:
        logging.exception("Exception occurred in process_query: %s", error)
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
            "CaseId" : " "
        }
        '''
        return final_result
    hf_response = app.hf_send(query, qa_result)
    try:
        firm_response = json.loads(hf_response)
        qa_result["HandlingFirm"] = firm_response["HandlingFirm"]
        qa_result["Explanation"] = qa_result["Explanation"] + "\n\n" + firm_response["Assignment Explanation"]
        qa_result["CaseId"] = str(generated_uuid)
        final_result = json.dumps(qa_result)
    except Exception as error:
        logging.error('An error occurred while processing handling firm response: %s', error)
        qa_result["HandlingFirm"] = "SAD"
        qa_result["CaseId"] = str(generated_uuid)
        final_result = json.dumps(qa_result)

    logging.info('Final result generated: %s', final_result)
    data_base(qa_result, query)
    return final_result

# if __name__ == "__main__":
#     while True:
#         logging.info("Please enter incorrect address here or type 'q' to quit")
#         query = input('you: ')
#         if query == 'q':
#             break
#         elif query.strip() == "":
#             continue
#         qa_result = process_query(query)
#         print("qa_result:", qa_result)
