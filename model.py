#from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import tkinter
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
#import chainlit as cl
#import pdb

db_path = 'vectorstores/'


#embedding_model = 'sentence-transformers/all-mpnet-base-v2'
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'

# MODEL = 'llama-2-13b-chat.ggmlv3.q8_0.bin'
MODEL = "D:\\Calsoft Projects\\Llama2\\app1\\llama-2-7b-chat.ggmlv3.q8_0.bin"

# custom_prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# <Instruction>: You are a Support executive for "VMware vSphere 8.0 with NSX" and your job is to help providing the best Product answer. 
# Use only information in the following paragraphs to answer the question at the end. Explain the answer with reference to these paragraphs. 
# If you don't know, say that you do not know.

# If it contains a sequence of instruction, rewrite those instructions in following format:
# step 1 - ...
# step 2 - ...
# ...
# step N - ...

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

custom_prompt_template = """You are virtual assistent of a law firm. 
Users will provide their case details.
Your responsibility is find similar case from following pieces of an information.
And predict the case rating for the given case and also provide details like Case Type, Case State, Handeling Firm that handles similar kind of case and won it.

Use the following pieces of information to classify.

Context: {context}

Classify the case as per "Case Rating", Provide output in following format:
    Case Type:
    Case Rating:
    Case State:
    Handling Firm:
        
If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 5}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = MODEL,
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.0
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                       model_kwargs={'device': 'cpu'})
    
    db = FAISS.load_local(db_path, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    print (response)
    return response


def chatbot_response(question):
    print(question)
    result = final_result(question)
    print(result['result'])
    print("----------------------------------------------------------")
    # for i in range(len(result["source_documents"])):
    #     print("----------------------------------------.------------------")
        # print("Document : ", i+1, "\n",  result["source_documents"][i])
    #     print("Document : ", i+1, "\n",  result["source_documents"][i].page_content)
    #     print("Source : ", result["source_documents"][i].metadata['source'].rsplit("\\", 1)[1])
    #     print("Page : ", result["source_documents"][i].metadata['page'] + 1)
    # print("----------------------------------------------------------")
    # print("*********************************************************")
    return (result)
    # input_ids = tokenizer(text, return_tensors="pt").input_ids
    # generated_ids = model.generate(input_ids, max_length=1024)
    # return(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",tkinter.END)

    # Creating a tuple containing 
    # the specifications of the font.
    Font_tuple = ("Arial", 8)
      
    # Parsed the specifications to the
    # Text widget using .configure( ) method.
    ChatLog.configure(font = Font_tuple, foreground="black", )


    if msg != '':
        ChatLog.config(state=tkinter.NORMAL)
        ChatLog.insert(tkinter.END, "Query : " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 10 ))

        res = chatbot_response(msg)
        ChatLog.insert(tkinter.END, "Response : " + res['result'] + '\n\n')
        # ChatLog.insert(tkinter.END, '===================================================\n\n')
        for i in range(len(res["source_documents"])):
            ChatLog.insert(tkinter.END, '------------------------------------------\n')
            ChatLog.insert(tkinter.END, "Document : " + res["source_documents"][i].metadata['source'].rsplit("\\", 1)[1]  + '\n')
            ChatLog.insert(tkinter.END, "Page No. : " + str(res["source_documents"][i].metadata['page'] + 1)  + '\n')

        ChatLog.insert(tkinter.END, '===================================================\n\n')

        ChatLog.config(state=tkinter.DISABLED)
        ChatLog.yview(tkinter.END)


if __name__ == "__main__":
    base = tkinter.Toplevel()
    base.title("Generative AI Powered Virtual Assistant for - HR !!!")
    
    # pack is used to show the object in the window
    #photo_image = tkinter.PhotoImage(file="C:\Cloudmoyo\Terracon\Terracon_1.png", master=base)
    photo_image = tkinter.PhotoImage(file="D:\\Calsoft Projects\\Llama2\\app1\\img.png", master=base)
    label = tkinter.Label(base, image=photo_image)
    label_1 = tkinter.Label(base, text="Enter your query here and click 'Ask':", font=("Arial", 12))
    
    base.geometry("700x700")
    
    # Add image file
    base.resizable(width=tkinter.FALSE, height=tkinter.FALSE)
    
    #Create Chat window
    ChatLog = tkinter.Text(base, bd=2, bg = "white", height="8", width="50", font=("Comic Sans MS", 14), foreground="red", relief=tkinter.SUNKEN )
    
    # ChatLog.insert(tkinter.END, "Hi, I am your Virtual Assistant to help you with your queries.\n")
    # ChatLog.insert(tkinter.END, "Please ask me your query and Click 'Ask'.\n")
    # ChatLog.insert(tkinter.END, '===================================================\n\n')
    
    ChatLog.config(state=tkinter.DISABLED)
    
    #Bind scrollbar to Chat window
    scrollbar = tkinter.Scrollbar(base, command=ChatLog.yview, bd=2, cursor="fleur")
    ChatLog['yscrollcommand'] = scrollbar.set
    
    #Create Button to send message
    #SendButton = tkinter.Button(base, font=("Verdana",12,'bold'), text="Classify", width="8", height=5,
    #                    bd=2, bg="#951307", activebackground="#951307",fg='#ffffff',
    #                    command= send )
    SendButton = tkinter.Button(base, font=("Verdana",10,'bold'), text="Ask", width="8", height=5,
                        bd=2, bg="#8c939f", activebackground="#565e68",fg='#222222',
                        command= send )
    
    
    #Create the box to enter message
    EntryBox = tkinter.Text(base, bd=2, bg="white",width="29", height="5", font="Arial")
    
    
    #Place all components on the screen
    scrollbar.place(x=676,y=210, height=360)
    label.place(x=6, y=0, height=200, width=688)
    ChatLog.place(x=6,y=210, height=360, width=670)
    label_1.place(x=0, y=575, height=20, width=350)
    EntryBox.place(x=6, y=601, height=70, width=570)
    SendButton.place(x=587, y=601, height=70, width=90)
    
    base.mainloop()




