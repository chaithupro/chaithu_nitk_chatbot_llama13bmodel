import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import requests
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Your existing code for loading the models and initializing the chatbot
# ...


# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "r8_3wm8BeDRHTUCljxnqk0ps3j78JSR6MD3Kuaqy"

# Initialize Pinecone
pinecone.init(api_key='c4d26737-a8ab-4842-b42f-b3178907e5ef', environment='gcp-starter')


# Load and preprocess the PDF document
loader = PyPDFLoader('/home/chaithu/NLP Project Files/try_final/copy_data_set.pdf')
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Use HuggingFace embeddings for transforming text into numerical vectors
embeddings = HuggingFaceEmbeddings()

# Set up the Pinecone vector database
index_name = "chatbotversion2"
index = pinecone.Index(index_name)
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# Initialize Replicate Llama2 Model
model_path = "llama2_model"
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 3000}
)

# Save the model

# llm.save("/content/save/model.json")

# Load the model (if needed)
# llm.load(model_path)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

# Hugging Face API for LaMini-T5-61M
API_URL = "https://api-inference.huggingface.co/models/MBZUAI/LaMini-T5-61M"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Start chatting with the chatbot
chat_history = []

@app.route('/')
def home():
    return render_template('index.html')




# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.form['user_input']

#     # Use the ConversationalRetrievalChain for answering queries
#     result = qa_chain({'question': user_input, 'chat_history': chat_history})

#     # Additional interaction with the Hugging Face model LaMini-T5-61M
#     payload = {"inputs": user_input}
#     result_hf = query(payload)

#     response = {
#         'qa_chain': result['answer'],
#         'hugging_face': result_hf.get('answer', 'No answer from Hugging Face Model')
#     }

#     chat_history.append((user_input, result['answer']))
#     return jsonify(response)




# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.form['user_input']

#     # Use the ConversationalRetrievalChain for answering queries
#     result = qa_chain({'question': user_input, 'chat_history': chat_history})

#     # Additional interaction with the Hugging Face model LaMini-T5-61M
#     payload = {"inputs": user_input}
#     result_hf = query(payload)

#     # Updated response structure with only 'answer' from qa_chain
#     response = {
#         'answer': result['answer'],
#         'hugging_face': result_hf.get('answer', 'No answer from Hugging Face Model')
#     }

#     chat_history.append((user_input, result['answer']))
#     return jsonify(response)



from sys import exit

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.form['user_input']

#     # Check if the user wants to quit
#     if user_input.lower() in ["exit", "quit", "q"]:
#         print('Exiting')
#         exit()

#     # Use the ConversationalRetrievalChain for answering queries
#     result = qa_chain({'question': user_input, 'chat_history': chat_history})

#     # Additional interaction with the Hugging Face model LaMini-T5-61M
#     payload = {"inputs": user_input}
#     result_hf = query(payload)

#     # Updated response structure
#     response = {
#         'qa_chain': result['answer'],
#         'hugging_face': result_hf.get('answer', 'No answer from Hugging Face Model')
#     }

#     chat_history.append((user_input, result['answer']))
#     return jsonify(response)





@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']

    # Check if the user wants to quit
    if user_input.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        exit()

    # Use the ConversationalRetrievalChain for answering queries
    result = qa_chain({'question': user_input, 'chat_history': chat_history})

    # Additional interaction with the Hugging Face model LaMini-T5-61M
    payload = {"inputs": user_input}
    result_hf = query(payload)

    # Check if 'answer' is present in the list
    hugging_face_answer = next((item['answer'] for item in result_hf if 'answer' in item), 'No answer from Hugging Face Model')

    # Updated response structure
    response = {
        'qa_chain': result['answer'],
        'hugging_face': hugging_face_answer
    }

    chat_history.append((user_input, result['answer']))
    return jsonify(response)








if __name__ == '__main__':
    app.run(debug=True)
