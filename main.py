import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from accelerate import Accelerator
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter

import nest_asyncio, logging, asyncio
from telegram.helpers import escape_markdown
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, ApplicationBuilder, ContextTypes 
from langchain.document_loaders import PyPDFLoader

accelerator = Accelerator()

# Define the RAG model and embeddings functions
def RAG_model(model_path="Qwen/Qwen2-0.5B-Instruct"):
    device = accelerator.device  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device=device,
        pad_token_id=0
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def embeddings(modelPath="sentence-transformers/all-MiniLM-L12-v2"):
    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': accelerator.device}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    return HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

# Step 2: Retrieve Relevant Information
def retrieve_context(query, top_k=3):
    # Initialize Qdrant client with on-disk storage
    with Qdrant.from_existing_collection(path="Qdrant_VDB", collection_name="CS_doc", embedding=embeddings()) as vector_store:
        """Retrieve the most relevant documents for a given query."""
        docs = vector_store.similarity_search(query, k=top_k)
        context = " ".join([doc.page_content for doc in docs])
    return context

# Load and process Documents sent to users in PDF Format
def load_and_process_documents(file_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split(text_splitter=text_splitter)
    for page in pages:
        page.page_content = page.page_content.replace('\n',' ')
    return pages

# Taking in the responses and models
def generate_response(model, tokenizer, text):
    model_inputs = tokenizer(text, return_tensors="pt").to("cpu")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id  # To avoid potential padding issues
    )
    
    return generated_ids

# Get question and answer (Main Function)
async def ask_question(user_query):
    prompt_template = """
    Your main role is to answer questions from the user. You are an assistant specializing in computer science principles and coding.
    Retrieve relevant information from the dataset and utilize inference and suggestions for the following tasks:
    - Responses should cover fundamental principles of computer science.
    - Inferences are allowed to provide comprehensive answers.
    - Use the provided context to list down relevant information and explanations.
    - Ensure all responses are accurate and aligned with computer science topics.
    Ensure responses are derived from the dataset, use inference and suggestions to provide comprehensive answers.
    """
    start_time = datetime.now()
    # Retrieve relevant context
    context = retrieve_context(user_query)
    model, tokenizer = RAG_model()

    # Prepare the prompt with context
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Context: {context}\n\n{user_query}"}
    ]

    # Concatenate the messages into a single string for the model
    text = "\n".join([f"{message['role']}: {message['content']}" for message in messages])

    # Tokenize and generate response
    generated_ids = generate_response(model, tokenizer, text)

    # Decode the generated response
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract the response after the user query
    cleaned_response = clean_response(generated_text)
    end_time = datetime.now()
    elapsed_time = (start_time - end_time).total_seconds

    return cleaned_response, elapsed_time

# Clean the responses
def clean_response(generated_text):
    response_start = generated_text.find("Answer:")
    if response_start != -1:
        cleaned_response = generated_text[response_start + len("Answer:"):].strip()
    else:
        cleaned_response = generated_text.strip()

    cleaned_response = "\n\n".join([line.strip() for line in cleaned_response.split("\n\n") if line.strip()])
    return cleaned_response


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /start command"""
    user_id = update.effective_user.id
    if user_id not in context.bot_data:
        context.bot_data[user_id] = {}
    await update.message.reply_text("Hi!\nI'm your AI assistant. Ask me any question about computer science and coding!")

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for receiving PDF documents"""
    user_id = update.effective_user.id
    
     # Initialize user-specific data if it doesn't exist
    if user_id not in context.bot_data:
        context.bot_data[user_id] = {}
        
    document = update.message.document
    if document.mime_type == 'application/pdf':
        file_id = document.file_id
        new_file = await context.bot.get_file(file_id)
        file_path = f"{file_id}.pdf"
        await new_file.download_to_drive(file_path)
        
        pages = load_and_process_documents(file_path)
        if 'vectordb' not in context.bot_data[user_id]:
            vectordb = Qdrant.from_documents(pages, embeddings)
            context.bot_data[user_id]['vectordb'] = vectordb
        else:
            vectordb = context.bot_data[user_id]['vectordb']
            vectordb.add_documents(pages)
        
        await update.message.reply_text('PDF document received and processed. You can now ask questions about the content.')
    else:
        await update.message.reply_text(f"Unsupported file type: {document.mime_type}. Skipping this file.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for answering questions based on the processed documents"""
    user_id = update.effective_user.id
    question = update.message.text
    response, duration = await ask_question(question)
    print(f"{user_id} => Query:{question}\nResponse:{response}\nDuration:{duration:.2f}")
    await context.bot.send_message(chat_id=update.effective_chat.id, text=escape_markdown(response))

async def main():
    """Main function to run the bot"""
 
    application = ApplicationBuilder().token(API_TOKEN).build()

    # Register command and message handlers
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, question_handler))

    # Start the bot
    await application.run_polling()
   # application.idle()

if __name__ == '__main__':
    main()
    