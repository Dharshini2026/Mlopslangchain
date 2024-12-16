from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model_name = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# # Load LLaMA Model from Hugging Face
# model_name = "meta-llama/LLaMA-2-7b-chat-hf"  # LLaMA-2 model, you can use a different version if needed

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a HuggingFace pipeline for text generation
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Use HuggingFacePipeline in LangChain
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a legal assistant. Please respond to the user's legal questions based on the context provided:
    Context: {context}
    Question: {question}
    """
)

# Memory for maintaining conversation context
memory = ConversationBufferMemory(memory_key="chat_history")

# Create an LLM Chain
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory
)

# Function to Get Response from LLaMA Model
def get_legal_response(context, question):
    response = llm_chain.run(context=context, question=question)
    return response

# Example Usage
if __name__ == "__main__":
    context = "Laws regarding tenant rights in California"
    question = "What protections are provided to tenants under California law?"
    response = get_legal_response(context, question)
    print(response)
