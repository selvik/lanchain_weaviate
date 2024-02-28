import locale

locale.getpreferredencoding = lambda: "UTF-8"

# Reference: https://github.com/tomasonjo/blogs/blob/master/weaviate/HubermanWeaviate.ipynb
# Blog: https://bratanic-tomaz.medium.com/how-to-implement-weaviate-rag-applications-with-local-llms-and-embedding-models-24a9128eaf84

# import dependencies
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from langchain.text_splitter import TokenTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Weaviate
import weaviate

WEAVIATE_URL=""
WEAVIATE_API_KEY=""

weaviate_client = weaviate.Client(
    url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)


# "all-mpnet-base-v2" is a sentence transformer model, it maps sentences and paras
# to a 768-dimensional vector space
# Ref: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name, model_kwargs=model_kwargs
)

##############################
# Data collection and preprocessing 

import requests
import xml.etree.ElementTree as ET

URL = "https://www.youtube.com/feeds/videos.xml?channel_id=UC2D2CMWXMOVWx7giW1n3LIg"

response = requests.get(URL)
xml_data = response.content

# Parse the XML data
root = ET.fromstring(xml_data)

# Define the namespace
namespaces = {
    "atom": "http://www.w3.org/2005/Atom",
    "media": "http://search.yahoo.com/mrss/",
}

# Extract YouTube links
youtube_links = [
    link.get("href")
    for link in root.findall(".//atom:link[@rel='alternate']", namespaces)
][1:]

print ("youtube links:", youtube_links)


###################################################################
# Process youtube links

from langchain_community.document_loaders import YoutubeLoader

all_docs = []
for link in youtube_links:
    loader = YoutubeLoader.from_youtube_url(link)
    docs = loader.load()
    all_docs.extend(docs)
text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)
split_docs = text_splitter.split_documents(all_docs)

###################################################################
#create a weaviate vector store and seed it with docs
# where,
# Embeddings can be OpenAIEmbeddings or HuggingFaceEmbeddings.
weaviate_db = Weaviate.from_documents(
    split_docs, embeddings, client=weaviate_client, by_text=False
)


###################################################################
# model inferencing

print(
    weaviate_db.similarity_search(
        "Which are tools to bolster your mental health?", k=3)
    )


###############@@@@@@@@@@@@@@@@@@##########
# Install and use a local LLM

# specify model huggingface mode name
model_name = "anakin87/zephyr-7b-alpha-sharded"

# function for loading 4-bit quantized model
def load_quantized_model(model_name: str):
    """
    :param model_name: Name or path of the model to be loaded.
    :return: Loaded quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    return model

# function for initializing tokenizer
def initialize_tokenizer(model_name: str):
    """
    Initialize the tokenizer with the specified model_name.

    :param model_name: Name or path of the model for tokenizer initialization.
    :return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer


# initialize tokenizer
tokenizer = initialize_tokenizer(model_name)
# load model
model = load_quantized_model(model_name)
# specify stop token ids
stop_token_ids = [0]


# build huggingface pipeline for using zephyr-7b-alpha
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=2048,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# specify the llm
llm = HuggingFacePipeline(pipeline=pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=weaviate_db.as_retriever()
)

response = qa_chain.run(
    "How does one increase their mental health?")
print(response)


response = qa_chain.run("How to increase your willpower?")
print(response)
