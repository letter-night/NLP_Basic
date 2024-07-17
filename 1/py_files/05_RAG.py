#### ==================== 1.  Vector DB ======================== ####
import re
import chromadb
import pandas as pd
import seaborn as sns
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pprint import pprint

# max length
# def max_word_count(txt_list:list):
# 	max_length = 0
# 	for txt in txt_list:
# 		word_count = len(re.findall(r'\w+', txt))
# 		if word_count > max_length:
# 			max_length = word_count 
# 	return f"Max Word Count: {max_length} words"

# Sentence splitter
# chroma default sentence model "all-MiniLM-L6-v2"
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# max input length: 256 characters
# model_max_chunk_length = 256
# token_splitter = SentenceTransformersTokenTextSplitter(
# 	tokens_per_chunk=model_max_chunk_length,
# 	model_name="all-MiniLM-L6-v2",
# 	chunk_overlap=0
# )

# Data Import 
# text_path = "movies.csv"
# df_movies_raw = pd.read_csv(text_path, parse_dates=["release_date"])

# # print(df_movies_raw.head(2))

# # filter movies for missing title or overview
# selected_cols = ['id', 'title', 'overview', 'vote_average', 'release_date']
# df_movies_filt = df_movies_raw[selected_cols].dropna()
# # filter for unique ids
# df_movies_filt = df_movies_filt.drop_duplicates(subset=['id'])

# # filter for movies after 01.01.2023
# df_movies_filt = df_movies_filt[df_movies_filt['release_date'] > '2023-01-01']
# print(df_movies_filt.shape) 

# max_word_count(df_movies_filt['overview'])

# embedding_fn = SentenceTransformerEmbeddingFunction()

# chroma_db = chromadb.PersistentClient(path='db')

# chroma_db.list_collections()

# Get / Create Collection
# chroma_collection = chroma_db.get_or_create_collection("movies", embedding_function=embedding_fn)

# # add all tokens to collection
# ids = [str(i) for i in df_movies_filt['id'].tolist()]
# documents = df_movies_filt['overview'].tolist()
# titles = df_movies_filt['title'].tolist()
# metadatas = [{'source':title} for title in titles]

# chroma_collection.add(documents=documents, ids=ids, metadatas=metadatas)

# print(len(chroma_collection.get()['ids']))

# Run a Query
# def get_title_by_description(query_text:str):
# 	n_results = 3
# 	chroma_collection = chroma_db.get_collection("movies")
# 	res = chroma_collection.query(query_texts=[query_text], n_results=n_results)
# 	for i in range(n_results):
# 		pprint(f"Title: {res['metadatas'][0][i]['source']} \n")
# 		pprint(f"Description: {res['documents'][0][i]} \n")
# 		pprint("-----------------------------------------")

# get_title_by_description(query_text="monster, underwater")

#### ==================== 2.  RAG ======================== ####
import openai 
from openai import OpenAI

chroma_db = chromadb.PersistentClient(path='db')
chroma_db.list_collections()
# Get / Create Collection
chroma_collection = chroma_db.get_or_create_collection("movies")

# res = chroma_collection.query(query_texts=['a monster in closet'], n_results=4)
# print(res)

# count of documents in collection
# print(len(chroma_collection.get()['ids']))

# Run a Query
def get_query_results(query_text:str, n_results:int=4):
	res = chroma_collection.query(query_texts=[query_text], n_results=n_results)
	docs = res['documents'][0]
	titles = [item['source'] for item in res['metadatas'][0]]
	res_string = ';'.join(f'{title}: {description}' for title, description in zip(titles, docs))
	return res_string 

# query_text = "a monster in the closet"
# retrieved_results = get_query_results(query_text)

# RAG
# system_role_definition = "You are an expert in movies. Users will ask you questions about movies. \
# You will get a user question, and relevant information. Relevant information is structured like \
# movie title:movie plot; ... Please answer the question only using the information provided."

# user_query = "What are the names of the movies and their plot where {user_query}?"

# messages = [
# 	{"role":"system", "content":system_role_definition},
# 	{"role":"user", "content":f"{user_query}; \n Information: {retrieved_results}"}
# ]

# openai_client = OpenAI()
# model = "gpt-3.5-turbo"
# response = openai_client.chat.completions.create(
# 	model=model, messages=messages
# )
# content = response.choices[0].message.content 

# print(content)

# RAG Function
def rag(user_query: str):
	retrieved_results = get_query_results(user_query)

	system_role_definition = "You are an expert in movies. Users will ask you questions about movies. \
You will get a user question, and relevant information. Relevant information is structured like \
movie title:movie plot; ... Please answer the question only using the information provided."

	user_query_complete = f"What are the names of the movies and their plot where {retrieved_results}?"

	messages = [
		{"role":"system", "content":system_role_definition},
		{"role":"user", "content":f"{user_query_complete}; \n Information: {retrieved_results}"}
	]
	openai_client = OpenAI()
	model = "gpt-3.5-turbo"
	response = openai_client.chat.completions.create(
		model=model, messages=messages
	)
	content = response.choices[0].message.content 
	return content 

# Response from Vector DB
print("Response from Vector DB")
print("-" * 90)
query = "a cop is chasing a criminal"
pprint(get_query_results(query))

# Response from RAG
print("Response from RAG")
print("-" * 90)
pprint(rag(query))
