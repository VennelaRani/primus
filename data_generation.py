import requests
import pandas as pd
import streamlit as st
import os
import re
# Define the API URL for the locally running Llama model
API_URL = "http://localhost:11434/api/chat"

# Load the CSV data containing tweets
csv_path = r"/root/twitter_codes_vennela/tweet_updated.csv"
df = pd.read_csv(csv_path)
st.write("welcome")

# Define output CSV file
output_csv_path = r"/root/twitter_codes_vennela/generated_crypto_questions_answers_with_types.csv"

# Check if file exists (to avoid rewriting headers)
file_exists = os.path.isfile(output_csv_path)

# Define a function to interact with the Llama API

def remove_think_tags(text):
    """Remove content between <think> and </think> tags from the text."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def get_llama_response(prompt,format):

    """Send a request to the Llama API and return the response."""
    st.write("format ",format)
    st.write("Prompt:", prompt)
    
    payload = {
        "model": "deepseek-r1:8b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json().get("message", {}).get("content", "").strip()
        result = remove_think_tags(result)
        st.write("Response:", result)
        return result
    else:
        st.write(f"Error: {response.status_code}, {response.text}")
        return None

# Define a function to generate crypto-related questions and answers
def generate_question_and_answer(tweet):
    """Generate different types of questions and answers for a tweet."""
    global file_exists  # Declare file_exists as global before using it

    question_types = [
        "Token Analysis",
        "Sarcastic Reply",
        "Casual Web3 Communication",
        "Price Fluctuations",
        "Technical Data"
    ]

    for question_type in question_types:
        # Generate a question
        question_prompt = f"Generate a '{question_type}' question based on this tweet: '{tweet}' in Twitter language strictly in one line only."
        question = get_llama_response(question_prompt,format='question')

        # Generate an answer for the generated question
        answer_prompt = f"Answer the following '{question_type}' question based on this tweet: '{tweet}' Question: '{question}' in Twitter language strictly in one line only."
        answer = get_llama_response(answer_prompt,format='answer')

        # Store result in a DataFrame
        result_df = pd.DataFrame([{
            "Crypto Query": question,
            "Answer": answer
        }])

        # Append to CSV file immediately
        result_df.to_csv(output_csv_path, mode='a', header=not file_exists, index=False)
        
        # Ensure headers are not rewritten after first write
        file_exists = True  

        st.write("Saved to CSV:", result_df)

# Loop through all tweets and process each one
for tweet in df.iloc[415:]['cleaned_tweet']:
    generate_question_and_answer(tweet)

st.write(f"Generated questions and answers saved to: {output_csv_path}")
