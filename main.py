
# this will only bring 10 mins of all what we got , dosent waste much apis
import tweepy
from pymongo import MongoClient
import requests
import json
from datetime import datetime
import os
from typing import Dict, List, Tuple
import asyncio
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
from dateutil.parser import parse

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

from openai import OpenAI 
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import logging
import warnings
import requests
from datetime import datetime
import difflib
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
import asyncio

import os
import logging
from transformers.utils.logging import set_verbosity_error
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
load_dotenv()

# Suppress all unnecessary logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
os.environ["BITSANDBYTES_NOWELCOME"] = "1"  # Suppress BitsAndBytes warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # Suppress Hugging Face warnings

# Suppress Hugging Face and PyTorch logs
set_verbosity_error()  # Suppress all Hugging Face model loading logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)




import torch
import logging
import transformers

# Suppress PyTorch and Transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load environment variables

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stopwords
STOPWORDS = set(stopwords.words('english'))
# Configure warnings and logging
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


# class Config:
#     def __init__(self):
#         self.model_path = "./crypto_bot_model1"
#         self.quant_config = BitsAndBytesConfig(load_in_4bit=True)
#         self.llm_model = "llama3.3"
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
#         model = AutoModelForCausalLM.from_pretrained(
#             self.model_path,
#             device_map="auto",
#             quantization_config=self.quant_config,
#             low_cpu_mem_usage=True,
#             torch_dtype=torch.float16
#         )
#         tokenizer = AutoTokenizer.from_pretrained(self.model_path)
#         self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
#         self.mongodb_uri = os.getenv("MONGO_URI")
#         self.max_chars = 270
#         self.banned_words = ["Deorkk, DestraBot,primus_sentient"]
#         self.llm_model = model
#         self.twitter_username = os.getenv("TWITTER_USERNAME")
#         self.db_name = os.getenv("DB_NAME")
#         self.COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")



class Config:
    def __init__(self):
        self.model_path = "./crypto_bot_model1"
        # self.quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self.llm_model_llama = "llama3.1"  # ✅ Keep as string for Ollama
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        self.quant_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            load_in_4bit=True,  # ✅ Use 4-bit quantization
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        self.mongodb_uri = os.getenv("MONGO_URI")
        self.max_chars = 270
        self.banned_words = ["Deorkk, DestraBot,primus_sentient"]
        self.llm_model = model
        self.twitter_username = os.getenv("TWITTER_USERNAME")
        self.db_name = os.getenv("DB_NAME")
        self.COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")





class Database:
    def __init__(self, config: Config):
        self.client = MongoClient(config.mongodb_uri)
        self.db = self.client.primus_twitter
        # self.db = self.client[config.db_name]
        self.replies = self.db.replies
        self.responses = self.db.responses
        self.MAX_REPLIES_PER_THREAD = 40
        self.reply_ids = self.db.reply_ids  # New collection for tracking reply IDs
        self.metadata = self.db.metadata  # New collection to store timestamps
        self.metadata.create_index('key', unique=True)

        self.replies.create_index('reply_id', unique=True)
        self.responses.create_index('reply_id', unique=True)
        self.reply_ids.create_index('reply_id', unique=True)



    def get_last_timestamp(self, key: str) -> str:
        record = self.metadata.find_one({'key': key})
        if record:
            try:
                # Validate timestamp format
                parse(record['timestamp'])
                return record['timestamp']
            except Exception as e:
                print(f"Invalid timestamp in database: {e}")
        return (datetime.utcnow() - timedelta(minutes=15)).isoformat("T") + "Z"
  

    def update_timestamp(self, key: str, timestamp: str):
        """Update the timestamp for the given key."""
        self.metadata.update_one(
            {'key': key},
            {'$set': {'key': key, 'timestamp': timestamp}},
            upsert=True
        )

    def store_reply_id(self, reply_id: int):
        """Store a reply ID in the tracking collection with an initial status."""
        try:
            result = self.reply_ids.update_one(
                {'reply_id': reply_id},
                {'$setOnInsert': {'reply_id': reply_id, 'status': 'pending', 'created_at': datetime.utcnow()}},
                upsert=True
            )
            print(f"✓ Stored reply ID: {reply_id} ({result.modified_count} modified, {result.upserted_id is not None})")
        except Exception as e:
            # print(f"Error storing reply ID {reply_id}: {e}")
            print(" ")

    def is_known_reply(self, tweet_id: int) -> bool:
        """Check if a tweet ID exists in our reply tracking collection."""
        response = self.reply_ids.find_one({'reply_id': tweet_id})
        if response:
            # Print status for debugging
            print(f"Reply ID {tweet_id} status: {response.get('status', 'unknown')}")
            return response.get('status') in ['processed', 'skipped']
        return False

    def get_user_replies_count(self, parent_tweet_id: str, username: str) -> int:
        """Get the number of times we've replied to this user in this thread."""
        count = self.responses.count_documents({
            'parent_tweet_id': parent_tweet_id,
            'username': username,
            'status': {'$in': ['pending', 'posted', 'completed']}
        })
        return count
    

    def store_reply(self, reply_data: Dict):
        """Store a reply in the database."""
        try:
            self.replies.insert_one(reply_data)
        except Exception as e:
            if "duplicate key error" not in str(e):
                print(f"Error storing reply: {e}")

    def store_response(self, response_data: Dict):
        """Store a generated response in the database."""
        try:
            # Check reply count for this user in this thread
            reply_count = self.get_user_replies_count(
                response_data.get('parent_tweet_id'),
                response_data.get('username')
            )

            if reply_count >= self.MAX_REPLIES_PER_THREAD:
                print(f"Reached maximum replies ({self.MAX_REPLIES_PER_THREAD}) for user {response_data.get('username')} in thread {response_data.get('parent_tweet_id')}")
                return

            # Store the response with the reply count
            response_data['reply_number'] = reply_count + 1
            # self.responses.update_one(
            #     {'reply_id': response_data['reply_id']},
            #     {'$set': response_data},
            #     upsert=True
            # )
            self.responses.update_one(
                                        {"reply_id": response_data["reply_id"]},  # Match existing entry
                                        {"$set": response_data},  # Update with new data
                                        upsert=True  # Insert if it doesn't exist
                                    )
            
            # Mark the reply as processed
            self.reply_ids.update_one(
                {'reply_id': response_data['reply_id']},
                {'$set': {'status': 'processed'}}
            )
            print(f"Marked reply ID {response_data['reply_id']} as processed")
            print("✅ Successfully inserted response into DB")
        except Exception as e:
            print(f"❌ Error inserting response into DB: {e}")


    def is_conversation_limit_reached(self, parent_tweet_id: str, username: str) -> bool:
        """Check if the conversation limit has been reached for this user in this thread."""
        count = self.get_user_replies_count(parent_tweet_id, username)
        return count >= self.MAX_REPLIES_PER_THREAD

class TwitterAPI:
    def __init__(self, config: Config, db: Database):
        self.client = tweepy.Client(
            bearer_token=config.twitter_bearer_token,
            wait_on_rate_limit=True
        )
        self.db = db

    def get_mentions(self, username: str, count: int = 10) -> List[Dict]:
        """Fetch recent mentions of the user in other people's tweets."""
        try:

            start_time = self.db.get_last_timestamp('curr_time_mentions')
            current_time = datetime.utcnow().isoformat("T") + "Z"
            self.db.update_timestamp('curr_time_mentions', current_time)
            
            user = self.client.get_user(username=username)
            user_id = user.data.id



            # Query to find all mentions
            # query = f"@{username} -from:{username} -is:retweet"
            query = f"@{username}"
            mentions = []

            for response in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'text', 'conversation_id', 
                            'referenced_tweets', 'in_reply_to_user_id'],
                user_fields=['username'],
                start_time=start_time,  # Add start_time filter
                max_results=100,
                limit=count
            ):
                if response.data:
                    for tweet in response.data:
                        try:

                            # Skip if tweet is by the account owner
                            if tweet.author_id == user_id:
                                continue

                            # Skip if it's a retweet
                            if hasattr(tweet, 'referenced_tweets') and any(
                                ref.type == 'retweeted' for ref in tweet.referenced_tweets
                            ):
                                print(f"Skipping retweet: {tweet.text}")
                                continue

                            # Skip if this tweet ID exists in our reply tracking
                            if self.db.is_known_reply(tweet.id):
                                # print(f"Skipping known reply: {tweet.text}")
                                print(f"Skipping known reply: {tweet.id} (already marked as skipped/processed)")
                                continue

                            # Get parent tweet info if it's a reply
                            parent_tweet_text = ""
                            parent_tweet_id = None
                                
                            if hasattr(tweet, 'referenced_tweets') and tweet.referenced_tweets:
                                for ref in tweet.referenced_tweets:
                                    if ref.type == 'replied_to':
                                        try:
                                            parent = self.client.get_tweet(ref.id)
                                            if parent and parent.data:
                                                parent_tweet_text = parent.data.text
                                                parent_tweet_id = ref.id
                                                print(f"Found parent tweet: {parent_tweet_text}")
                                        except Exception as e:
                                            # print(f"Error fetching parent tweet: {e}")
                                            print(" ")

                            # Process the mention
                            mentions.append({
                                'parent_tweet_id': parent_tweet_id,
                                'parent_tweet_text': parent_tweet_text,
                                'reply_id': tweet.id,
                                'reply_text': tweet.text,
                                'username': username,
                                'created_at': tweet.created_at,
                                'status': 'pending',
                                'is_mention': True
                            })
                            print(f"Found mention from {username}: {tweet.text}")

                        except Exception as e:
                            # print(f"Error processing tweet: {e}")
                            print(" ")
                            continue

            # print(f"Updated curr_time_mentions to {last_timestamp}")

            return mentions
        except Exception as e:
            # print(f"Error fetching mentions: {e}")
            print(" ")
            return []

    def get_full_mentions(self, username: str, count: int = 10) -> List[Dict]:
        """Fetch recent mentions of the user in other people's tweets."""
        try:

            # Get the last timestamp for full mentions
            start_time = self.db.get_last_timestamp('curr_time_full_mentions')
            print("full_mention started to fetch from :", start_time)
            current_time = datetime.utcnow().isoformat("T") + "Z"
            self.db.update_timestamp('curr_time_full_mentions', current_time)

            user = self.client.get_user(username=username)
            user_id = user.data.id

            # Get today's date (UTC) and set start_time to midnight


            # Broader query to catch all mentions and direct tweets
            query = f"@{username} -from:{username} -is:retweet"
            mentions = []

            for response in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'text', 'conversation_id', 
                            'referenced_tweets', 'in_reply_to_user_id'],
                user_fields=['username'],
                start_time=start_time,  # Add start_time filter
                max_results=100,
                limit=count
            ):
                if response.data:
                    for tweet in response.data:
                        try:
                            # Skip if tweet is by the account owner
                            if tweet.author_id == user_id:
                                continue

                            # Determine if this is a direct mention or a reply in a thread
                            is_direct_mention = (
                                not hasattr(tweet, 'referenced_tweets') or 
                                not tweet.referenced_tweets or
                                not any(ref.type == 'replied_to' for ref in tweet.referenced_tweets)
                            )

                            # Get author info
                            author = self.client.get_user(id=tweet.author_id, user_fields=['username'])
                            author_username = author.data.username if author.data else "UnknownUser"

                            # Get parent tweet info
                            parent_tweet_text = ""
                            parent_tweet_id = None
                                
                            if hasattr(tweet, 'referenced_tweets') and tweet.referenced_tweets:
                                for ref in tweet.referenced_tweets:
                                    if ref.type == 'replied_to':
                                        try:
                                            parent = self.client.get_tweet(ref.id)
                                            if parent and parent.data:
                                                parent_tweet_text = parent.data.text
                                                parent_tweet_id = ref.id
                                        except Exception as e:
                                            # print(f"Error fetching parent tweet: {e}")
                                            print(" ")

                            # Always process direct mentions
                            if is_direct_mention:
                                mentions.append({
                                    'parent_tweet_id': parent_tweet_id,
                                    'parent_tweet_text': parent_tweet_text,
                                    'reply_id': tweet.id,
                                    'reply_text': tweet.text,
                                    'username': author_username,
                                    'created_at': tweet.created_at,
                                    'status': 'pending',
                                    'is_mention': True
                                })
                                print(f"Found mention from {author_username}: {tweet.text}")

                        except Exception as e:
                            # print(f"Error processing tweet: {e}")
                            print(" ")
                            continue

                # print(f"Updated curr_time_full_mentions to {last_timestamp}")

            return mentions

        except Exception as e:
            # print(f"Error fetching mentions: {e}")
            print(" ")
            return []

    def get_recent_replies(self, username: str, count: int = 10) -> List[Dict]:
        """Fetch recent replies to the user's tweets, including nested conversations up to 4 levels deep."""
        try:
            # Get the last timestamp for replies
            start_time = self.db.get_last_timestamp('curr_time_replies')
            print("fetching replies from:", start_time)
            current_time = datetime.utcnow().isoformat("T") + "Z"
            self.db.update_timestamp('curr_time_replies', current_time)

            user = self.client.get_user(username=username)
            user_id = user.data.id

            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=count,
                tweet_fields=['created_at', 'conversation_id', 'in_reply_to_user_id', 'referenced_tweets'],
                exclude=['replies', 'retweets'] 
            )
            print(f"Fetched tweets: {tweets.data if tweets else 'No tweets found'}")
            if not tweets or not tweets.data:
                return []

            all_replies = []
            for tweet in tweets.data:
                # Track conversation depth for each user in this thread
                conversation_depth = {}  # {username: current_depth}
                processed_tweets = set()  # Track processed tweet IDs
                
                query = f"conversation_id:{tweet.id}"  # Remove is:reply to get all tweets in conversation
                print(f"Querying replies for tweet {tweet.id}: {query}")
                

                for response in tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=query,
                    tweet_fields=['created_at', 'conversation_id', 'author_id', 'text', 
                                'in_reply_to_user_id', 'referenced_tweets'],
                    user_fields=['username'],
                    start_time=start_time,
                    max_results=100,
                    limit=10
                ):
                    print(f"Replies fetched: {response.data if response.data else 'None'}")
                    if response.data:
                        for reply in response.data:
                            try:
                                # Skip if we've already processed this tweet
                                print(f"Reply text: {reply.text}, Reply author ID: {reply.author_id}, In reply to user ID: {reply.in_reply_to_user_id}")
                                if reply.id in processed_tweets:
                                    continue
                                
                                # Skip if reply is by the account owner
                                if reply.author_id == user_id:
                                    continue
                                
                                # Get the author of the reply
                                author = self.client.get_user(id=reply.author_id, user_fields=['username'])
                                username = author.data.username if author.data else "UnknownUser"
                                
                                # Get the parent tweet this reply is responding to
                                parent_id = None
                                if hasattr(reply, 'referenced_tweets'):
                                    for ref in reply.referenced_tweets:
                                        if ref.type == 'replied_to':
                                            parent_id = ref.id
                                            break
                                
                                # Calculate conversation depth
                                if parent_id == tweet.id:
                                    # Direct reply to original tweet
                                    current_depth = 1
                                elif parent_id:
                                    # Find the depth of the parent tweet and add 1
                                    parent_depth = next((r['depth'] for r in all_replies if r['reply_id'] == parent_id), 0)
                                    current_depth = parent_depth + 1
                                else:
                                    continue  # Skip if we can't determine the depth
                                
                                # Skip if beyond maximum depth of 4
                                if current_depth > 4:
                                    continue
                                
                                # Store the reply ID in tracking collection
                                self.db.store_reply_id(reply.id)
                                
                                # Add to processed tweets
                                processed_tweets.add(reply.id)
                                
                                # Add to replies list with depth information
                                all_replies.append({
                                    'parent_tweet_id': tweet.id,
                                    'parent_tweet_text': tweet.text,
                                    'immediate_parent_id': parent_id,  # Store immediate parent for threading
                                    'reply_id': reply.id,
                                    'reply_text': reply.text,
                                    'username': username,
                                    'created_at': reply.created_at,
                                    'status': 'pending',
                                    'depth': current_depth  # Store the depth level
                                })
                                
                                print(f"Processed reply at depth {current_depth} from {username}: {reply.text}")
                                
                            except Exception as e:
                                # print(f"Error processing individual reply: {e}")
                                print(" ")
                                continue
                # print(f"Updated curr_time_full_mentions to {last_timestamp}")

            return all_replies
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return []


    # def fetch_related_tweets(self, keyword: str) -> List[str]:
    #     """Fetch top 10 tweets related to the given keyword."""
    #     try:
    #         if not keyword:  # Avoid unnecessary calls if no keyword
    #             return []
    #         # query = f"{keyword} -is:retweet"
    #         query = f"{keyword.replace('$', '')} -is:retweet"
    #         tweets = self.client.search_recent_tweets(
    #             query=query,


    #             max_results=10,
    #             tweet_fields=['text']
    #         )
    #         if tweets and tweets.data:
    #             return [tweet.text for tweet in tweets.data]
    #         return []
    #     except Exception as e:
    #         print(f"Error fetching related tweets: {e}")
    #         return []


    
    def fetch_related_tweets(self, keyword: str) -> List[str]:
        """Fetch top 10 tweets related to the given keyword."""
        try:
            if not keyword:  # Avoid unnecessary calls if no keyword
                return []
            
            # ✅ Ensure keyword is a string and sanitize it
            keyword = str(keyword).replace("$", "").strip()

            # ✅ Remove invalid characters (curly braces, excessive quotes, etc.)
            keyword = re.sub(r"[{}']", "", keyword)

            # ✅ Convert comma-separated keywords into Twitter API-friendly format
            if "," in keyword:
                keywords_list = [word.strip() for word in keyword.split(",")]
                query = " OR ".join(keywords_list) + " -is:retweet"
            else:
                query = f"{keyword} -is:retweet"


            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=10,
                tweet_fields=['text']
            )

            if tweets and tweets.data:
                return [tweet.text for tweet in tweets.data]
            return []
        
        except Exception as e:
            print(f"Error fetching related tweets: {e}")
            return []



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class CryptoDataFetcher:
    def __init__(self):
        self.crypto_dict = self.get_supported_cryptos()  # ✅ Load crypto dictionary
        self.COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")

    def get_supported_cryptos(self):
        """Fetch supported cryptocurrencies dynamically from CoinGecko."""
        url = "https://api.coingecko.com/api/v3/coins/list"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # ✅ Create a dictionary with both full names and symbols
            crypto_dict = {coin["id"]: coin["id"] for coin in data}  # Full names
            crypto_dict.update({coin["symbol"].lower(): coin["id"] for coin in data})  # Symbols

            # ✅ Save to a file for caching
            with open("crypto_dict.txt", "w", encoding="utf-8") as f:
                f.write(str(crypto_dict))

            return crypto_dict
        except Exception as e:
            print(f"Error fetching crypto list: {e}")
            return {}

    def fetch_coin_gecko_data(self, key_word):
        """Fetch data from CoinGecko."""
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": key_word, "vs_currencies": "usd", "include_24hr_vol": "true", "include_market_cap": "true"}
        try:
            response = requests.get(url, params=params)
            # print("fetch_coin_gecko_data -> ",response.json())
            response.raise_for_status()
            COINGECKO_RATE_LIMIT_DELAY = 1.5  # seconds between API calls

            time.sleep(COINGECKO_RATE_LIMIT_DELAY)  # Rate limit delay
            return response.json().get(key_word, {})
        except requests.exceptions.RequestException:
            return {}

    def fetch_coin_market_cap_data(self, key_word):
        """Fetch data from CoinMarketCap."""

        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        headers = {"X-CMC_PRO_API_KEY": self.COINMARKETCAP_API_KEY}
        params = {"symbol": key_word.upper(), "convert": "USD"}

        try:
            response = requests.get(url, headers=headers, params=params)
            response_data = response.json()
            # print("fetch_coin_market_cap_data -> ", response_data)  # Debugging output
            
            response.raise_for_status()  # Ensure no HTTP errors
            
            if "data" in response_data and key_word.upper() in response_data["data"]:
                return response_data["data"][key_word.upper()]
            else:
                # print(f"⚠️ No data found for {key_word.upper()}")
                return {}

        except requests.exceptions.RequestException as e:
            # print(f"❌ API Request Error: {e}")
            return {}


    def fetch_defi_llama_data(self, key_word):
        """Fetch DeFi data from DefiLlama."""
        url = f"https://api.llama.fi/protocol/{key_word}"
        try:
            response = requests.get(url)
            # print("fetch_defi_llama_data -> ",response.json())

            response.raise_for_status()
            return response.json()  # Returns TVL and liquidity data
        except requests.exceptions.RequestException:
            return {}

    def fetch_dune_analytics_data(self, query_id):
        """Fetch on-chain analytics from Dune Analytics (Public API, no API key required)."""
        url = f"https://api.dune.com/api/v1/query/{query_id}/results"
        try:
            response = requests.get(url)
            # print("fetch_dune_analytics_data -> ",response.json())

            response.raise_for_status()
            return response.json()  # Returns structured query data
        except requests.exceptions.RequestException:
            return {}



    def fetch_defi_llama_data(self, key_word):
        """Fetch DeFi data from DefiLlama."""
        url = f"https://api.llama.fi/protocol/{key_word}"
        try:
            response = requests.get(url)
            # print("fetch_defi_llama_data -> ",response.json())

            response.raise_for_status()
            return response.json()  # Returns TVL and liquidity data
        except requests.exceptions.RequestException:
            return {}

    def fetch_dune_analytics_data(self, query_id):
        """Fetch on-chain analytics from Dune Analytics (Public API, no API key required)."""
        url = f"https://api.dune.com/api/v1/query/{query_id}/results"
        try:
            response = requests.get(url)
            # print("fetch_dune_analytics_data -> ",response.json())

            response.raise_for_status()
            return response.json()  # Returns structured query data
        except requests.exceptions.RequestException:
            return {}


    def fetch_crypto_data(self, key_word, dune_query_id=None):
        """Fetch data from all sources and return aggregated results."""
        data = {
            "coingecko": self.fetch_coin_gecko_data(key_word),
            "coinmarketcap": self.fetch_coin_market_cap_data(key_word),
            "defillama": self.fetch_defi_llama_data(key_word),
        }

        
        if dune_query_id:  # If a Dune Analytics query ID is provided
            data["dune"] = self.fetch_dune_analytics_data(dune_query_id)
        
        # print("org data ",data)
        cleaned_data = {}
        for k, v in data.items():
            if v:  # Ensure v is not empty
                if isinstance(v, str):
                    if "error" not in v.lower():  
                        cleaned_data[k] = v
                else:
                    cleaned_data[k] = v  # If v is a dictionary, keep it

        data = cleaned_data
        # print("data -----------------",data)
        return data

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# class LLMHandler:
#     def __init__(self, config: Config, data_fetcher: CryptoDataFetcher):
#         self.config = config
#         self.model = config.llm_model
#         self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
#         self.emotion_value = os.getenv("HUMOR_LEVEL", "medium")
#         self.nsfw_value = os.getenv("NSFW_LEVEL", "low")
        
#         # Load the combined prompt settings from a JSON file
#         self.prompt_settings = self._load_prompt_file("prompt_settings.json")
#         self.data_fetcher = data_fetcher
#         self.crypto_dict = data_fetcher.crypto_dict 
#         self.ollama_model = ChatOllama(
#             model=config.llm_model,  # Ensure this is a string ("llama3.3")
#             temperature=0.7
#         )

#         self.tools = [self.search_tool]
#         self.memory = MemorySaver()
#         self.agent_executor = create_react_agent(
#             self.ollama_model,
#             self.tools,
#             checkpointer=self.memory
#         )


class LLMHandler:
    def __init__(self, config: Config, data_fetcher: CryptoDataFetcher):
        self.config = config
        self.model = config.llm_model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
          # ✅ Use the tokenizer from config
        self.emotion_value = os.getenv("HUMOR_LEVEL", "medium")
        self.nsfw_value = os.getenv("NSFW_LEVEL", "low")

        # Load the combined prompt settings from a JSON file
        self.prompt_settings = self._load_prompt_file("prompt_settings.json")
        self.data_fetcher = data_fetcher
        self.crypto_dict = data_fetcher.crypto_dict 
        
        # ✅ Pass the correct string model to ChatOllama
        self.ollama_model = ChatOllama(
            model=config.llm_model_llama,  # "llama3.3" (String)
            temperature=0.7
        )
        self.search_tool = TavilySearchResults(max_results=10, tavily_api_key=os.getenv("TAVILY_API_KEY"))
        self.tools = [self.search_tool]

        self.memory = MemorySaver()
        self.agent_executor = create_react_agent(
            self.ollama_model,
            self.tools,
            checkpointer=self.memory
        )

        
    def _load_prompt_file(self, filename: str):
        """Load the prompt settings from a JSON file."""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Prompt file {filename} not found. Using default values.")
            return {}

    def _validate_response_length(self, response: str) -> str:
        """Ensure the response is within the character limit."""
        if len(response) > self.config.max_chars:
            return response[:self.config.max_chars].strip()
        return response



    async def _generate_openai_text(self, prompt: str):
        try:
            client = OpenAI(api_key=self.config.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            print("response from open ai in function is :----------------------------------- ",response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()
        except Exception as e:
            # print(f"Error optimizing response with ChatGPT: {e}")
            return response
        # """Generate text using OpenAI GPT-4 while handling rate limits."""
        # try:
        #     client = openai.OpenAI(api_key=self.config.openai_api_key)
        #     max_input_tokens = 4000  # Ensure input prompt is within limits
        #     truncated_prompt = prompt[:max_input_tokens]

        #     response = client.chat.completions.create(
        #         model="gpt-4",
        #         messages=[{"role": "user", "content": truncated_prompt}],
        #         temperature=0.7,
        #         max_tokens=1024  # ✅ Limit output length
        #     )
        #     return response.choices[0].message.content.strip()

        # except openai.RateLimitError:  # ✅ Handle rate limit errors properly
        #     print("Rate limit hit. Retrying after 60 seconds...")
        #     await asyncio.sleep(60)  # ⏳ Wait before retrying
        #     return await self._generate_openai_text(prompt)  # Retry request

        # except Exception as e:  # Generic error handling
        #     print(f"Error in OpenAI generation: {e}")
        #     return "Error generating response."

    async def _generate_ollama_text(self, prompt: str):
        # """Generate text using Llama 3.3 via Ollama."""
        # try:
            # response = await asyncio.to_thread(ollama.chat, model="llama3.3", messages=[{"role": "user", "content": prompt}])
            # return response['message']['content']

        config = {"configurable": {"thread_id": "default"}}
        human_message = HumanMessage(content=prompt)

        response = []
        for chunk in self.agent_executor.stream({"messages": [human_message]}, config):
            if "agent" in chunk:
                for msg in chunk["agent"].get("messages", []):
                    if hasattr(msg, "content"):
                        response.append(msg.content)
            elif "tools" in chunk:
                response.append(" ")

        print(" response from llama in function is :-----------------------------------  ",response)
        return " ".join(response)
        
        # except Exception as e:
        #     print(f"Error in Ollama generation: {e}")
        #     return "Error generating response."

    # async def _generate_text(self, prompt: str) -> str:
    #     """Generate text using the model."""
    #     # try:
    #     inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    #     inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
    #     outputs = self.model.generate(
    #         **inputs,
    #         max_new_tokens=500,  # Controls the length of generated tokens separately
    #         num_return_sequences=1,
    #         temperature=0.7,
    #         do_sample=True,
    #         pad_token_id=self.tokenizer.eos_token_id
    #     )

        
    #     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     # Remove the original prompt from the response
    #     print(" ✅ response from local model :- ",response)
    #     response = response[len(prompt):].strip()
    #     print("✅ response from local model :- ",response)

    #     print("local model response ------------------- ",response)
    #     return response
    #     # except Exception as e:
    #     #     print(f"Error generating text: {e}")
    #     #     return ""

    
    async def _generate_text(self, prompt: str) -> str:
        """Generate text using the model with improved error handling."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,  # Limit generation length
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id
            )

            # Check if model generated output
            if outputs is None or len(outputs) == 0:
                print("🚨 Model did not generate any output.1")
                return "Sorry, I couldn't generate a response."

            # Decode the output
            response1 = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Ensure response is meaningful
            if not response1:
                print("🚨 Model generated an empty response.2")
                return "Sorry, I couldn't generate a response."

            # Extract meaningful response, if format includes "Response:"
            response_start = response1.lower().find("response:")
            if response_start != -1:
                response2 = response1[response_start + len("response:"):].strip()
            else:
                response2 = response1

            # Final check for an empty response
            if not response2 or response2.lower() in ["", "response:"]:
                print("🚨 Model generated an empty response after processing.3")
                return "Sorry, I couldn't generate a response."

            # Log and return final response
            print("📝 Response from local model ----------------:", response2)
            return response2

        except Exception as e:
            print(f"🚨 Error in _generate_text: {e}")
            return "Sorry, an error occurred while generating the response."





    # async def _generate_text(self, prompt: str) -> str:
    #     """Generate text using the model."""
    #     try:
    #         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    #         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
    #         outputs = self.model.generate(
    #             **inputs,
    #             max_length=512,
    #             num_return_sequences=1,
    #             temperature=0.7,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.eos_token_id
    #         )
            
    #         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         # Remove the original prompt from the response
    #         response = response[len(prompt):].strip()
    #         print("response ------------------- ",response)
    #         return response
    #     except Exception as e:
    #         print(f"Error generating text: {e}")
    #         return ""

    async def classify_content(self, parent_tweet: str, reply: str, conversation_context: List[str] = None) -> Tuple[bool, str]:
       
        context_str="\n".join(conversation_context) if conversation_context else ""
        Previous_context=context_str
        Parent_tweet=parent_tweet
        Reply=reply
        user_input=Previous_context+" "+Parent_tweet +" "+Reply

        """Cleans user input by tokenizing and removing stopwords."""
        words = self.tokenizer.tokenize(user_input.lower())
        filtered_words = [word for word in words if word not in STOPWORDS]
      
        # FIX: Convert list to string before tokenizing again
        words = self.tokenizer.tokenize(" ".join(filtered_words).lower())
       
        matched_cryptos = set()
        
        # Normalize dictionary keys to lowercase
        normalized_crypto_dict = {k.lower(): v for k, v in self.crypto_dict.items()}
      
        # ✅ Exact Match Check
        for word in words:
            if word in normalized_crypto_dict:
                matched_cryptos.add(normalized_crypto_dict[word])

        data={}
        

        if not matched_cryptos:
            return False,matched_cryptos,data
        else:
            data = {}  # ✅ Store data for multiple cryptos
            for key_word in matched_cryptos:
                crypto_data = self.data_fetcher.fetch_crypto_data(key_word)
                if crypto_data:
                    data[key_word] = crypto_data
            return True,matched_cryptos,data
            

    



    async def generate_crypto_response(self, parent_tweet: str, reply: str, related_tweets: List[str], conversation_context: List[str] = None,data: dict = {}) -> str:
        """Generate a response for crypto-related content."""
        print("in generate_crypto_response")
        try:
            context_str = "\n".join(conversation_context) if conversation_context else ""
            emotion_prompt = self.prompt_settings["humor"].get(self.emotion_value, "Include some humor.")
            nsfw_prompt = self.prompt_settings["nsfw"].get(self.nsfw_value, "Do not include any NSFW content.")
            current_date = datetime.today().strftime("%Y-%m-%d")

            prompt = f"""
            Generate a crypto-focused response with these requirements:
            - Maximum 270 characters
            - No price predictions
            - No 'based on' or 'according to' phrases
            - No quotation marks or hashtags
            - Complete sentences only
            - Strictly do not repeat or rephrase the user’s question
            - Ensure the response adds value and remains relevant
            - Make sure to provide correct response using 'Data' provided without null and do not repeat any kind of Parent tweet or Previous context or User reply or any kind or  Related tweets. 
            Context:
            Parent tweet: {parent_tweet}
            Previous context: {context_str}
            User reply: {reply}
            Data :{data}
            Date: {current_date}
            Related tweets: {', '.join(related_tweets)}

            Response:
            """
            with open("generate_crypto_response.txt", "w", encoding="utf-8") as file:
                file.write(prompt)
            gpt_response, llama_response, fine_tuned_response = await asyncio.gather(
                self._generate_openai_text(prompt),
                self._generate_ollama_text(prompt),
                self._generate_text(prompt)
            )

          


            return (
                    self._validate_response_length(gpt_response),
                    self._validate_response_length(llama_response),
                    self._validate_response_length(fine_tuned_response)
                )

            
        
        except Exception as e:
            print(f"Error in generate_crypto_response: {e}")
            return "Sorry, I couldn't generate a response at this time."


    
    async def generate_non_crypto_response(self, parent_tweet: str, reply: str, conversation_context: List[str] = None) -> str:
        """Generate a response for non-crypto content."""
        print("in generate_non_crypto_response")

        try:
            context_str = "\n".join(conversation_context) if conversation_context else ""
            current_date = datetime.today().strftime("%Y-%m-%d")

            prompt = f"""
            Generate a witty response with these requirements:
            - Maximum 270 characters
            - Include humor or playful sarcasm
            - Can use internet slang and emojis sparingly
            - No 'based on' or 'according to' phrases
            - No quotation marks or hashtags
            - Make sure to respond accurately.
            - Complete sentences only
            - Strictly do not repeat or rephrase the user’s question
            - Ensure the response adds value and remains relevant
            - Make sure to provide correct response using 'Data' provided without null and do not repeat any kind of Parent tweet or Previous context or User reply or any kind or  Related tweets. 
            

            Context:
            Parent tweet: {parent_tweet}
            Previous context: {context_str}
            User reply: {reply}
            Date: {current_date}
            Response:
            """

            with open("prompt.txt", "w", encoding="utf-8") as file:
                file.write(prompt)
            gpt_response, llama_response, fine_tuned_response = await asyncio.gather(
             self._generate_openai_text(prompt),
             self._generate_ollama_text(prompt),
             self._generate_text(prompt)
            )
         

            return (
    self._validate_response_length(gpt_response),
    self._validate_response_length(llama_response),
    self._validate_response_length(fine_tuned_response)
)

            
        except Exception as e:
            print(f"Error in generate_non_crypto_response: {e}")
            return "Sorry, I couldn't generate a response at this time."



async def main():
    config = Config()
    db = Database(config)
    twitter_api = TwitterAPI(config,db)
    data_fetcher = CryptoDataFetcher()

    # ✅ Pass data_fetcher to LLMHandler
    llm_handler = LLMHandler(config, data_fetcher)

    # username = "primus_sentient"
    # username = config.twitter_username
    username = "deorkk"

    while True:  # Infinite loop to fetch replies every 30 minutes
        try:
            print("Fetching replies...")
            replies = twitter_api.get_recent_replies(username)

            if not replies:
                print("No replies found to process.")
            else:
                print(f"Found {len(replies)} replies to process")

                processed_count = 0
                skipped_count = 0

                replies.sort(key=lambda x: (x['depth'], x['created_at']))
                for reply in replies:
                    try:
                        #adding this to skip already processed ones from here 
                                # Check if reply ID is already stored
                        if db.is_known_reply(reply['reply_id']):
                            print(f"Skipping known reply: {reply['reply_text']}")
                            continue  # Skip processing this reply if already stored
                        
                        # Proceed with processing if not already stored
                        db.store_reply_id(reply['reply_id'])  # Store reply ID to track it
                        print(f"Processing reply: {reply['reply_text']}")
                        #adding this to skip already processed ones to here


                                # Get the context of the conversation up to this point
                        conversation_context = []
                        if reply['immediate_parent_id']:
                            # Find previous replies in this chain
                            current_parent_id = reply['immediate_parent_id']
                            while current_parent_id:
                                parent_reply = next(
                                    (r for r in replies if r['reply_id'] == current_parent_id), 
                                    None
                                )
                                if parent_reply:
                                    conversation_context.insert(0, parent_reply['reply_text'])
                                    current_parent_id = parent_reply['immediate_parent_id']
                                else:
                                    break
                        
                        if db.is_conversation_limit_reached(reply['parent_tweet_id'], reply['username']):
                            print(f"Conversation limit reached for user {reply['username']} in thread {reply['parent_tweet_id']}. Skipping...")
                            skipped_count += 1
                            continue

                        # Classify the reply and determine if it's crypto-related
                        is_crypto, keyword,data = await llm_handler.classify_content(
                            reply['parent_tweet_text'], reply['reply_text'], conversation_context
                        )

                        # Generate an appropriate response
                        if is_crypto:
                            related_tweets = twitter_api.fetch_related_tweets(keyword)
                            gpt_response,llama_response,finetune_response = await llm_handler.generate_crypto_response(
                                reply['parent_tweet_text'], reply['reply_text'], related_tweets, conversation_context,data
                            )
                        else:
                            gpt_response,llama_response,finetune_response = await llm_handler.generate_non_crypto_response(
                                reply['parent_tweet_text'], reply['reply_text'], conversation_context
                            )


                        
                        response_data = {
                            'reply_id': reply['reply_id'],
                            'parent_tweet_id': reply['parent_tweet_id'],
                            'immediate_parent_id': reply['immediate_parent_id'],
                            'parent_tweet': reply['parent_tweet_text'],
                            'reply': reply['reply_text'],
                            'username': reply['username'],
                            'gpt_response': gpt_response,
                            'llama_response': llama_response,
                            'finetune_response': finetune_response,
                            'is_crypto': is_crypto,
                            'keywords': list(reply.get('keywords', set())) if is_crypto else [],
                            'status': 'pending',
                            'created_at': datetime.utcnow(),
                            'depth': reply['depth']
                        }

                        print(f"\nSaving response to DB: {response_data}")  # Print before storing
                        db.store_response(response_data)
                        print("✅ Response successfully saved in DB!") 
                        latest_entry = db.responses.find_one(sort=[("_id", -1)])  
                        print("\nLatest Entry in DB:", latest_entry)
                        print("\n===== Processing Response: =====\n")
                        print(f"Parent Tweet: {reply['parent_tweet_text']}")
                        print(f"\nUser Reply: {reply['reply_text']}")
                        print(f"\nGPT Response: {gpt_response}")
                        print(f"\nLLaMA Response: {llama_response}")
                        print(f"\nFine-Tuned Response: {finetune_response}")
                        processed_count += 1
                    except Exception as e:
                        print(f"Error processing reply: {e}")
                        skipped_count += 1

                print(f"\nProcessing Summary: {processed_count} processed, {skipped_count} skipped.")

           # Process mentions
            mentions = twitter_api.get_mentions(username)
            print(f"\n\n*******************************\n\n")
            print(f"MENTIONS FROM BELOW") 
            # if mentions:
            #     print(f"\nProcessing {len(mentions)} mentions...")
            if not mentions:
                
                print("No mentions found to process.")
            else:
                
                print(f"\nProcessing {len(mentions)} mentions...")
                processed_count = 0
                skipped_count = 0
                for mention in mentions:
                    try:
                        if mention.get('parent_tweet_id') and db.is_conversation_limit_reached(mention['parent_tweet_id'], mention['username']):
                            print(f"Conversation limit reached for user {mention['username']} in thread {mention['parent_tweet_id']}. Skipping...")
                            skipped_count += 1
                            continue
                        existing_response = db.responses.find_one({
                            'reply_id': mention['reply_id']
                        })
                        if existing_response:
                            print(f"Already processed mention from {mention['username']}. Skipping...")
                            skipped_count += 1
                            continue

                        # Store the mention
                        db.store_reply(mention)

                        # Classify and generate response
                        is_crypto, keyword,data = await llm_handler.classify_content(
                            "", mention['reply_text']
                        )

                        if is_crypto:
                            related_tweets = twitter_api.fetch_related_tweets(keyword)
                            gpt_response,llama_response,finetune_response = await llm_handler.generate_crypto_response(
                                "", mention['reply_text'], related_tweets,data
                            )
                        else:
                            gpt_response,llama_response,finetune_response = await llm_handler.generate_non_crypto_response(
                                "", mention['reply_text']
                            )

                       

                        # Store response
                        # response_data = {
                        #     'reply_id': mention['reply_id'],
                        #     'parent_tweet_id': mention['parent_tweet_id'],
                        #     'parent_tweet': mention['parent_tweet_text'],
                        #     'reply': mention['reply_text'],
                        #     'username': mention['username'],
                        #     'gpt_response': gpt_response,
                        #     'llama_response': llama_response,
                        #     'finetune_response': finetune_response,
                            
                        #     'is_crypto': is_crypto,
                        #     'keywords': list(data.get('keywords', set())) if is_crypto else [],

                        #     'status': 'pending',
                        #     'created_at': datetime.utcnow(),
                        #     'is_mention': True
                        # }
                        response_data = {
                                'reply_id': mention['reply_id'],
                                'parent_tweet_id': mention['parent_tweet_id'],
                                'parent_tweet': mention['parent_tweet_text'],
                                'reply': mention['reply_text'],
                                'username': mention['username'],
                                'gpt_response': gpt_response,
                                'llama_response': llama_response,
                                'finetune_response': finetune_response,
                                'is_crypto': is_crypto,
                                'keywords': list(data.get('keywords', set())) if is_crypto else [],  # Changed from 'reply' to 'data'
                                'status': 'pending',
                                'created_at': datetime.utcnow(),
                                'is_mention': True
                            }


                        print(f"\nSaving response to DB: {response_data}")  # Print before storing
                        db.store_response(response_data)
                        latest_entry = db.responses.find_one(sort=[("_id", -1)])  
                        print("\nLatest Entry in DB:", latest_entry)
                        print("✅ Response successfully saved in DB!") 
                        
                        print("\n===== Processing Mention: =====")
                        print(f"User Mention: {mention['reply_text']}")
                        print(f"\nGPT Response: {gpt_response}")
                        print(f"\nLLaMA Response: {llama_response}")
                        print(f"\nFine-Tuned Response: {finetune_response}")
                        
                        processed_count += 1
                    except Exception as e:
                        print(f"Error processing mention: {e}")
                        skipped_count += 1

            if not replies and not mentions:
                print("No new interactions to process.")    

            print(f"\n\n*****************************************\n\n")

            ind_mentions = twitter_api.get_full_mentions(username)
            # if mentions:
            #     print(f"\nProcessing {len(mentions)} mentions...")
            if not ind_mentions:
                
                print("No mentions found to process.")
            else:
                
                print(f"\nProcessing {len(ind_mentions)} mentions...")
                processed_count = 0
                skipped_count = 0
                for mention in ind_mentions:
                    try:
                        if mention.get('parent_tweet_id') and db.is_conversation_limit_reached(mention['parent_tweet_id'], mention['username']):
                            print(f"Conversation limit reached for user {mention['username']} in thread {mention['parent_tweet_id']}. Skipping...")
                            skipped_count += 1
                            continue
                        existing_response = db.responses.find_one({
                            'reply_id': mention['reply_id']
                        })
                        if existing_response:
                            print(f"Already processed mention from {mention['username']}. Skipping...")
                            skipped_count += 1
                            continue

                        # Store the mention
                        db.store_reply(mention)

                        # Classify and generate response
                        is_crypto, keyword,data = await llm_handler.classify_content(
                            "", mention['reply_text']
                        )

                        if is_crypto:
                            related_tweets = twitter_api.fetch_related_tweets(keyword)
                            gpt_response,llama_response,finetune_response = await llm_handler.generate_crypto_response(
                                "", mention['reply_text'], related_tweets,data
                            )
                        else:
                            gpt_response,llama_response,finetune_response = await llm_handler.generate_non_crypto_response(
                                "", mention['reply_text']
                            )
                   
                        response_data = {
                            'reply_id': mention['reply_id'],
                            'parent_tweet_id': mention['parent_tweet_id'],
                            'parent_tweet': mention['parent_tweet_text'],
                            'reply': mention['reply_text'],
                            'username': mention['username'],
                            'gpt_response': gpt_response,
                            'llama_response': llama_response,
                            'finetune_response': finetune_response,
                            'is_crypto': is_crypto,
                            'keywords': list(data.get('keywords', set())) if is_crypto else [],  # Changed from 'reply' to 'data'
                            'status': 'pending',
                            'created_at': datetime.utcnow(),
                            'is_mention': True
                        }

                        print(f"\nSaving response to DB: {response_data}")  # Print before storing
                        db.store_response(response_data)
                        latest_entry = db.responses.find_one(sort=[("_id", -1)])  
                        print("\nLatest Entry in DB:", latest_entry)
                        print("✅ Response successfully saved in DB!") 
                        print("\n===== Processing Mention: =====")
                        print(f"User Mention: {mention['reply_text']}")
                        print(f"\nGPT Response: {gpt_response}")
                        print(f"\nLLaMA Response: {llama_response}")
                        print(f"\nFine-Tuned Response: {finetune_response}")
                        
                        processed_count += 1
                    except Exception as e:
                        print(f"Error processing mention: {e}")
                        skipped_count += 1

            if not replies and not mentions:
                print("No new interactions to process.")  

            print("\nWaiting 10 minutes before the next fetch...")
            print(f"Current Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            await asyncio.sleep(600)  # Wait for 30 minutes
        except Exception as e:
            print(f"Unhandled exception occurred: {e}")




if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Unhandled exception occurred in the main loop: {e}")
