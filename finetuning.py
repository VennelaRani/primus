# # # # import torch
# # # # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# # # # from datasets import load_dataset
# # # # from torch.utils.data import DataLoader
# # # # from transformers import Trainer, TrainingArguments
# # # # import os

# # # # # Set the environment variable to enable FP32 offloading
# # # # os.environ["llm_int8_enable_fp32_cpu_offload"] = "true"

# # # # # Load the pre-trained model and tokenizer
# # # # model_path = "/root/twitter_codes_vennela/new"

# # # # # Define the quantization configuration (8-bit)
# # # # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# # # # # Create a custom device map to offload parts of the model to CPU if needed
# # # # device_map = "auto"  # Use 'auto' to let Hugging Face decide, or define specific mappings

# # # # # Load the model with the custom device map and quantization configuration
# # # # model = AutoModelForCausalLM.from_pretrained(
# # # #     model_path,
# # # #     device_map=device_map,
# # # #     quantization_config=quantization_config,
# # # #     trust_remote_code=True  # Allow custom remote code to run
# # # # )

# # # # # Load the tokenizer
# # # # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)



# # # from transformers import AutoModelForCausalLM, AutoTokenizer

# # # # Model path and tokenizer
# # # model_path = "/root/twitter_codes_vennela/new"

# # # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
# # # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # # print("model ---------------------",model)
# # # print("tokenizer -----------------------",tokenizer)

# # # You can continue your training setup, trainer, etc., as needed

# # # # If you want to use a custom dataset, you can load it
# # # # For this example, we'll use the 'wikitext' dataset from Hugging Face Datasets
# # # # You can replace this with your own dataset loading logic
# # # dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# # # # Tokenize the data
# # # def tokenize_function(examples):
# # #     return tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True)

# # # tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # # # Prepare DataLoader
# # # train_dataset = tokenized_datasets["train"]
# # # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# # # # Training arguments
# # # training_args = TrainingArguments(
# # #     output_dir="./finetuned_model",
# # #     num_train_epochs=3,
# # #     per_device_train_batch_size=4,
# # #     per_device_eval_batch_size=4,
# # #     logging_dir="./logs",
# # #     save_steps=10_000,
# # #     save_total_limit=3,
# # #     prediction_loss_only=True,
# # # )

# # # # Trainer for the model
# # # trainer = Trainer(
# # #     model=model,
# # #     args=training_args,
# # #     train_dataset=train_dataset,
# # # )

# # # # Start training
# # # trainer.train()

# # # # Save the fine-tuned model
# # # model.save_pretrained("./finetuned_model")
# # # tokenizer.save_pretrained("./finetuned_model")

# # # # You can now load this model for inference or further fine-tuning later
# # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# # quantization_config = BitsAndBytesConfig(
# #     load_in_4bit=True,  # Matches "bitsandbytes_4bit"
# #     bnb_4bit_compute_dtype="bfloat16",  # or "float16"
# #     bnb_4bit_quant_type="nf4",  # or "fp4", check which is supported
# #     llm_int8_threshold=6.0,
# # )

# # model = AutoModelForCausalLM.from_pretrained(
# #     "/root/twitter_codes_vennela/new",
# #     trust_remote_code=True,
# #     quantization_config=quantization_config,
# # )

# # tokenizer = AutoTokenizer.from_pretrained(
# #     "/root/twitter_codes_vennela/new",
# #     trust_remote_code=True,
# # )


# # # import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load the model and tokenizer
# model_path = "/root/twitter_codes_vennela/4bit_model"  # Path to your quantized model
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


# # Function to generate responses with a guiding prompt
# def generate_response(user_input):
#     # Define the prompt to guide the bot
#     prompt = f"Generate a response based on the following user query: '{user_input}'"
    
#     # Tokenize the user input with the guiding prompt
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs['input_ids'].to(model.device)
    
#     # Generate the response using the model
#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return response


# user_input = input("You: ")


# # if user_input:
# #     # Add user input to chat history
# #     st.session_state.chat_history.append(f"You: {user_input}")
    
#     # Generate the model's response
# bot_response = generate_response(user_input)
# print("bot_response ",bot_response)
#     # # Add bot response to chat history
#     # st.session_state.chat_history.append(f"Bot: {bot_response}")
    
#     # # Display chat history
#     # for message in st.session_state.chat_history:
#     #     print(message)



# ****************_________________________________________
# import json
# import os
# import re
# import logging
# from pathlib import Path
# from typing import List, Dict, Any
# import torch
# import pandas as pd
# from datasets import Dataset
# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template
# from trl import SFTTrainer
# from transformers import TrainingArguments
# from unsloth import is_bfloat16_supported

# # 🔹 Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
# )

# # 🔹 Load dataset (Top 10 rows)
# DATA_PATH = "/root/twitter_codes_vennela/cleaned_crypto_questions.csv"
# logging.info("📂 Loading dataset...")

# df = pd.read_csv(DATA_PATH)  # Take the first 10 rows
# df = df.iloc[:, :2]  # Keep only 'Crypto Query' and 'Answer' columns
# df.columns = ["Crypto Query", "Answer"]

# # 🔹 Convert dataset into conversation format
# formatted_data = []
# for _, row in df.iterrows():
#     formatted_data.append({
#         "conversations": [
#             {"role": "user", "content": row["Crypto Query"]},
#             {"role": "assistant", "content": row["Answer"]}
#         ]
#     })

# # 🔹 Convert to Hugging Face dataset
# dataset = Dataset.from_list(formatted_data)

# # 🔹 Load model & tokenizer from quantized directory
# MODEL_PATH = "/root/twitter_codes_vennela/new1"
# logging.info("🚀 Loading model from Unsloth...")

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=MODEL_PATH,
#     max_seq_length=2048,
#     dtype=None,  # Auto-select precision
#     load_in_4bit=True,  # 4-bit precision
# )

# # 🔹 Apply LoRA fine-tuning
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     lora_alpha=16,
#     lora_dropout=0,
#     bias="none",
#     use_gradient_checkpointing="unsloth",
#     random_state=42,
# )

# # 🔹 Set chat template for tokenizer
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="chatml",
#     mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
#     map_eos_token=True,
# )

# # 🔹 Dataset Formatting
# def format_data(examples):
#     texts = [
#         tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
#         for conv in examples["conversations"]
#     ]
#     return {"text": texts}

# logging.info("✍️ Formatting dataset...")
# dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)

# # 🔹 Training Configuration
# logging.info("⚙️ Setting up training arguments...")
# training_args = TrainingArguments(
#     output_dir="crypto_bot_model",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     warmup_steps=5,
#     learning_rate=2e-4,
#     fp16=not is_bfloat16_supported(),
#     bf16=is_bfloat16_supported(),
#     logging_steps=1,
#     num_train_epochs=3,
#     optim="adamw_8bit",
#     weight_decay=0.01,
#     lr_scheduler_type="linear",
#     seed=42,
#     evaluation_strategy="no",
#     save_strategy="steps",
#     save_steps=50,
# )

# # 🔹 Train/Validation Split
# train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
# train_dataset, eval_dataset = train_val_split["train"], train_val_split["test"]

# # 🔹 Trainer Setup
# logging.info("🔥 Initializing Trainer...")
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     dataset_text_field="text",
#     max_seq_length=2048,
#     dataset_num_proc=2,
#     packing=False,
#     args=training_args,
# )

# # 🔹 Train the Model
# logging.info("🚀 Starting training...")
# trainer.train()

# # 🔹 Save the trained model
# logging.info("💾 Saving trained model...")
# output_dir = Path("crypto_bot_model")
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# logging.info("✅ Training completed! Model saved in ./crypto_bot_model")
# ****************_________________________________________



# import json
# import os
# import re
# import logging
# from pathlib import Path
# from typing import List, Dict, Any
# import torch
# import pandas as pd
# import numpy as np
# from datasets import Dataset
# import evaluate
# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template
# from trl import SFTTrainer
# from transformers import TrainingArguments, EvalPrediction
# from unsloth import is_bfloat16_supported

# # 🔹 Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
# )

# # 🔹 Load dataset
# DATA_PATH = "/root/twitter_codes_vennela/cleaned_crypto_questions.csv"
# logging.info("📂 Loading dataset...")

# df = pd.read_csv(DATA_PATH)
# df = df.iloc[:, :2]  # Keep only 'Crypto Query' and 'Answer' columns
# df.columns = ["Crypto Query", "Answer"]

# formatted_data = []
# for _, row in df.iterrows():
#     query = str(row["Crypto Query"]).strip().replace("\n", " ").replace('"', "'")
#     answer = str(row["Answer"]).strip().replace("\n", " ").replace('"', "'")

#     formatted_data.append({
#         "conversations": [
#             {"role": "user", "content": query},
#             {"role": "assistant", "content": answer}
#         ]
#     })


# # 🔹 Convert to Hugging Face dataset
# dataset = Dataset.from_list(formatted_data)

# # 🔹 Load pre-trained model & tokenizer
# MODEL_PATH = "/root/twitter_codes_vennela/new1"
# logging.info("🚀 Loading model from Unsloth...")

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=MODEL_PATH,
#     max_seq_length=2048,
#     dtype=None,  # Auto-select precision
#     load_in_4bit=True,  # 4-bit precision
# )

# # 🔹 Apply LoRA fine-tuning (Transfer Learning)
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     lora_alpha=16,
#     lora_dropout=0,
#     bias="none",
#     use_gradient_checkpointing="unsloth",
#     random_state=42,
# )

# # 🔹 Set chat template for tokenizer
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="chatml",
#     mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
#     map_eos_token=True,
# )

# # 🔹 Dataset Formatting
# def format_data(examples):
#     texts = [
#         tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
#         for conv in examples["conversations"]
#     ]
#     return {"text": texts}

# logging.info("✍️ Formatting dataset...")
# dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)

# # 🔹 Train/Validation Split
# train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
# train_dataset, eval_dataset = train_val_split["train"], train_val_split["test"]

# def compute_metrics(eval_pred: EvalPrediction):
#     predictions, labels = eval_pred
#     predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Load metrics using evaluate
#     bleu = evaluate.load("bleu")
#     rouge = evaluate.load("rouge")
#     exact_match = evaluate.load("accuracy")
#     f1 = evaluate.load("f1")

#     # Compute scores
#     bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
#     rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
#     exact_match_score = exact_match.compute(predictions=decoded_preds, references=decoded_labels)
#     f1_score = f1.compute(predictions=decoded_preds, references=decoded_labels)

#     return {
#         "bleu": bleu_score["bleu"],
#         "rougeL": rouge_score["rougeL"].mid.fmeasure,
#         "exact_match": exact_match_score["accuracy"],
#         "f1": f1_score["f1"]
#     }


# # 🔹 Training Configuration
# logging.info("⚙️ Setting up training arguments...")
# training_args = TrainingArguments(
#     output_dir="crypto_bot_model",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     warmup_steps=5,
#     learning_rate=2e-4,
#     fp16=not is_bfloat16_supported(),
#     bf16=is_bfloat16_supported(),
#     logging_steps=1,
#     num_train_epochs=3,
#     optim="adamw_8bit",
#     weight_decay=0.01,
#     lr_scheduler_type="linear",
#     seed=42,
#     evaluation_strategy="no",
#     save_strategy="steps",
#     save_steps=50,
# )

# # 🔹 Trainer Setup
# logging.info("🔥 Initializing Trainer...")
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     dataset_text_field="text",
#     max_seq_length=2048,
#     dataset_num_proc=2,
#     packing=False,
#     args=training_args,
#     compute_metrics=compute_metrics,  # Attach evaluation function
# )

# # 🔹 Train the Model
# logging.info("🚀 Starting training...")
# trainer.train()

# # 🔹 Save the trained model
# logging.info("💾 Saving trained model...")
# output_dir = Path("crypto_bot_model")
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# logging.info("✅ Training completed! Model saved in ./crypto_bot_model")



# import json
# import os
# import re
# import logging
# from pathlib import Path
# from typing import List, Dict, Any
# import torch
# import pandas as pd
# import numpy as np
# from datasets import Dataset
# import evaluate
# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template
# from trl import SFTTrainer
# from transformers import TrainingArguments, EvalPrediction
# from transformers.trainer_utils import get_last_checkpoint
# from unsloth import is_bfloat16_supported

# # 🔹 Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
# )

# # 🔹 Load dataset
# DATA_PATH = "/root/twitter_codes_vennela/cleaned_crypto_questions.csv"
# logging.info("📂 Loading dataset...")

# df = pd.read_csv(DATA_PATH)
# df = df.iloc[:, :2]  # Keep only 'Crypto Query' and 'Answer' columns
# df.columns = ["Crypto Query", "Answer"]

# formatted_data = []
# for _, row in df.iterrows():
#     query = str(row["Crypto Query"]).strip().replace("\n", " ").replace('"', "'")
#     answer = str(row["Answer"]).strip().replace("\n", " ").replace('"', "'")

#     formatted_data.append({
#         "conversations": [
#             {"role": "user", "content": query},
#             {"role": "assistant", "content": answer}
#         ]
#     })

# # 🔹 Convert to Hugging Face dataset
# dataset = Dataset.from_list(formatted_data)

# # 🔹 Load pre-trained model & tokenizer
# MODEL_PATH = "/root/twitter_codes_vennela/new1"
# logging.info("🚀 Loading model from Unsloth...")

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=MODEL_PATH,
#     max_seq_length=2048,
#     dtype=None,  # Auto-select precision
#     load_in_4bit=True,  # 4-bit precision
# )

# # 🔹 Apply LoRA fine-tuning (Transfer Learning)
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     lora_alpha=16,
#     lora_dropout=0,
#     bias="none",
#     use_gradient_checkpointing="unsloth",
#     random_state=42,
# )

# # 🔹 Set chat template for tokenizer
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="chatml",
#     mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
#     map_eos_token=True,
# )

# # 🔹 Dataset Formatting
# def format_data(examples):
#     texts = [
#         tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
#         for conv in examples["conversations"]
#     ]
#     return {"text": texts}

# logging.info("✍️ Formatting dataset...")
# dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)

# # 🔹 Train/Validation Split
# train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
# train_dataset, eval_dataset = train_val_split["train"], train_val_split["test"]

# def compute_metrics(eval_pred: EvalPrediction):
#     predictions, labels = eval_pred
#     predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Load metrics using evaluate
#     bleu = evaluate.load("bleu")
#     rouge = evaluate.load("rouge")
#     exact_match = evaluate.load("accuracy")
#     f1 = evaluate.load("f1")

#     # Compute scores
#     bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
#     rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
#     exact_match_score = exact_match.compute(predictions=decoded_preds, references=decoded_labels)
#     f1_score = f1.compute(predictions=decoded_preds, references=decoded_labels)

#     return {
#         "bleu": bleu_score["bleu"],
#         "rougeL": rouge_score["rougeL"].mid.fmeasure,
#         "exact_match": exact_match_score["accuracy"],
#         "f1": f1_score["f1"]
#     }


# # 🔹 Training Configuration
# logging.info("⚙️ Setting up training arguments...")
# training_args = TrainingArguments(
#     output_dir="crypto_bot_model",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     warmup_steps=5,
#     learning_rate=2e-4,
#     fp16=not is_bfloat16_supported(),
#     bf16=is_bfloat16_supported(),
#     logging_steps=1,
#     num_train_epochs=50,  # Increase epochs
#     optim="adamw_8bit",
#     weight_decay=0.01,
#     lr_scheduler_type="linear",
#     seed=42,
#     evaluation_strategy="no",
#     save_strategy="steps",
#     save_steps=50,
# )


# # 🔹 Check for the Last Checkpoint
# last_checkpoint = get_last_checkpoint("crypto_bot_model")
# if last_checkpoint:
#     logging.info(f"🔄 Resuming training from {last_checkpoint}")
# else:
#     logging.info("🚀 No checkpoint found. Starting training from scratch.")

# # 🔹 Trainer Setup
# logging.info("🔥 Initializing Trainer...")
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     dataset_text_field="text",
#     max_seq_length=2048,
#     dataset_num_proc=10,
#     packing=False,
#     args=training_args,
#     compute_metrics=compute_metrics,  # Attach evaluation function
# )

# # 🔹 Train the Model (Resume if Checkpoint Exists)
# logging.info("🚀 Starting training...")
# if last_checkpoint:
#     trainer.train(resume_from_checkpoint=last_checkpoint)

# else:
#     trainer.train()

# # 🔹 Save the trained model
# logging.info("💾 Saving trained model...")
# output_dir = Path("crypto_bot_model")
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# logging.info("✅ Training completed! Model saved in ./crypto_bot_model")





# import json
# import os
# import re
# import logging
# from pathlib import Path
# from typing import List, Dict, Any
# import torch
# import pandas as pd
# import numpy as np
# from datasets import Dataset
# import evaluate
# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template
# from trl import SFTTrainer
# from transformers import TrainingArguments, EvalPrediction
# from transformers.trainer_utils import get_last_checkpoint
# from unsloth import is_bfloat16_supported

# # 🔹 Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
# )

# # 🔹 Load dataset
# DATA_PATH = "/root/twitter_codes_vennela/cleaned_crypto_questions.csv"
# logging.info("📂 Loading dataset...")

# df = pd.read_csv(DATA_PATH)
# df = df.iloc[:, :2]  # Keep only 'Crypto Query' and 'Answer' columns
# df.columns = ["Crypto Query", "Answer"]

# formatted_data = []
# for _, row in df.iterrows():
#     query = str(row["Crypto Query"]).strip().replace("\n", " ").replace('"', "'")
#     answer = str(row["Answer"]).strip().replace("\n", " ").replace('"', "'")

#     formatted_data.append({
#         "conversations": [
#             {"role": "user", "content": query},
#             {"role": "assistant", "content": answer}
#         ]
#     })

# # 🔹 Convert to Hugging Face dataset
# dataset = Dataset.from_list(formatted_data)

# # 🔹 Load pre-trained model & tokenizer
# MODEL_PATH = "/root/twitter_codes_vennela/new1"
# logging.info("🚀 Loading model from Unsloth...")

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=MODEL_PATH,
#     max_seq_length=2048,
#     dtype=None,  # Auto-select precision
#     load_in_4bit=True,  # 4-bit precision
# )

# # 🔹 Apply LoRA fine-tuning (Transfer Learning)
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     lora_alpha=16,
#     lora_dropout=0,
#     bias="none",
#     use_gradient_checkpointing="unsloth",
#     random_state=42,
# )

# # 🔹 Set chat template for tokenizer
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="chatml",
#     mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
#     map_eos_token=True,
# )

# # 🔹 Dataset Formatting
# def format_data(examples):
#     texts = [
#         tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
#         for conv in examples["conversations"]
#     ]
#     return {"text": texts}

# logging.info("✍️ Formatting dataset...")
# dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)

# # 🔹 Train/Validation Split
# train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
# train_dataset, eval_dataset = train_val_split["train"], train_val_split["test"]

# def compute_metrics(eval_pred: EvalPrediction):
#     predictions, labels = eval_pred
#     predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Load metrics using evaluate
#     bleu = evaluate.load("bleu")
#     rouge = evaluate.load("rouge")
#     exact_match = evaluate.load("accuracy")
#     f1 = evaluate.load("f1")

#     # Compute scores
#     bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
#     rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
#     exact_match_score = exact_match.compute(predictions=decoded_preds, references=decoded_labels)
#     f1_score = f1.compute(predictions=decoded_preds, references=decoded_labels)

#     return {
#         "bleu": bleu_score["bleu"],
#         "rougeL": rouge_score["rougeL"].mid.fmeasure,
#         "exact_match": exact_match_score["accuracy"],
#         "f1": f1_score["f1"]
#     }


# # 🔹 Training Configuration
# logging.info("⚙️ Setting up training arguments...")
# training_args = TrainingArguments(
#     output_dir="crypto_bot_model1",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     warmup_steps=5,
#     learning_rate=2e-4,
#     fp16=not is_bfloat16_supported(),
#     bf16=is_bfloat16_supported(),
#     logging_steps=1,
#     num_train_epochs=50,  # Increase epochs
#     optim="adamw_8bit",
#     weight_decay=0.01,
#     lr_scheduler_type="linear",
#     seed=42,
#     evaluation_strategy="no",
#     save_strategy="steps",
#     save_steps=50,
# )


# # 🔹 Check for the Last Checkpoint
# last_checkpoint = get_last_checkpoint("crypto_bot_model1")
# if last_checkpoint:
#     logging.info(f"🔄 Resuming training from {last_checkpoint}")
# else:
#     logging.info("🚀 No checkpoint found. Starting training from scratch.")

# # 🔹 Trainer Setup
# logging.info("🔥 Initializing Trainer...")
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     dataset_text_field="text",
#     max_seq_length=2048,
#     dataset_num_proc=10,
#     packing=False,
#     args=training_args,
#     compute_metrics=compute_metrics,  # Attach evaluation function
# )

# # 🔹 Train the Model (Resume if Checkpoint Exists)
# logging.info("🚀 Starting training...")
# if last_checkpoint:
#     trainer.train(resume_from_checkpoint=last_checkpoint)

# else:
#     trainer.train()

# # 🔹 Save the trained model
# logging.info("💾 Saving trained model...")
# output_dir = Path("crypto_bot_model1")
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# logging.info("✅ Training completed! Model saved in ./crypto_bot_model1")






import json
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
import evaluate
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments, EvalPrediction, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from unsloth import is_bfloat16_supported

# 🔹 Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

# 🔹 Load dataset
DATA_PATH = "/root/twitter_codes_vennela/cleaned_crypto_questions.csv"
logging.info("📂 Loading dataset...")

df = pd.read_csv(DATA_PATH)
df = df.iloc[:, :2]  # Keep only 'Crypto Query' and 'Answer' columns
df.columns = ["Crypto Query", "Answer"]

formatted_data = []
for _, row in df.iterrows():
    query = str(row["Crypto Query"]).strip().replace("\n", " ").replace('"', "'")
    answer = str(row["Answer"]).strip().replace("\n", " ").replace('"', "'")

    formatted_data.append({
        "conversations": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ]
    })

# 🔹 Convert to Hugging Face dataset
dataset = Dataset.from_list(formatted_data)

# 🔹 Load pre-trained model & tokenizer
MODEL_PATH = "/root/twitter_codes_vennela/new1"
logging.info("🚀 Loading model from Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,  # Auto-select precision
    load_in_4bit=True,  # 4-bit precision
)

# 🔹 Apply LoRA fine-tuning (Transfer Learning)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.005,  # Prevent overfitting
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 🔹 Set chat template for tokenizer
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    map_eos_token=True,
)

# 🔹 Dataset Formatting
def format_data(examples):
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in examples["conversations"]
    ]
    return {"text": texts}

logging.info("✍️ Formatting dataset...")
dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)

# 🔹 Train/Validation Split
train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = train_val_split["train"], train_val_split["test"]

def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Load metrics using evaluate
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    exact_match = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    # Compute scores
    bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    exact_match_score = exact_match.compute(predictions=decoded_preds, references=decoded_labels)
    f1_score = f1.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_score["bleu"],
        "rougeL": rouge_score["rougeL"].mid.fmeasure,
        "exact_match": exact_match_score["accuracy"],
        "f1": f1_score["f1"]
    }

# 🔹 Training Configuration
logging.info("⚙️ Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="crypto_bot_model1",
    per_device_train_batch_size=2,
    warmup_steps=5,
    learning_rate=2e-4,  # Slightly increased learning rate
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    num_train_epochs=1,
    optim="adamw_8bit",
    weight_decay=0.01,  # Reduced weight decay
    evaluation_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    save_steps=5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=3407,
    report_to="none",
    max_grad_norm=1.0,  # Adding gradient clipping
)

# 🔹 Check for the Last Checkpoint
last_checkpoint = get_last_checkpoint("crypto_bot_model1")
if last_checkpoint:
    logging.info(f"🔄 Resuming training from {last_checkpoint}")
else:
    logging.info("🚀 No checkpoint found. Starting training from scratch.")

# 🔹 Trainer Setup
logging.info("🔥 Initializing Trainer...")
trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,     
            eval_dataset=eval_dataset,      
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

# 🔹 Train the Model (Resume if Checkpoint Exists)
logging.info("🚀 Starting training...")
if last_checkpoint:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# 🔹 Save the trained model
logging.info("💾 Saving trained model...")
output_dir = Path("crypto_bot_model1")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

logging.info("✅ Training completed! Model saved in ./crypto_bot_model1")
