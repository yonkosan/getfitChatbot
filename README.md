# Getfit Chatbot 
## Overview
Having a chatbot has become an increasing need in the buisness environment in today's world but making a chatbot from scratch is a extensive task that depletes too much resource and funds. This is basic Fitness chatbot that utilizes pre-trained LLM fine tuned over our custom dataset to answer queries related fitness and health.

## Installation
* clone the rep
```bash
git clone https://github.com/yonkosan/getfitChatbot.git
cd getfitChatbot
```
* install required dependencies
```bash
pip install -r requirements.txt
```
## Usage
* run the streamlit app
```bash
streamlit run chatbot.py
```
* Interact with the chatbot by typing questions, such as:
    * "What are some practical tips for staying hydrated throughout the day?"
 
# Walkthrough of the code:
### **1. Importing Libraries**

```python
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
import json

```

- **`streamlit`**: Used to create a web-based interface for the chatbot.
- **`transformers`**: Hugging Face library for working with pre-trained models, tokenizers, and pipelines.
- **`json`**: For loading the dataset.

---

### **2. Function: `fine_tune_model`**

This function fine-tunes a base language model (`gpt2`) on a custom dataset.

### **Input Parameters:**

- `base_model`: Name of the base model (e.g., `'gpt2'`).
- `custom_dataset`: A list of dictionaries containing "instruction" (input) and "output" (expected response).

### **Steps:**

1. **Load Tokenizer and Model:**
    - Loads the tokenizer and model architecture for `gpt2`.
        
        ```python
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        
        ```
        
2. **Tokenize Dataset:**
    
    ```python
    instructions = [entry["instruction"] for entry in custom_dataset]
    outputs = [entry["output"] for entry in custom_dataset]
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    inputs = tokenizer(instructions, outputs, return_tensors='pt', truncation=True, padding=True)
    
    ```
    
    - Extracts "instruction" and "output" fields from the dataset.
    - Adds a `[PAD]` token to handle padding for sequence alignment.
    - Tokenizes the dataset, converting text into numerical format (`inputs['input_ids']`).
3. **Define Training Arguments:**
    
    ```python
    training_args = TrainingArguments(output_dir='./model', num_train_epochs=3, per_device_train_batch_size=16,
     per_device_eval_batch_size=64, eval_steps=400, save_steps=800, warmup_steps=500)
    
    ```
    
    - Sets hyperparameters for fine-tuning:
        - `output_dir`: Directory to save the model.
        - `num_train_epochs`: Number of training iterations.
        - `per_device_train_batch_size`: Number of samples per device during training.
        - `eval_steps` and `save_steps`: Frequency of evaluation and checkpoint saving.
        - `warmup_steps`: Gradual increase in learning rate for the first 500 steps.
4. **Trainer Initialization:**
    
    ```python
    trainer = Trainer(model=model, args=training_args, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), train_dataset=inputs['input_ids'])
    
    ```
    
    - `Trainer`: Automates the training process.
    - `data_collator`: Handles batching; here, Masked Language Modeling (MLM) is disabled.
    - `train_dataset`: Tokenized input sequences for training.
5. **Train the Model:**
    
    ```python
    trainer.train()
    
    ```
    
    - Fine-tunes the `gpt2` model on the dataset.

---

### **3. Function: `chat_with_bot`**

This function creates a chatbot interface using the fine-tuned model.

### **Input Parameter:**

- `model_path`: Path to the fine-tuned model directory.

### **Steps:**

1. **Load the Chatbot Pipeline:**
    
    ```python
    chat_pipeline = pipeline('text-generation', model=model_path)
    
    ```
    
    - Creates a text generation pipeline using the fine-tuned model.
2. **Setup Streamlit Interface:**
    
    ```python
    st.title("getfit chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    ```
    
    - Adds a title for the chatbot and initializes session state to store chat history.
3. **Display Chat History:**
    
    ```python
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    ```
    
    - Loops through previous messages stored in `st.session_state` to display chat history.
4. **Handle User Input:**
    
    ```python
    if prompt := st.chat_input("Hey, i am a fitness assistant, ask me anything"):
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    ```
    
    - Captures user input via `st.chat_input` and appends it to the session state.
5. **Generate and Display Response:**
    
    ```python
    with st.chat_message("assistant"):
        response = chat_pipeline(prompt)[0]['generated_text']
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    ```
    
    - Sends the user's input to the model for response generation.
    - Displays the response and adds it to the session history.

---

### **4. Main Execution**

```python
custom_dataset_path = "./dataset/fitness-chat-prompt-completion-dataset.json"
with open(custom_dataset_path, "r", encoding="utf-8") as file:
    custom_dataset = json.load(file)

base_model = 'gpt2'
fine_tune_model(base_model, custom_dataset)

model_path = './model'
chat_with_bot(model_path)

```

1. **Load Dataset:**
    - Reads a JSON dataset containing fitness-related instructions and completions.
2. **Fine-Tune Model:**
    - Calls `fine_tune_model` to customize the `gpt2` model.
3. **Run Chatbot:**
    - Launches the chatbot using `chat_with_bot`.

---
