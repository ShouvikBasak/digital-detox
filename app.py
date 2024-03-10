import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

accelerator = Accelerator()

# Initialize the model from Hugging Face
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
torch.set_default_dtype(torch.float32)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# torch.set_default_device("cpu")
# torch.set_default_dtype(torch.float32)

##### App Design #####

# Title
st.title("Digital Detox")

# Sidebar title
st.sidebar.markdown("# User Information")

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Gender',
    ('Male', 'Female', 'Others')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select Age',
    13, 21, (15)
)

# Add text input
disclaimer_text = st.write("Disclaimer : Chatbot responses are not a substitute for professional medical advice or diagnosis,for any specific concerns, consult with a qualified expert")

# Add a checkbox
checkbox=st.checkbox("User Consent: By engaging with this chatbot, you consent to use it responsibly as a supplementary resource only")

######################

# Store the chat sessions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
# st.write(st.session_state.messages)
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# Function to generate chat output
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=200, eos_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = tokenizer.decode(model.generate(input_ids=tokenizer.encode(f"{prompt} {generated_text}", return_tensors="pt"), max_length=150, eos_token_id=tokenizer.eos_token_id))[0]
    return assistant_response

# User input
user_input = st.text_input("Enter your message:")

if user_input:
    st.write("Chat Assistant:", generate_response(user_input))
