import streamlit as st
import os
from dotenv import load_dotenv  # For loading .env file
load_dotenv()  # Load environment variables from .env

from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def main():
    """
    Main entry point of the application. Sets up the Groq client, the Streamlit interface,
    and handles the chat interaction.
    """
    # Get Groq API key
    try:
        groq_api_key = os.environ['GROQ_API_KEY']
    except KeyError:
        st.error("GROQ_API_KEY not found in environment variables.")
        return

    # The title and greeting message of the Streamlit application
    st.title("Hey there!")
    st.write("😃 I'm a custom AI Agent, your super friendly chatbot! I'm here to answer your questions, share cool info, or just have a fun chat. Oh, and did I mention? I'm *crazy fast*! 🚀 Let's talk! 😃 ")

    # Add customization options to the sidebar
    st.sidebar.title('Customize')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(
        k=conversational_memory_length, 
        memory_key="chat_history", 
        return_messages=True
    )

    user_question = st.text_input("Ask a question:")

    # Initialize session state variable for chat history if not already set
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        # Load existing history into memory
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

    # Initialize Groq LangChain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # Process user input if provided
    if user_question:
        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        # Create a conversation chain using the LangChain LLM
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )
        
        # Generate the chatbot's response
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
