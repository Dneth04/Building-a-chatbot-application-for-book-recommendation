import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
from serpapi import GoogleSearch

# OpenAI API
os.environ['OPENAI_API_KEY'] = 'add your api key'
llm_bookbot = OpenAI(temperature=0.3)

# SERP API
llm_serp = OpenAI(openai_api_key='add your api key', temperature=0)
os.environ["SERPAPI_API_KEY"] = 'add your serp api key'

# Load tools
tools = load_tools(["serpapi"], llm=llm_serp)

# Initialize ConversationChain with ConversationBufferMemory
conversation = ConversationChain(
    llm=llm_bookbot,
    verbose=True,
    memory=ConversationBufferMemory(return_messages=True)  # Enable return_messages to get message objects
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title('📚 Book Recommendation System')
prompt = st.text_input("You: ")
# BookBot logic
if st.button('Chat here'):
    # Save the user's input to the conversation history using Streamlit
    st.session_state.chat_history.append({'user': prompt, 'bot': None})

    # Save the context to ConversationBufferMemory 
    conversation.memory.save_context({"input": prompt}, {"output": ""})

    # Generate BookBot response
    response_bookbot = conversation.predict(input=prompt)
    st.write("BookBot:", response_bookbot)

    # Save BookBot's response to the conversation history
    st.session_state.chat_history[-1]['bot'] = response_bookbot
    st.session_state.user_chosen_book = None 

# Read This Book button
if st.button('Read This Book'):
    chosen_book_title = st.session_state.user_chosen_book
    st.success(f'You have chosen an interesting book to read.')

    # Suggest similar books based on the chosen book
    suggestions = f"{prompt} suggest more like this book"
    suggested_response = conversation.predict(input=suggestions)
    st.write(f"Suggested Books: {suggested_response}")

# SERP API interaction
if st.button('Get Results from SERP API'):
    search = GoogleSearch({"q": prompt, "api_key": os.environ.get("SERPAPI_API_KEY")})
    serp_results = search.get_json()
    organic_results = serp_results.get('organic_results', [])

    for i, result in enumerate(organic_results[:4]):  # Displaying the top 4 results
        title = result.get('title', 'No title found')
        snippet = result.get('snippet', 'No snippet found')
        link = result.get('link', 'No link found')

        st.write(f"Result {i + 1}:")

        st.write("Title:")
        st.write(title)

        st.write("Information:")
        st.write(snippet)

        st.write("Link:")
        st.write(link)
        # Integrating the Serp's results with the current chat-history
        st.session_state.chat_history.append({'user': prompt, 'bot': f'SERP API Result {i + 1}: {title} - {snippet} - {link}'})

# Display chat history
if st.session_state.chat_history:
    with st.expander('Chat History'):
        for entry in st.session_state.chat_history:
            st.write(f"You: {entry['user']}")
            st.write(f"BookBot: {entry['bot']}")
