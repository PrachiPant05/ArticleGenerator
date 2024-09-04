import streamlit as st
from crewai import Agent
from tools import tool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables
load_dotenv()

# Initialize the LLM with the Google API key
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize the Writer agent
article_writer = Agent(
    role='Writer',
    goal='Narrate compelling tech stories about {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft "
        "engaging narratives that captivate and educate, bringing new "
        "discoveries to light in an accessible manner."
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=False
)

# Streamlit UI
st.title("Article Writer")

# Topic input
topic = st.text_input("Enter the topic you want to explore:", "")

# Run button
if st.button("Write Article"):
    if topic:
        # Generate the article
        result = article_writer.llm.predict(f"Write a compelling tech story about {topic}")
        
        st.write("### Article:")
        st.write(result)

        # Prepare Markdown content for download
        md_content = f"# {topic}\n\n{result}"
        
        # Download button
        st.download_button(
            label="Download Article as .md",
            data=md_content,
            file_name=f"{topic.replace(' ', '_')}.md",
            mime="text/markdown"
        )
    else:
        st.error("Please enter a topic to explore.")
