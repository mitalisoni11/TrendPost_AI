import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Set OpenAI key explicitly
openai.api_key = openai_api_key

# Streamlit UI
st.set_page_config(page_title="AI Social Media Content Generator", layout="wide")

st.title("TrendPost AI")
st.markdown("Generate platform-specific, engaging social media posts instantly!")

# Sidebar
with st.sidebar:
    st.header("Post Settings")
    topic = st.text_area("Enter Topic", height=100, placeholder="e.g., Indian Union Budget")
    platform = st.selectbox("Choose Platform", ["Instagram", "Twitter", "LinkedIn"])
    generate_button = st.button("Generate Post", type="primary", use_container_width=True)


def generate_post(topic, platform):
    # Set up LLM
    llm = LLM(model="gpt-3.5-turbo", temperature=0.7)

    # Search Tool
    search_tool = SerperDevTool(api_key=serper_api_key, n_results=10)

    # Research Agent
    research_agent = Agent(
        role="Social Media Researcher",
        goal="Find the latest trending news related to {topic}",
        backstory="You are an expert with 10 years of experience in social media trends and digital marketing. "
                  "You analyze news and social media trends to provide viral content ideas for various topics."
                  "You excel at finding, analyzing and synthesizing relevant information across the internet"
                  "using search tools. You make sure that your analysis contains raw data as well as information"
                  "one can derive from it that might make the content interesting and appealing to the masses.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    # Content Writer Agent
    content_agent = Agent(
        role="Social Media Content Creator",
        goal="Create engaging and viral social media posts for {platform} about {topic}.",
        backstory="You are a creative social media writer with expertise in writing "
                  "engaging content with a knack to make content witty and appealing."
                  "You keep the tone of the post professional when social media platform is LinkedIn."
                  "You write an engaging tweet when the selected platform is Twitter."
                  "A witty caption pertaining to the topic when the selected platform is Instagram."
                  "Your content is optimized for each platform depending on the word limit "
                  "and recent trends on that particular social media platform.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Research Task
    research_task = Task(
        description="1. Conduct comprehensive research on {topic} including:"
                    "- Recent developments and news"
                    "- Key industry trends and innovations"
                    "2. Organize findings into a structured research brief",
        expected_output="A summary of recent trends, events, and key talking points about {topic}.",
        agent=research_agent
    )

    # Content Writing Task
    writing_task = Task(
        description="1. Create an engaging social media post for {platform} using research insights."
                    "2. Create a post that might be most relevant for the social media {platform}"
                    "3. The level of detail in the post will depend on what is most popular on that "
                    "particular social media platform.",
        expected_output="A social media post optimized for {platform} containing"
                        "- relevant content with hashtags and emojis.",
        agent=content_agent
    )

    # Create Crew

    crew = Crew(
        agents=[research_agent, content_agent],
        tasks=[research_task, writing_task],
        verbose=True
    )

    result = crew.kickoff(inputs={"topic": topic, "platform": platform})

    # Extract the final text output
    if isinstance(result, list):  # Sometimes CrewAI returns a list
        result_text = "\n\n".join(str(res) for res in result)
    else:
        result_text = str(result)  # Convert CrewOutput to a string

    return result_text


# Generate Content
if generate_button:
    with st.spinner('Generating content...'):
        try:
            post = generate_post(topic, platform)
            st.subheader(f"Generated {platform} Post")
            st.write(post)

            # Copy to clipboard button
            st.code(post, language="markdown")

            # Download Button
            st.download_button("Download Post", data=post, file_name="social_media_post.txt", mime="text/plain")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
