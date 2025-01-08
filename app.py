# Import necessary libraries
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from Streamlit secrets or .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="AI News Generator - OOP Edition",
    page_icon="🧠"
)

# Title and description
st.title("🤖 AI News Generator, powered by CrewAI and Google Gemini 1.5 Flash")
st.markdown("Generate comprehensive blog posts using an advanced OOP structure for AI agents.")

# Sidebar input and settings
with st.sidebar:
    st.header("Content Settings")
    topic = st.text_area(
        "Enter your topic",
        height=100,
        placeholder="Enter the topic you want to generate content about..."
    )
    st.markdown("### Advanced Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    st.markdown("---")
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)

    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Enter your desired topic in the text area above.
        2. Adjust the temperature (higher = more creative).
        3. Click 'Generate Content' to start.
        4. Wait for the AI to generate your article.
        5. Download the result as a markdown file.
        """)


# Define a class-based OOP solution for the content generator
class ContentGenerator:
    def __init__(self, topic, temperature):
        self.topic = topic
        self.temperature = temperature
        self.serper_api_key = st.secrets['SERPER_API_KEY']  # Use st.secrets for Serper API key
        self.gemini_api_key = st.secrets['GEMINI_API_KEY']  # Use st.secrets for Gemini API key
        self.llm = self._initialize_llm()
        self.search_tool = self._initialize_search_tool()

    def _initialize_llm(self):
        """Initialize LLM with Google Gemini 1.5 Flash."""
        return LLM(
            model="gemini/gemini-1.5-flash",
            api_key=self.gemini_api_key,
            temperature=self.temperature
        )

    def _initialize_search_tool(self):
        """Initialize SerperDevTool with API key."""
        return SerperDevTool(api_key=self.serper_api_key, n_results=4)

    def _create_agents(self):
        """Create and configure AI agents."""
        senior_research_analyst = Agent(
            role="Senior Research Analyst",
            goal=f"Research, analyze, and synthesize comprehensive information on {self.topic} from reliable web sources",
            backstory="Expert in advanced web research with strong fact-checking and analysis skills.",
            allow_delegation=False,
            verbose=True,
            tools=[self.search_tool],
            llm=self.llm
        )

        content_writer = Agent(
            role="Content Writer",
            goal="Transform research findings into engaging blog posts while maintaining accuracy",
            backstory="Specializes in creating engaging, well-structured, and accurate content.",
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )

        return senior_research_analyst, content_writer

    def _create_tasks(self, agents):
        """Create and define tasks for agents."""
        research_task = Task(
            description=f"""
                1. Conduct comprehensive research on {self.topic} including:
                    - Recent developments and news
                    - Key industry trends and innovations
                    - Expert opinions and analyses
                    - Statistical data and market insights
                2. Fact-check and organize findings into a structured research brief.
                3. Include all relevant citations and sources.
            """,
            expected_output="""A detailed research report with an executive summary, 
                key findings, trends, statistics, and citations.""",
            agent=agents[0]  # Senior Research Analyst
        )

        writing_task = Task(
            description=f"""
                Using the research brief provided, create an engaging blog post:
                1. Transform technical information into accessible content.
                2. Maintain factual accuracy and citations.
                3. Include an attention-grabbing introduction, structured body, and compelling conclusion.
                4. Format content in markdown with inline citations.
            """,
            expected_output="""A polished blog post in markdown format with a reference section and hyperlinked citations.""",
            agent=agents[1]  # Content Writer
        )

        return research_task, writing_task

    def generate(self):
        """Generate content using CrewAI agents."""
        agents = self._create_agents()
        tasks = self._create_tasks(agents)

        # Initialize Crew and kickoff tasks
        crew = Crew(agents=agents, tasks=tasks, verbose=True)
        return crew.kickoff(inputs={"topic": self.topic})


# Main content generation logic
if generate_button:
    if topic.strip() == "":
        st.warning("Please enter a topic to generate content.")
    else:
        with st.spinner('Generating content... This may take a moment.'):
            try:
                generator = ContentGenerator(topic=topic, temperature=temperature)
                result = generator.generate()

                # Display the generated content
                st.markdown("### Generated Content")
                st.markdown(result)

                # Add download button
                st.download_button(
                    label="Download Content",
                    data=result.raw,
                    file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with CrewAI, Streamlit, and powered by Google Gemini 1.5 Flash")
