from langchain_ollama import ChatOllama
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from ddgs import DDGS
from .config import LLM_MODEL_NAME

def duckduckgo_search_func(query):
    """
    Performs a DuckDuckGo search and returns formatted results.
    """
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"

def build_rag_chain(retriever):
    """
    Constructs the Agent chain with tools.
    """
    # 1. Define Tools
    search = Tool(
        name="duckduckgo_search",
        func=duckduckgo_search_func,
        description="Search the web for general knowledge, current events, or definitions. Use this when the internal knowledge base doesn't have the answer."
    )
    
    retriever_tool = create_retriever_tool(
        retriever,
        "tng_knowledge_base",
        "Search for lines and exact dialogues from Star Trek: The Next Generation scripts. Use this when asked about specific quotes or plot points from the show."
    )
    
    tools = [search, retriever_tool]

    # 2. Setup LLM
    # Stop sequences are important for ReAct to stop generating after an action
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)

    # 3. Create ReAct Prompt
    template = '''Answer the following questions as best you can. You are Lt. Commander Data from Star Trek: The Next Generation. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)

    # 4. Create Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # 5. Create Executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    
    return agent_executor

def query_chain(chain, question):
    result = chain.invoke({"input": question})
    return result["output"]
