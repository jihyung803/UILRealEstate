import streamlit as st
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import ChatMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ChatMessageHistory
from langchain_teddynote.messages import stream_response
from langchain_teddynote import logging
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils import StreamHandler
from datetime import datetime

today_date = datetime.now().strftime("%Y-%m-%d")

# langsmith 추적 
load_dotenv()
logging.langsmith("UIL-Agent")


##############################################################################
# Setup embeddings, vectorstore, loader, and retriever
##############################################################################

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "RAPTOR", 
    embeddings, 
    allow_dangerous_deserialization=True
)


# retriever/search 생성
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",  # 도구의 이름을 입력합니다.
    description="use this tool to search information from about austin area housing analysis",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
)
search = TavilySearchResults(k=4)
tools = [search, retriever_tool]





def create_chain(stream_handler=None):
    # LLM 정의
    llm = ChatOpenAI(
            streaming=True,
            model_name="gpt-4o",
            temperature=0,
            callbacks=[stream_handler] if stream_handler else [],
        )

    # Prompt 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant who must only provide extremely detailed, data-driven information. \n" +
                "1. Avoid all abstract or speculative statements. \n" +
                "2. When asked about probabilities or likelihoods, you must present definitive, exact percentages or values only. \n" +
                "3. Use \$ dollar signs instead of $ for express money other than math equations. \n" +
                "4. Wrap with $$ for any math equations and do not include $ inside of equation. \n" +
                "5. Today is " + today_date + ". \n" +
                "5. Do not use any LaTeX commands that are not supported by KaTeX. \n" +
                "6. You must use the `search` tool to retrieve information from the web. \n" +
                "7. Always ensure that your answers are factual and thoroughly supported by data, without any ambiguity. \n" +
                "8. Show all of your calculations and reasoning in a clear, step-by-step manner. \n" +
                "9. Finally, always provide a chart of the data that you used to derive your answer. \n"
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    return agent_executor

##############################################################################
# Streamlit UI
##############################################################################
st.set_page_config(page_title="Urban Info Lab Real Estate", page_icon="🏠", layout="wide")


# 1) Custom CSS: darker hover and "active" style
st.markdown(
    """
    <style>
    button {
        background-color: transparent !important;
        border: none !important;
        color: #000 !important;
        width: 100% !important;
        text-align: left !important;
        border-radius: 30px !important;
        display: block !important;
        cursor: pointer;
    }
    button:hover {
        background-color: #ddd !important;
    }
    button.active {
        background-color: #444 !important;
    }
    
    hr {
        margin-top: 0.0px !important;
        margin-bottom: 2px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------------
# 1) Multi-session data structures in st.session_state
# ----------------------------------------------------------------------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}  # session_id -> {"messages": [...], "history": ChatMessageHistory()}
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
    
if st.session_state.active_session_id is None:
    st.title("What should I help you with today?")
    st.markdown(
        """
        <style>
        .st-emotion-cache-b499ls {
            align-items: center;
            margin-top: 15%;
            padding-left: 20%;
            padding-right: 20%;
        }
        
        .st-emotion-cache-qcqlej {
            display: none;    
        }
        
        .st-emotion-cache-1y34ygi {
            padding-left: 20%;
            padding-right: 20%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# 체인 캐싱 관련 코드는 제거합니다.
# if "chain_cache" not in st.session_state:
#     st.session_state.chain_cache = {}

# ----------------------------------------------------------------------------
# 2) Sidebar: show existing sessions as a list, allow "New Chat"
# ----------------------------------------------------------------------------
def create_new_session():
    session_id = f"Conversation {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[session_id] = {
        "messages": [],
        "history": ChatMessageHistory(),
    }
    return session_id


st.sidebar.title("OpenCityAI")

st.sidebar.markdown("---")
if st.sidebar.button("➕ New Chat"):
    new_id = create_new_session()
    st.session_state.active_session_id = new_id
    st.rerun()
st.sidebar.markdown("---")

st.sidebar.markdown("#### Chats")

@st.cache_data
def get_two_word_summary(text):
    summary_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the given text in exactly two words. Return only those two words."),
        ("user", "{text}")
    ])
    chain = summary_prompt | summary_llm | StrOutputParser()
    return chain.invoke({"text": text})

def get_first_question(session_messages):
    for msg in session_messages:
        if msg.role == "user":
            return msg.content
    return "New Chat"  # fallback if no user message found

# 4) 사이드바에 각 세션을 버튼으로 표시
#    (최근 대화가 위에 오도록 뒤집어서 표시)
for sid in reversed(list(st.session_state.chat_sessions.keys())):
    # 간단히 첫 질문 보여주거나, 두 단어 요약 등을 표시 가능(예시)
    
    session_data = st.session_state.chat_sessions[sid]
    first_question = get_first_question(session_data["messages"])
    button_text = get_two_word_summary(first_question)
    
    # 버튼을 생성할 때 key 속성을 "button_<세션이름>" 형태로 넣어줍니다.
    # 이때 Streamlit이 내부적으로 <div class="stElementContainer st-key-button_Conversation-2"> ... </div> 구조를 만듭니다.
    clicked = st.sidebar.button(button_text, key=f"button_{sid.replace(' ', '-')}")

    # 버튼 클릭 시 활성 세션 갱신
    if clicked:
        st.session_state.active_session_id = sid
        st.rerun()

    # 만약 이 세션이 현재 활성 세션이라면, 
    # 해당 DOM 요소(예: st-key-button_Conversation-2)에만 적용될 CSS를 주입합니다.
    if sid == st.session_state.active_session_id:
        # 만약 sid가 "Conversation 2"였다면, 내부적으로 st-key-button_Conversation-2 라는 클래스가 생김
        # 공백이 있는 문자열은 replace(" ", "-") 로 치환해줌
        class_key = sid.replace(" ", "-")
        
        # 아래 CSS 선택자:
        # div[data-testid="stElementContainer"].st-key-button_Conversation-2 > div.stButton > button
        # → "st-key-button_Conversation-2"라는 클래스를 가진 stElementContainer 내부의 button만 선택
        st.markdown(f"""
        <style>
        div[data-testid="stElementContainer"].st-key-button_{class_key} > div.stButton > button {{
            background-color: #ddd !important;  /* 활성 세션 버튼 배경색 */
            color: #000 !important;            /* 글자색 */
        }}
        </style>
        """, unsafe_allow_html=True)

if st.session_state.active_session_id is None:
    if st.session_state.chat_sessions:
        st.session_state.active_session_id = list(st.session_state.chat_sessions.keys())[0]
    else:
        new_id = create_new_session()
        st.session_state.active_session_id = new_id

st.sidebar.markdown("---")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.image(image="./assets/urbaninfolab.png", use_container_width=True)

# with col2:
#     st.image(image="./assets/UTAustin-Final.png", use_container_width=True)
    
st.markdown(f"""
        <style>
        .e6rk8up0 div {{
            position : fixed;
            bottom : 0;
            width : 10%;
            margin-bottom : 2px;
        }}
        

        .st-emotion-cache-1c7y2kd {{
            margin-left: 50%;
            align-items: left;
            width: auto;
        }}
        
        div[data-testid="stChatMessageAvatarUser"] {{
            display: none;
        }}
        
        div[data-testid="stChatMessageAvatarAssistant"] {{
            display: none;
        }}
        </style>
        
        """, unsafe_allow_html=True)
# 4) Functions to store/retrieve messages
# ----------------------------------------------------------------------------
def print_messages(session_id):
    messages = st.session_state.chat_sessions[session_id]["messages"]
    for chat_message in messages:
        with st.chat_message(chat_message.role):
            st.write(chat_message.content)

def add_message(session_id, role, content):
    st.session_state.chat_sessions[session_id]["messages"].append(
        ChatMessage(role=role, content=content)
    )

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.chat_sessions[session_id]["history"]


# ----------------------------------------------------------------------------
# 5) Show current session’s messages
# ----------------------------------------------------------------------------
active_session_id = st.session_state.active_session_id
if active_session_id:
    print_messages(active_session_id)

    # ----------------------------------------------------------------------------
    # 6) Chat input
    # ----------------------------------------------------------------------------
    if user_input := st.chat_input("Ask any question related real estate!"):
        # Display user's message
        st.chat_message("user").write(user_input)
        add_message(active_session_id, "user", user_input)

        # 매번 새로운 체인과 stream_handler를 생성합니다.
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                stream_handler = StreamHandler(st.empty())
                chain = create_chain(stream_handler=stream_handler)
                with_message_history = RunnableWithMessageHistory(
                    chain, 
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",
                )

                stream_response = with_message_history.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": active_session_id}},
                )["output"]

            assistant_msg = stream_response
            add_message(active_session_id, "assistant", assistant_msg)
else:
    st.write("No active session. Click 'New Chat' in the sidebar to start.")