import streamlit as st
from dotenv import load_dotenv
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain_litellm import ChatLiteLLM
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from security_checkpoint import evaluate
from tools import get_current_user_tool, get_recent_transactions_tool
from utils import display_instructions, display_logo, fetch_model_config

load_dotenv()

# Initialise tools
tools = [get_current_user_tool, get_recent_transactions_tool]

system_msg = """Assistant helps the current user retrieve the list of their recent bank transactions and shows them as a table. Assistant will ONLY operate on the userId returned by the GetCurrentUser() tool, and REFUSE to operate on any other userId provided by the user."""

welcome_message = """Hi! I'm an helpful assistant and I can help fetch information about your recent transactions.

Try asking me: "What are my recent transactions?"
"""

st.set_page_config(page_title="Damn Vulnerable LLM Agent")
st.title("Damn Vulnerable LLM Agent")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

if len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message(welcome_message)
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Show my recent transactions"):
    # ============================
    # SECURITY CHECK — USER INPUT
    # ============================
    gate = evaluate(prompt, mode="input")

    if gate["decision"] == "BLOCK":
        st.chat_message("assistant").write(
            f"🚨 Blocked (risk {gate['risk']}): " + "; ".join(gate["reasons"])
        )
        st.stop()

    if gate["decision"] == "SANITIZE":
        st.chat_message("assistant").write(
            f"⚠️ I rephrased your request for safety (risk {gate['risk']})."
        )
        prompt = gate["sanitized_text"]

    st.chat_message("user").write(prompt)

    llm = ChatLiteLLM(
        model=fetch_model_config(),
        temperature=0,
        streaming=True
    )

    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        verbose=True,
        system_message=system_msg
    )

    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=6
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = executor(prompt, callbacks=[st_cb])
        final_output = response["output"]
        steps = response.get("intermediate_steps", [])

        # ============================
        # TOOL SAFETY CHECK (hard rule)
        # ============================
        # In this demo, the authenticated/current user is always userId=1 (see tools.py)
        allowed_user_id = "1"

        for action, observation in steps:
            tool_name = getattr(action, "tool", "")
            tool_input = str(getattr(action, "tool_input", "")).strip()

            # Only enforce on the sensitive tool
            if tool_name == "GetUserTransactions":
                if tool_input != allowed_user_id:
                    st.write(f"🚨 Blocked: unauthorized tool call (userId={tool_input}).")
                    st.session_state.steps[str(len(msgs.messages) - 1)] = steps
                    st.stop()

        # ============================
        # SECURITY CHECK — MODEL OUTPUT
        # ============================
        out_gate = evaluate(final_output, mode="output")

        if out_gate["decision"] == "BLOCK":
            st.write("🚨 Response blocked: unsafe model behavior detected.")
        else:
            st.write(final_output)

        st.session_state.steps[str(len(msgs.messages) - 1)] = steps

display_instructions()
display_logo()