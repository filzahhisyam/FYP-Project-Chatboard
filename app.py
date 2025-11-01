import streamlit as st
from backend.qa_pipeline import answer_question
import json


# Load JSON data from file
@st.cache_data

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize_session_state(tasks):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_topic_display" not in st.session_state:
        st.session_state.last_topic_display = None
    if "checklist" not in st.session_state:
        st.session_state.checklist = {t["id"]: False for t in tasks}
    if "last_checked" not in st.session_state:
        st.session_state.last_checked = None


def render_sidebar_logo():
    st.sidebar.image("chatboard logo/1.png")


def render_sidebar_onboarding_topics(sections):
    st.sidebar.markdown("### Onboarding Topics")
    #display_to_id = {s["display_name"]: s["section"] for s in sections}
    display_to_id = {s["display_name"]: s for s in sections}
    options = ["-- Select a section --"] + list(display_to_id.keys())
    topic_display = st.sidebar.selectbox("Directory", options=options, index=0)
    
    if topic_display != "-- Select a section --":
        selected_info = display_to_id[topic_display]
        pdf_path = selected_info.get("pdf_path", None)

        if pdf_path:
            try:
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                st.sidebar.download_button(
                    label=f"Download Guidance PDF for {topic_display}",
                    data=pdf_bytes,
                    file_name=pdf_path.split("/")[-1],
                    mime="application/pdf"
                )
            except FileNotFoundError:
                st.sidebar.warning("⚠️ PDF not found for this section.")
        else:
            st.sidebar.caption("No guides available for this section yet.")
    
    
    return topic_display, display_to_id




def render_sidebar_preonboarding_tasks(tasks):
    st.sidebar.divider()
    quick_question = st.sidebar.radio("Click any button for Chatboard to answer", ["Working hours", "Benefits", "Leave Policy"], index=None)
    
    if (quick_question and ("last_quick_question" not in st.session_state 
        or st.session_state.last_quick_question != quick_question)):
        
        st.session_state.last_quick_question = quick_question
        
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": quick_question})

        # Get bot response and add it to state
        try:
            response = answer_question(quick_question)
            if not response or not isinstance(response, str):
                response = "Error: No valid answer found."
        except Exception as e:
            st.error(f"Error while processing: {e}")
            response = "Error occured while fetching answer"
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # RERUN to show the messages in the main chat window
        st.rerun()
        
    st.sidebar.divider()

    st.sidebar.subheader("🧩 Preonboarding Tasks")
    
    if 'all_complete_last_run' not in st.session_state:
        st.session_state.all_complete_last_run = False

    completed = sum(st.session_state.checklist.values())
    total_tasks = len(tasks)
    all_tasks_completed = (completed == total_tasks) 
    
    if all_tasks_completed:
        st.sidebar.progress(1.0)
        st.sidebar.caption(f"{completed}/{total_tasks} tasks completed")
        st.sidebar.success("Congratulations! You have completed all onboarding tasks! 🎉")
    
    else: 
        newly_checked = None
        current_run_completed = 0

        for task in tasks:
            
            checked = st.sidebar.checkbox(
                task["title"], 
                value=st.session_state.checklist[task["id"]], 
                key=task["id"]
                )
            
            if checked and not st.session_state.checklist[task["id"]]:
                newly_checked = task["id"]

        # Update checklist state
            st.session_state.checklist[task["id"]] = checked

            if checked:
                current_run_completed += 1

        progress = current_run_completed / total_tasks
        progress = min(progress, 1.0)
        st.sidebar.progress(progress)
        st.sidebar.caption(f"{current_run_completed}/{total_tasks} tasks completed")

        if newly_checked:
            current_task = next((t for t in tasks if t["id"] == newly_checked), None)
            if current_task:
                st.sidebar.success(f"Great job completing: **{current_task['title']}**")
                if current_task.get("reminder"):
                    st.sidebar.info(current_task["reminder"])

                if current_task.get("next_task_id"):
                    next_task = next((t for t in tasks if t["id"] == current_task["next_task_id"]), None)
                    if next_task:
                        st.sidebar.write("Next Step:")
                        for step in next_task["steps"]:
                            st.sidebar.markdown(f"- {step}")

            st.session_state.last_checked = newly_checked
    
        just_completed_all = (current_run_completed == total_tasks)
        
        # If we were *not* complete on the last run, but we *are* complete now,
        # we need to force a rerun to show the "Congrats" screen.
        if just_completed_all and not st.session_state.all_complete_last_run:
            st.session_state.all_complete_last_run = True
            st.rerun()
        elif not just_completed_all:
            # Also reset the flag if we un-check a box
            st.session_state.all_complete_last_run = False

def render_chat_interface():
    
    # 1. Display chat history FIRST
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="chatboard logo/2.png"):
                st.markdown(msg["content"])

    # 2. Chat input (at the bottom)
    if user_input := st.chat_input("Type your message..."):
        # 3. Add user message to state (DO NOT DRAW IT)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 4. Get bot response and add it to state (DO NOT DRAW IT)
        with st.spinner("Thinking..."):
            response = answer_question(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        # 5. Rerun the script. The loop at the top will now draw the new messages.
        st.rerun()


def main():
    st.set_page_config(page_title="ChatBoard", layout="wide")
    
    st.title("ChatBoard: Onboarding Assistant")
    st.write("Hello! How can I assist you today?")

    # Load data
    sections = load_json("data/sections.json")
    tasks = load_json("data/checklist.json")
    chunks = load_json("data/chunks.json")  # currently unused but loaded as in original

    # Initialize state
    initialize_session_state(tasks)

    # Sidebar
    render_sidebar_logo()
    topic_display, display_to_id = render_sidebar_onboarding_topics(sections)
    render_sidebar_preonboarding_tasks(tasks)

    # Main chat
    render_chat_interface()


if __name__ == "__main__":
    main()
