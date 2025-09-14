import gradio as gr
import pandas as pd
from db import init_db, get_all_sessions, create_new_session, get_session, update_session
import sqlite3
import datetime
from app.api.combine import summary_generation
# from app.client.example import summary_generation
MAX_VIDEOS = 5  # jumlah slot video yang kamu siapin

# ==============================
# File Processing
# ==============================


def process_files(files, session_id):
    if files is None:
        return None

    summary = summary_generation()
    if session_id:
        update_session(session_id, summary=summary)
    return summary


# ==============================
# UI
# ==============================
def chat_page():
    with gr.Blocks(js="""
    () => {
    console.log("Seeked video:sdfasdfas");
    const observer = new MutationObserver(() => {
        const vids = document.querySelectorAll('[id^="myvideo-"] video');
        vids.forEach(v => {
            if (!v._hasStartTime) {
                v._hasStartTime = true;

                v.addEventListener("canplay", () => {
                    console.log("Video ready:", v.src, "duration:", v.duration);
                    v.currentTime = 2;
                    console.log("Current time after set:", v.currentTime);
                });
            }
        });
    });

    observer.observe(document.body, { childList: true, subtree: true });
    }
    """) as demo:

        state_session_id = gr.State(value=None)

        with gr.Row():
            # Sidebar
            with gr.Column(scale=1):
                gr.Markdown("### Chatbot")

                new_chat_btn = gr.Button("➕ New Chat")
                sessions = [f"{sid}:{title}" for sid,
                            title in get_all_sessions()]
                sessions_dropdown = gr.Dropdown(
                    choices=sessions if sessions else ["No sessions"],
                    value=sessions[0] if sessions else "No sessions",
                    label="Chats",
                    interactive=True  # kalau kosong → non-interaktif
                )
                # sessions_dropdown = gr.Dropdown(
                #     choices=[f"{sid}:{title}" for sid,
                #              title in get_all_sessions()],
                #     label="Chats")

            # Main Chat Area
            with gr.Column(scale=4):
                gr.Markdown("## Let's get started!")

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message, or upload a file ...", scale=4, label="Type your message")
                    upload_button = gr.UploadButton(
                        "📂 Upload Files",
                        file_count="multiple"
                    )

                summary = gr.Markdown("Summary will appear here...")
                # container kosong untuk video
                with gr.Column():
                    video_container = []
                    for i in range(MAX_VIDEOS):
                        with gr.Row() as slot:
                            v = gr.Video(
                                visible=False, label=f"Video {i+1}", elem_id=f"myvideo-{i}")
                            t = gr.Markdown(
                                "", elem_classes="video-title", visible=False)
                            video_container.append(v)
                            video_container.append(t)
                gr.HTML("""
                <script>
                console.log("Seeked video:sdfasdfas");
                const observer = new MutationObserver(() => {
                    const vids = document.querySelectorAll('[id^="myvideo-"] video');
                    vids.forEach(v => {
                        if (!v._hasStartTime) {
                            v._hasStartTime = true;
                            v.addEventListener("loadeddata", () => {
                                setTimeout(() => {
                                    v.currentTime = 2;
                                    console.log("Seeked video:", v.src, "->", v.currentTime);
                                }, 300);
                            });
                        }
                    });
                });
                observer.observe(document.body, { childList: true, subtree: true });
                </script>
                """)
        # Event: New Chat

        def start_new_chat():

            session_id, title = create_new_session()
            sessions = [f"{sid}:{t}" for sid, t in get_all_sessions()]
            updates = []
            for i in range(MAX_VIDEOS):
                updates.append(gr.update(value=None, visible=False))
                updates.append(gr.update(value="", visible=False))
            return session_id, gr.update(choices=sessions, value=f"{session_id}:{title}"), f"### {title}\n\n(no summary yet)", *updates

        new_chat_btn.click(
            start_new_chat,
            inputs=None,
            outputs=[state_session_id, sessions_dropdown,
                     summary, *video_container]
        )

        # Event: Select session
        def load_session(choice):
            if not choice:
                return None, "No session selected", None

            print("Loading session:", choice)
            print("Type of choice:", type(choice))
            print("Full choice string:", choice.split(":"))
            print("session id part:", choice.split(":")[0])
            session_id = int(choice.split(":")[0])
            title, query, summ, vids = get_session(session_id)
            updates = []
            for i in range(MAX_VIDEOS):
                if i < len(vids):
                    url, title_vid = vids[i]
                    updates.append(
                        gr.update(value=url, visible=True))   # video
                    updates.append(
                        gr.update(value=f"**{title_vid}**", visible=True))  # title
                else:
                    updates.append(
                        gr.update(value=None, visible=False))  # video
                    updates.append(
                        gr.update(value="", visible=False))    # t
            return session_id, gr.update(value=query), f"### {title}\n\n{summ}", *updates

        sessions_dropdown.change(
            load_session,
            inputs=sessions_dropdown,
            outputs=[state_session_id, msg, summary, *video_container]
        )

        # Event: Upload Files
        upload_button.upload(
            fn=process_files,
            inputs=[upload_button, state_session_id],
            outputs=summary
        )

        # Event: Save Video Link
        def save_generate(session_id, msg):

            result = summary_generation(msg)
            summary = result.get("report_title", "") + \
                "\n" + result.get("summary", "")
            vids = result.get("lists", [])
            print("save generate link:", session_id, summary, vids)
            updates = []
            if session_id:
                update_session(session_id, msg,
                               summary=summary, video_link=vids)
                for i in range(MAX_VIDEOS):
                    if i < len(vids) and vids[i].get("embedding_scope", "") == "clip":
                        url = vids[i].get("link", "")
                        transcription = vids[i].get("transcription", "")
                        updates.append(
                            gr.update(value=url, visible=True))   # video
                        updates.append(
                            gr.update(value=f"**{transcription}**", visible=True))  # title
                    else:
                        updates.append(
                            gr.update(value=None, visible=False))  # video
                        updates.append(
                            gr.update(value="", visible=False))    # t
            return summary, *updates

        msg.submit(save_generate, inputs=[
                   state_session_id, msg], outputs=[summary, *video_container])

    return demo


def data_page():
    with gr.Blocks() as demo:
        gr.Markdown("## 📂 Data Page")
        gr.File()
    return demo


# Init DB
init_db()

# App
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Chat"):
            chat_page()
        with gr.Tab("Data"):
            data_page()

demo.launch(server_name="0.0.0.0", server_port=7860)
