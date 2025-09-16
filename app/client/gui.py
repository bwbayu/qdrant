import gradio as gr
import pandas as pd
from app.client.db import init_db, get_all_sessions, create_new_session, get_session, update_session
import sqlite3
import datetime
from app.api.combine import summary_generation
from app.api.upload_data_pipeline import pipeline_process_files, handle_uploaded_image
# from app.client.example import summary_generation
MAX_VIDEOS = 5  # jumlah slot video yang kamu siapin

# ==============================
# File Processing
# ==============================


def display_video_from_url(video_url, start_time=3):
    """
    Menghasilkan HTML untuk menampilkan video dari URL pada detik tertentu.
    """
    # Cek apakah URL sudah diisi
    if not video_url:
        return "<p style='text-align:center; color:grey;'>Masukkan URL video untuk memulai</p>"

    # Tambahkan #t=<seconds> ke URL video
    video_url_with_time = f"{video_url}#t={start_time}"

    # Buat konten HTML dengan tag video
    html_content = f"""
    <div style="text-align: center;">
        <video width="100%" controls autoplay muted>
            <source src="{video_url_with_time}" type="video/mp4">
            Browser Anda tidak mendukung tag video.
        </video>
        <p style="margin-top: 8px;"><em>Memutar video dari detik ke-{start_time}.</em></p>
    </div>
    """
    return html_content


def process_files(files, session_id):
    if files is None:
        return None

    summary = summary_generation()
    if session_id:
        update_session(session_id, summary=summary)
    return summary


def open_image(url):
    try:
        from PIL import Image
        import requests
        from io import BytesIO
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print("Error loading image:", e)
        return None
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
                    img_input = gr.Image(type="pil", label="Upload Image", sources=["upload"], scale=1, height=100, width=100, placeholder="Click to Upload Image") 
                    
                img_input.upload(fn=handle_uploaded_image, inputs=[img_input, state_session_id], outputs=msg )

                summary = gr.Markdown("Summary will appear here...")
                # container kosong untuk video
                with gr.Column():
                    source_container = []
                    for i in range(MAX_VIDEOS):
                        with gr.Row() as slot:
                            v = gr.HTML(
                                visible=False, label=f"Video {i+1}", elem_id=f"myvideo-{i}")
                            t = gr.Markdown(
                                "", elem_classes="video-title", visible=False)
                            source_container.append(v)
                            source_container.append(t)
                    for i in range(MAX_VIDEOS):
                        with gr.Row() as slot:
                            v = gr.Image(
                                visible=False, label=f"Image {i+1}", elem_id=f"myimage-{i}")
                            t = gr.Markdown(
                                "", elem_classes="image-title", visible=False)
                            source_container.append(v)
                            source_container.append(t)
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
            for i in range(MAX_VIDEOS):
                updates.append(gr.update(value=None, visible=False))
                updates.append(gr.update(value="", visible=False))
            return session_id, gr.update(choices=sessions, value=f"{session_id}:{title}"), f"### {title}\n\n(no summary yet)", *updates

        new_chat_btn.click(
            start_new_chat,
            inputs=None,
            outputs=[state_session_id, sessions_dropdown,
                     summary, *source_container]
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
                    url, title_vid, start_offset_sec = vids[i]
                    if url.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
                        video = display_video_from_url(
                            url, start_time=start_offset_sec)
                        updates.append(
                            gr.update(value=video, visible=True))   # video
                        updates.append(
                            gr.update(value=f"**{title_vid}**", visible=True))  # title
                        continue
                updates.append(
                    gr.update(value=None, visible=False))  # video
                updates.append(
                    gr.update(value="", visible=False))    # t
                # save gambar
            for i in range(MAX_VIDEOS):
                if i < len(vids):
                    url, title_vid, start_offset_sec = vids[i]
                    if url.split('.')[-1] in ['png', 'jpg', 'jpeg'] and open_image(url) is not None:
                        updates.append(
                            gr.update(value=url, visible=True))   # video
                        updates.append(
                            gr.update(value=f"**{title_vid}**", visible=True))
                        continue  # title

                updates.append(
                    gr.update(value=None, visible=False))  # video
                updates.append(
                    gr.update(value="", visible=False))    # t
            return session_id, gr.update(value=query), f"### {title}\n\n{summ}", *updates

        sessions_dropdown.change(
            load_session,
            inputs=sessions_dropdown,
            outputs=[state_session_id, msg, summary, *source_container]
        )

        # Event: Upload Files
        # upload_button.upload(
        #     fn=process_files,
        #     inputs=[upload_button, state_session_id],
        #     outputs=summary
        # )

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
                # save video
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

                # save gambar
                for i in range(MAX_VIDEOS):
                    if i < len(vids) and vids[i].get("link", "").split('.')[-1] in ['png', 'jpg', 'jpeg'] and open_image(vids[i].get("link", "")) is not None:
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
                   state_session_id, msg], outputs=[summary, *source_container])

    return demo


def data_page():
    with gr.Blocks() as demo:
        file_input = gr.File(file_types=[".pdf", ".png", ".jpg", ".jpeg", ".mp4", ".mov", ".avi"], file_count="multiple")
        btn = gr.Button("Process Files")
        output = gr.HTML(label="Result")
        btn.click(pipeline_process_files, inputs=[file_input], outputs=[output])
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

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
