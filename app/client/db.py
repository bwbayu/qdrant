# ==============================
# SQLite Setup
# ==============================
import datetime
import sqlite3


def init_db():
    conn = sqlite3.connect("chat_sessions.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            query TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        
    ''')
    c.execute('''
              CREATE TABLE IF NOT EXISTS source_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            link TEXT,
            description TEXT,
            start_offset_sec INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
                ''')
    conn.commit()
    conn.close()


def create_new_session():
    check = check_empty_summary()
    if check:
        print("Found empty summary sessions:", check)
        return check[0], check[1]
    conn = sqlite3.connect("chat_sessions.db")
    c = conn.cursor()
    title = f"Chat {datetime.datetime.now().strftime('%H:%M:%S')}"
    c.execute(
        "INSERT INTO sessions (title, query, summary) VALUES (?, ?, ?)", (title, "", ""))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id, title


def get_all_sessions():
    conn = sqlite3.connect("chat_sessions.db")
    c = conn.cursor()
    c.execute("SELECT id, title FROM sessions ORDER BY created_at DESC")
    sessions = c.fetchall()
    conn.close()
    print("All sessions:", sessions)
    return sessions


def update_session(session_id, query=None, summary=None, video_link: list = None):
    conn = sqlite3.connect("chat_sessions.db")
    c = conn.cursor()
    print("Updating session:", session_id)
    print("With summary:", summary)
    print("With video links:", video_link)
    if summary is not None:
        print("Executing update for summary")
        c.execute("UPDATE sessions SET query=?, summary=? WHERE id=?",
                  (query, summary, session_id))
    if video_link is not None:
        c.execute("DELETE FROM source_data WHERE session_id=?", (session_id,))
        print("Executing insert for video links")
        for video in video_link:
            link = video.get("link", "")
            description = video.get("transcription", "")
            start_offset_sec = video.get("start_offset_sec", "")
            c.execute("INSERT INTO source_data (session_id, link, description, start_offset_sec) VALUES (?, ?, ?, ?)",
                      (session_id, link, description, start_offset_sec))
    conn.commit()
    conn.close()


def check_empty_summary():
    conn = sqlite3.connect("chat_sessions.db")
    c = conn.cursor()
    c.execute("SELECT id, title FROM sessions WHERE summary IS NULL OR summary = ''")
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return None


def get_session(session_id):
    conn = sqlite3.connect("chat_sessions.db")
    c = conn.cursor()
    c.execute(
        "SELECT title, query, summary FROM sessions WHERE id=?", (session_id,))
    row = c.fetchone()

    # fetch video links
    c.execute(
        "SELECT link, description, start_offset_sec FROM source_data WHERE session_id=?", (session_id,))
    video_links = [r for r in c.fetchall()]
    conn.close()
    return row[0], row[1] if row else ("", ""), row[2] if row else ("", ""), video_links if video_links else []
