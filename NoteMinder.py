import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import json
import os
import datetime
import threading
from ctypes import windll # Only works on Windows

# --- AI Configuration ---
# To use AI features, set your API key in environment variables or hardcode it here (not recommended for sharing)
# os.environ["API_KEY"] = "YOUR_API_KEY_HERE"

HAS_GENAI = False
try:
    import google.generativeai as genai
    api_key = os.environ.get("API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        HAS_GENAI = True
except ImportError:
    pass

# --- Constants ---
DATA_FILE = "noteminder_data.json"
DEFAULT_COLOR = "#fef3c7"

class Note:
    def __init__(self, id, content, created_at, event_time=None, location=None, status="PENDING", reminder=False, color=DEFAULT_COLOR, x=100, y=100, is_pinned=False):
        self.id = id
        self.content = content
        self.created_at = created_at
        self.event_time = event_time
        self.location = location
        self.status = status
        self.reminder = reminder
        self.color = color
        self.x = x
        self.y = y
        self.is_pinned = is_pinned

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self):
        return self.__dict__

class StickyNoteWindow(tk.Toplevel):
    def __init__(self, master, note, on_update, on_close):
        super().__init__(master)
        self.note = note
        self.on_update = on_update
        self.on_close = on_close
        
        # Remove window decorations (borderless)
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        self.geometry(f"250x200+{int(note.x)}+{int(note.y)}")
        self.configure(bg=note.color)

        # Add a subtle border
        self.frame_border = tk.Frame(self, bg="#999", bd=1)
        self.frame_border.pack(fill=tk.BOTH, expand=True)
        self.inner_frame = tk.Frame(self.frame_border, bg=note.color)
        self.inner_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Drag bar (Header)
        self.header = tk.Frame(self.inner_frame, bg="black", height=20)
        self.header.pack(fill=tk.X)
        self.header.bind("<Button-1>", self.start_move)
        self.header.bind("<B1-Motion>", self.do_move)
        
        # Make the semi-transparent tape look
        tk.Label(self.header, bg="#555", fg="white", text="::::", font=("Arial", 6)).pack(pady=2)
        
        # Close button (Unpin)
        btn_close = tk.Label(self.header, text="x", bg="black", fg="white", cursor="hand2")
        btn_close.pack(side=tk.RIGHT, padx=5)
        btn_close.bind("<Button-1>", lambda e: self.close_note())

        # Content Area
        self.text_area = tk.Text(self.inner_frame, bg=note.color, font=("Segoe Print", 10), bd=0, wrap=tk.WORD, height=5)
        self.text_area.insert(1.0, note.content)
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text_area.bind("<<Modified>>", self.on_text_change)

        # Info Footer
        self.footer = tk.Label(self.inner_frame, text=self.get_footer_text(), bg=note.color, fg="#555", font=("Arial", 7), justify=tk.LEFT, anchor="w")
        self.footer.pack(fill=tk.X, padx=5, pady=2)

        # Resize grip (bottom right)
        self.grip = tk.Label(self.inner_frame, text="◢", bg=note.color, fg="#999", cursor="sizing")
        self.grip.place(relx=1.0, rely=1.0, anchor="se")
        self.grip.bind("<Button-1>", self.start_resize)
        self.grip.bind("<B1-Motion>", self.do_resize)

        self._drag_data = {"x": 0, "y": 0}

    def get_footer_text(self):
        parts = []
        if self.note.event_time: parts.append(f"⏰ {self.note.event_time}")
        if self.note.location: parts.append(f"📍 {self.note.location}")
        return " | ".join(parts)

    def start_move(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        self.lift() # Bring to front

    def do_move(self, event):
        deltax = event.x - self._drag_data["x"]
        deltay = event.y - self._drag_data["y"]
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"+{x}+{y}")
        self.note.x = x
        self.note.y = y
        self.on_update(self.note)

    def start_resize(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def do_resize(self, event):
        deltax = event.x - self._drag_data["x"]
        deltay = event.y - self._drag_data["y"]
        w = self.winfo_width() + deltax
        h = self.winfo_height() + deltay
        if w > 100 and h > 100:
            self.geometry(f"{w}x{h}")

    def on_text_change(self, event=None):
        if self.text_area.edit_modified():
            self.note.content = self.text_area.get(1.0, "end-1c")
            self.on_update(self.note)
            self.text_area.edit_modified(False)

    def close_note(self):
        self.on_close(self.note)

class NoteMinderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NoteMinder Local")
        self.geometry("900x600")
        
        # Data
        self.notes = []
        self.load_data()
        self.windows = {} # Map note_id -> Toplevel

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        # --- Toolbar ---
        toolbar = tk.Frame(self, bg="#f0f0f0", height=50)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        
        self.entry_content = tk.Entry(toolbar, width=40, font=("Segoe UI", 11))
        self.entry_content.pack(side=tk.LEFT, padx=10, pady=10)
        self.entry_content.bind("<Return>", lambda e: self.add_note())

        btn_add = tk.Button(toolbar, text="Add Note", command=self.add_note, bg="#333", fg="white", relief=tk.FLAT)
        btn_add.pack(side=tk.LEFT, padx=5)

        if HAS_GENAI:
            btn_ai = tk.Button(toolbar, text="✨ AI Add", command=self.add_note_ai, bg="indigo", fg="white", relief=tk.FLAT)
            btn_ai.pack(side=tk.LEFT, padx=5)

        btn_bg = tk.Button(toolbar, text="🎨 Desktop Color", command=self.choose_bg, relief=tk.FLAT)
        btn_bg.pack(side=tk.RIGHT, padx=10)

        # --- Main Notebook Area (Treeview) ---
        cols = ("Content", "Created", "Time", "Location", "Status", "Pinned")
        self.tree = ttk.Treeview(self, columns=cols, show='headings')
        
        self.tree.heading("Content", text="Content")
        self.tree.heading("Created", text="Added Time")
        self.tree.heading("Time", text="Event Time")
        self.tree.heading("Location", text="Location")
        self.tree.heading("Status", text="Status")
        self.tree.heading("Pinned", text="📌")

        self.tree.column("Content", width=300)
        self.tree.column("Created", width=100)
        self.tree.column("Time", width=100)
        self.tree.column("Location", width=100)
        self.tree.column("Status", width=80)
        self.tree.column("Pinned", width=30, anchor="center")

        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Events
        self.tree.bind("<Button-2>", self.on_middle_click) # Middle click
        self.tree.bind("<Button-3>", self.on_right_click) # Right click context
        self.tree.bind("<Double-1>", self.on_double_click)

        self.refresh_list()
        self.refresh_stickies()

    def add_note(self):
        content = self.entry_content.get()
        if not content: return
        
        import uuid
        new_note = Note(
            id=str(uuid.uuid4()),
            content=content,
            created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        self.notes.insert(0, new_note)
        self.save_data()
        self.refresh_list()
        self.entry_content.delete(0, tk.END)

    def add_note_ai(self):
        content = self.entry_content.get()
        if not content: return
        
        def run_ai():
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = f"""Extract task info from: "{content}". 
                Return JSON with keys: content, event_time (YYYY-MM-DD HH:MM or null), location (or null), status (PENDING/URGENT)."""
                response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                data = json.loads(response.text)
                
                import uuid
                new_note = Note(
                    id=str(uuid.uuid4()),
                    content=data.get("content", content),
                    created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    event_time=data.get("event_time"),
                    location=data.get("location"),
                    status=data.get("status", "PENDING")
                )
                self.notes.insert(0, new_note)
                self.save_data()
                self.after(0, self.refresh_list)
                self.after(0, lambda: self.entry_content.delete(0, tk.END))
            except Exception as e:
                print(e)
                self.after(0, lambda: messagebox.showerror("AI Error", str(e)))

        threading.Thread(target=run_ai, daemon=True).start()

    def refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for note in self.notes:
            pin_icon = "📌" if note.is_pinned else ""
            vals = (note.content, note.created_at, note.event_time or "-", note.location or "-", note.status, pin_icon)
            self.tree.insert("", tk.END, iid=note.id, values=vals)

    def refresh_stickies(self):
        # Open windows for pinned notes, close others
        active_ids = [n.id for n in self.notes if n.is_pinned]
        
        # Create missing
        for note in self.notes:
            if note.is_pinned and note.id not in self.windows:
                self.windows[note.id] = StickyNoteWindow(self, note, self.on_note_update, self.toggle_pin_from_window)
        
        # Remove unpinned
        for nid in list(self.windows.keys()):
            if nid not in active_ids:
                self.windows[nid].destroy()
                del self.windows[nid]

    def on_note_update(self, updated_note):
        self.save_data()
        self.refresh_list() # To update content in list

    def on_middle_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if item_id:
            self.toggle_pin(item_id)

    def on_double_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if item_id:
            # Edit details dialog could go here
            pass

    def on_right_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if item_id:
            self.tree.selection_set(item_id)
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Toggle Pin (Middle Click)", command=lambda: self.toggle_pin(item_id))
            menu.add_command(label="Delete", command=lambda: self.delete_note(item_id))
            menu.add_separator()
            menu.add_command(label="Mark DONE", command=lambda: self.set_status(item_id, "DONE"))
            menu.add_command(label="Mark URGENT", command=lambda: self.set_status(item_id, "URGENT"))
            menu.post(event.x_root, event.y_root)

    def toggle_pin(self, note_id):
        for note in self.notes:
            if note.id == note_id:
                note.is_pinned = not note.is_pinned
                if note.is_pinned:
                    # Set initial position near mouse if possible, else center
                    note.x = self.winfo_pointerx() - 100
                    note.y = self.winfo_pointery() - 50
                break
        self.save_data()
        self.refresh_list()
        self.refresh_stickies()

    def toggle_pin_from_window(self, note):
        self.toggle_pin(note.id)

    def delete_note(self, note_id):
        self.notes = [n for n in self.notes if n.id != note_id]
        if note_id in self.windows:
            self.windows[note_id].destroy()
            del self.windows[note_id]
        self.save_data()
        self.refresh_list()

    def set_status(self, note_id, status):
        for note in self.notes:
            if note.id == note_id:
                note.status = status
                break
        self.save_data()
        self.refresh_list()

    def choose_bg(self):
        # In a real app this might change the Windows wallpaper or just the app theme
        # Here we just pick a default note color
        c = colorchooser.askcolor(title="Choose default note color")[1]
        if c:
            global DEFAULT_COLOR
            DEFAULT_COLOR = c

    def load_data(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r") as f:
                    data = json.load(f)
                    self.notes = [Note.from_dict(d) for d in data]
            except:
                self.notes = []

    def save_data(self):
        with open(DATA_FILE, "w") as f:
            json.dump([n.to_dict() for n in self.notes], f)

if __name__ == "__main__":
    app = NoteMinderApp()
    app.mainloop()
