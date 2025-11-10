
import sys, os, threading, subprocess, json, webbrowser, time
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QMessageBox, QCheckBox, QFileDialog, QListWidget, QDialog, QFormLayout,
    QDialogButtonBox, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# Optional imports
try:
    import speech_recognition as sr_mod
except Exception:
    sr_mod = None
try:
    import pyttsx3
except Exception:
    pyttsx3 = None
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None
try:
    import cv2, mediapipe as mp
except Exception:
    cv2 = None; mp = None

APP_STATE = {"listening": False, "gesture_running": False}

DARK_STYLE = """
QMainWindow { background-color: #0f1724; color: #dbe7ff; }
QTextEdit, QLineEdit { background-color: #111827; color: #dbe7ff; border: 1px solid #374151; padding:6px; }
QPushButton { background-color: #1f2937; color: #dbe7ff; border-radius:8px; padding:8px; }
QPushButton:hover { background-color: #374151; }
QLabel { color: #9fb3ff; }
QCheckBox { color: #9fb3ff; }
QListWidget { background-color: #0b1220; color: #dbe7ff; border: 1px solid #26303a; }
"""

class RAYWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAY - Reassuring AI for You (Dark)")
        self.resize(900, 600)
        self._build_ui()
        self.apply_dark_theme()

        # speech recognizer
        self.recognizer = sr_mod.Recognizer() if sr_mod else None
        self.mic = None
        self.listen_thread = None

    def apply_dark_theme(self):
        self.setStyleSheet(DARK_STYLE)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top status row
        top_row = QHBoxLayout()
        self.status_label = QLabel("RAY: Ready")
        top_row.addWidget(self.status_label)
        top_row.addStretch()
        self.gesture_label = QLabel("Gesture: Idle")
        top_row.addWidget(self.gesture_label)
        self.sentiment_label = QLabel("Sentiment: Idle")
        top_row.addWidget(self.sentiment_label)
        layout.addLayout(top_row)

        # Chat area
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        layout.addWidget(self.chat, 3)

        # Input row
        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a command or question...")
        input_row.addWidget(self.input, 1)
        self.send_btn = QPushButton("Send")
        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

        # Controls row with feature buttons
        controls = QHBoxLayout()
        for name, handler in [
            ("Music", self.open_music),
            ("Tasks", self.open_tasks),
            ("Calculator", self.open_calculator),
            ("Calendar", self.open_calendar),
            ("Files", self.open_files),
            ("Settings", self.open_settings),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(handler)
            controls.addWidget(btn)
        layout.addLayout(controls)

        # Voice and gesture controls
        vg_row = QHBoxLayout()
        self.voice_check = QCheckBox("Enable voice (requires microphone & speech_recognition)")
        vg_row.addWidget(self.voice_check)
        self.listen_btn = QPushButton("Start Listening")
        self.stop_listen_btn = QPushButton("Stop Listening")
        self.stop_listen_btn.setEnabled(False)
        vg_row.addWidget(self.listen_btn)
        vg_row.addWidget(self.stop_listen_btn)

        self.gesture_btn = QPushButton("Start Gesture Detection")
        vg_row.addWidget(self.gesture_btn)

        layout.addLayout(vg_row)

        # Event binds
        self.send_btn.clicked.connect(self.on_send)
        self.input.returnPressed.connect(self.on_send)
        self.listen_btn.clicked.connect(self.start_listening)
        self.stop_listen_btn.clicked.connect(self.stop_listening)
        self.gesture_btn.clicked.connect(self.toggle_gesture)

        # Tasks storage
        self.tasks_file = Path.home() / ".ray_tasks.json"
        if not self.tasks_file.exists():
            self.tasks_file.write_text("[]")

        # Welcome text
        self._append_chat("RAY: Good morning! I'm RAY. You can type or use voice.")

    def _append_chat(self, text):
        ts = time.strftime("%H:%M:%S")
        self.chat.append(f"[{ts}] {text}")

    # ---------------- Functional buttons ----------------
    def open_music(self):
        # let user choose an audio file and open it with system default player
        fname, _ = QFileDialog.getOpenFileName(self, "Select music file", str(Path.home()), "Audio Files (*.mp3 *.wav *.ogg);;All Files (*)")
        if fname:
            try:
                if sys.platform.startswith("win"):
                    os.startfile(fname)
                elif sys.platform.startswith("darwin"):
                    subprocess.Popen(["open", fname])
                else:
                    subprocess.Popen(["xdg-open", fname])
                self._append_chat(f"Playing music: {Path(fname).name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open file: {e}")

    def open_tasks(self):
        dlg = TasksDialog(self.tasks_file, self)
        dlg.exec_()

    def open_calculator(self):
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["calc"])
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", "-a", "Calculator"])
            else:
                # try gnome-calculator then xcalc
                for cmd in (["gnome-calculator"], ["xcalc"]):
                    try:
                        subprocess.Popen(cmd)
                        break
                    except Exception:
                        continue
            self._append_chat("Opened system calculator.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open calculator: {e}")

    def open_calendar(self):
        # Open web Google Calendar as fallback
        try:
            webbrowser.open("https://calendar.google.com/")
            self._append_chat("Opened calendar in browser.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open calendar: {e}")

    def open_files(self):
        # open user's home directory
        try:
            home = str(Path.home())
            if sys.platform.startswith("win"):
                os.startfile(home)
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", home])
            else:
                subprocess.Popen(["xdg-open", home])
            self._append_chat("Opened Files (Home directory).")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open files: {e}")

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec_():
            self._append_chat("Settings saved.")

    # ---------------- Chat / Sentiment ----------------
    def on_send(self):
        text = self.input.text().strip()
        if not text:
            return
        self._append_chat("You: " + text)
        self.input.clear()
        # perform sentiment analysis if available
        if TextBlob:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                sent = "Neutral"
                if polarity > 0.2: sent = "Positive"
                elif polarity < -0.2: sent = "Negative"
                self.sentiment_label.setText(f"Sentiment: {sent} ({polarity:.2f})")
                self._append_chat(f"Sentiment analysis: {sent} ({polarity:.2f})")
            except Exception as e:
                self._append_chat("Sentiment analysis failed: " + str(e))
        else:
            self._append_chat("Sentiment: (textblob not installed)")

    # ---------------- Voice listening ----------------
    def start_listening(self):
        if not self.voice_check.isChecked():
            QMessageBox.information(self, "Voice", "Please enable the voice checkbox first.")
            return
        if not sr_mod:
            QMessageBox.warning(self, "Missing", "speech_recognition is not installed.")
            return
        if APP_STATE["listening"]:
            return
        APP_STATE["listening"] = True
        self.listen_btn.setEnabled(False)
        self.stop_listen_btn.setEnabled(True)
        self._append_chat("Voice listening started...")
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()

    def stop_listening(self):
        APP_STATE["listening"] = False
        self.listen_btn.setEnabled(True)
        self.stop_listen_btn.setEnabled(False)
        self._append_chat("Voice listening stopped.")

    def _listen_loop(self):
        recognizer = sr_mod.Recognizer()
        mic = None
        try:
            mic = sr_mod.Microphone()
        except Exception as e:
            self._append_chat("Microphone error: " + str(e))
            APP_STATE["listening"]=False
            return
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            while APP_STATE["listening"]:
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                    text = recognizer.recognize_google(audio)
                    self._append_chat("Voice: " + text)
                except sr_mod.WaitTimeoutError:
                    continue
                except Exception as e:
                    self._append_chat("Voice error: " + str(e))

    # ---------------- Gesture detection ----------------
    def toggle_gesture(self):
        if not (cv2 and mp):
            QMessageBox.warning(self, "Missing", "OpenCV and MediaPipe are required for gesture detection.")
            return
        if APP_STATE["gesture_running"]:
            APP_STATE["gesture_running"] = False
            self.gesture_btn.setText("Start Gesture Detection")
            self.gesture_label.setText("Gesture: Idle")
            self._append_chat("Gesture detection stopped.")
        else:
            APP_STATE["gesture_running"] = True
            self.gesture_btn.setText("Stop Gesture Detection")
            self.gesture_label.setText("Gesture: Running")
            t = threading.Thread(target=self._gesture_loop, daemon=True)
            t.start()

    def _gesture_loop(self):
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._append_chat("Gesture detection started. Press 'q' in the window to exit.")
        while APP_STATE["gesture_running"]:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                # simple gesture: count fingers
                try:
                    h, w, _ = frame.shape
                    lm = results.multi_hand_landmarks[0].landmark
                    fingers = 0
                    # thumb (compare x positions)
                    if lm[4].x < lm[3].x:
                        fingers += 1
                    # other four fingers: tip y < pip y
                    tips = [8,12,16,20]
                    pips = [6,10,14,18]
                    for t,p in zip(tips,pips):
                        if lm[t].y < lm[p].y:
                            fingers +=1
                    self._append_chat(f"Gesture detected: {fingers} fingers")
                except Exception:
                    pass
            cv2.imshow("Gesture Mode - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        APP_STATE["gesture_running"] = False
        cap.release()
        cv2.destroyAllWindows()
        self.gesture_label.setText("Gesture: Idle")
        self.gesture_btn.setText("Start Gesture Detection")
        self._append_chat("Gesture mode ended.")

# ---------------- Presentation mode ----------------
self.presentation_btn = QPushButton("Start Presentation Mode 🎤")
self.presentation_btn.setStyleSheet(
    "background-color: #1E90FF; color: white; font-weight: bold; border-radius: 10px; padding: 6px;"
)
self.presentation_btn.clicked.connect(self.toggle_presentation_mode)
self.feature_layout.addWidget(self.presentation_btn)

# ---------------- Tasks dialog ----------------
class TasksDialog(QDialog):
    def __init__(self, tasks_file, parent=None):
        super().__init__(parent)
        self.tasks_file = Path(tasks_file)
        self.setWindowTitle("Tasks")
        self.resize(400,300)
        layout = QVBoxLayout(self)
        self.listw = QListWidget()
        layout.addWidget(self.listw)
        btn_row = QHBoxLayout()
        add = QPushButton("Add Task"); rem = QPushButton("Remove Task"); done = QPushButton("Mark Done")
        btn_row.addWidget(add); btn_row.addWidget(rem); btn_row.addWidget(done)
        layout.addLayout(btn_row)
        add.clicked.connect(self.add_task)
        rem.clicked.connect(self.remove_task)
        done.clicked.connect(self.mark_done)
        self.load_tasks()

    def load_tasks(self):
        try:
            data = json.loads(self.tasks_file.read_text(encoding='utf-8'))
        except Exception:
            data = []
        self.listw.clear()
        for t in data:
            self.listw.addItem(t)

    def save_tasks(self, items):
        self.tasks_file.write_text(json.dumps(items, indent=2))

    def add_task(self):
        text, ok = QInputDialog.getText(self, "New Task", "Task description:")
        if ok and text.strip():
            self.listw.addItem(text.strip())
            self._save_from_list()

    def remove_task(self):
        row = self.listw.currentRow()
        if row >= 0:
            self.listw.takeItem(row)
            self._save_from_list()

    def mark_done(self):
        row = self.listw.currentRow()
        if row >=0:
            item = self.listw.item(row)
            item.setText(item.text() + " ✅")
            self._save_from_list()

    def _save_from_list(self):
        items = [self.listw.item(i).text() for i in range(self.listw.count())]
        self.save_tasks(items)

# ---------------- Settings dialog ----------------
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(350,200)
        layout = QFormLayout(self)
        self.voice_checkbox = QCheckBox("Enable voice by default")
        layout.addRow(self.voice_checkbox)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

def main():
    app = QApplication(sys.argv)
    w = RAYWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
