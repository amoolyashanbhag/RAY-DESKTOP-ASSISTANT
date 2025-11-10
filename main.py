
import sys, os, threading, subprocess, json, webbrowser, time, traceback
from presentation_mode import PresentationMode
from tkinter import filedialog, messagebox
import tkinter as tk
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QMessageBox, QCheckBox, QFileDialog, QListWidget, QDialog, QFormLayout,
    QDialogButtonBox, QInputDialog, QProgressBar
)
from PyQt5.QtCore import Qt


# Optional libraries (handled gracefully)
try:
    import speech_recognition as sr
except Exception:
    sr = None
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

# App-wide state
APP_STATE = {"listening": False, "gesture_running": False, "presentation": False}

# Dark theme stylesheet
DARK_STYLE = """
QMainWindow { background-color: #0b1220; color: #dbe7ff; }
QWidget { color: #dbe7ff; }
QTextEdit, QLineEdit { background-color: #0f1724; color: #e6f0ff; border: 1px solid #26303a; padding:6px; }
QPushButton { background-color: #1f2937; color: #e6f0ff; border-radius:8px; padding:8px; }
QPushButton:hover { background-color: #374151; }
QLabel { color: #9fb3ff; }
QCheckBox { color: #9fb3ff; }
QProgressBar { background: #111827; color: #dbe7ff; border: 1px solid #26303a; height:14px; }
"""

class RAYWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAY - Reassuring AI for You")
        self.resize(980, 640)
        self.apply_dark_theme()
        self._build_ui()

        # ensure tasks file exists
        self.tasks_file = Path.home() / "RAY_tasks.json"
        if not self.tasks_file.exists():
            self.tasks_file.write_text("[]", encoding="utf-8")

        # speech
        self.listen_thread = None

        # voice engine (optional)
        self.tts_engine = None
        if pyttsx3:
            try:
                self.tts_engine = pyttsx3.init()
            except Exception:
                self.tts_engine = None

        # presentation mode instance placeholder
        presentation_mode = None
        ppt_path = None

        self._append_chat("✅ All dependencies loaded successfully!" if self._check_deps() else "⚠ Some optional dependencies are missing. See status line.")

    def apply_dark_theme(self):
        self.setStyleSheet(DARK_STYLE)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        #presentation
        self.presentation_mode = PresentationMode()

        # Top status bar
        top_bar = QHBoxLayout()
        self.status_label = QLabel("Status: Ready")
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()

        self.gesture_status = QLabel("Gesture: Idle")
        top_bar.addWidget(self.gesture_status)

        self.sentiment_status = QLabel("Sentiment: Idle")
        top_bar.addWidget(self.sentiment_status)

        self.presentation_status = QLabel("Presentation: OFF")
        top_bar.addWidget(self.presentation_status)

        self.hand_energy_label = QLabel("Hand Energy: 0%")
        top_bar.addWidget(self.hand_energy_label)

        main_layout.addLayout(top_bar)

        # Chat area
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        main_layout.addWidget(self.chat, 4)

        # Input row
        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a command or question...")
        input_row.addWidget(self.input, 1)
        self.send_btn = QPushButton("Send")
        input_row.addWidget(self.send_btn)
        main_layout.addLayout(input_row)

        # Buttons row (feature buttons)
        btn_row = QHBoxLayout()
        buttons = [
            ("Browser", self.open_browser),
            ("Music", self.open_music),
            ("Tasks", self.open_tasks),
            ("Calculator", self.open_calculator),
            ("Calendar", self.open_calendar),
            ("Files", self.open_files),
            ("Settings", self.open_settings),
        ]
        for name, handler in buttons:
            b = QPushButton(name)
            b.clicked.connect(handler)
            btn_row.addWidget(b)
        main_layout.addLayout(btn_row)

        # Voice / Gesture control row
        vg_row = QHBoxLayout()
        self.voice_checkbox = QCheckBox("Enable voice")
        vg_row.addWidget(self.voice_checkbox)

        self.start_listen_btn = QPushButton("Start Listening")
        self.stop_listen_btn = QPushButton("Stop Listening")
        self.stop_listen_btn.setEnabled(False)
        vg_row.addWidget(self.start_listen_btn)
        vg_row.addWidget(self.stop_listen_btn)

        self.gesture_btn = QPushButton("Start Gesture Mode")
        vg_row.addWidget(self.gesture_btn)

        # Sentiment / Gesture control badges
        self.start_sentiment_btn = QPushButton("Start Sentiment Analysis")
        vg_row.addWidget(self.start_sentiment_btn)
        main_layout.addLayout(vg_row)

        # Presentation button in a separate row (below voice/gesture row)
        pres_row = QHBoxLayout()
        self.start_presentation_btn = QPushButton("Start Presentation Mode")
        pres_row.addWidget(self.start_presentation_btn)
        main_layout.addLayout(pres_row)

        # Bind events
        self.send_btn.clicked.connect(self.on_send)
        self.input.returnPressed.connect(self.on_send)
        self.start_listen_btn.clicked.connect(self.start_listening)
        self.stop_listen_btn.clicked.connect(self.stop_listening)
        self.gesture_btn.clicked.connect(self.toggle_gesture)
        self.start_sentiment_btn.clicked.connect(self.analyze_input_sentiment)
        self.start_presentation_btn.clicked.connect(self.toggle_presentation_mode)

        # Small footer
        footer = QHBoxLayout()
        footer.addStretch()
        main_layout.addLayout(footer)

        # Welcome
        self._append_chat("RAY: Hello! I'm RAY — your smart assistant.")
        self._append_chat("RAY: Try 'analyze sentiment I am happy' or 'start gesture mode'.")

    def _check_deps(self):
        missing = []
        if sr is None: missing.append("speech_recognition")
        if TextBlob is None: missing.append("textblob")
        if cv2 is None or mp is None: missing.append("opencv-python/mediapipe")
        if missing:
            self.status_label.setText("⚠ Missing: " + ", ".join(missing))
            return False
        else:
            self.status_label.setText("✅ All dependencies loaded successfully!")
            return True

    def _append_chat(self, text):
        ts = time.strftime("%H:%M:%S")
        self.chat.append(f"[{ts}] {text}")

    # ---------------- Feature implementations ----------------
    def open_browser(self):
        webbrowser.open("https://www.google.com")
        self._append_chat("Opened browser (Google).")

    def open_music(self):
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
                self._append_chat("Error opening music: " + str(e))

    def open_tasks(self):
        dlg = TasksDialog(self.tasks_file, self)
        dlg.exec_()
        self._append_chat("Tasks dialog closed.")

    def open_calculator(self):
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["calc"])
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", "-a", "Calculator"])
            else:
                for cmd in (["gnome-calculator"], ["xcalc"], ["gnome-calculator"]):
                    try:
                        subprocess.Popen(cmd)
                        break
                    except Exception:
                        continue
            self._append_chat("Opened calculator.")
        except Exception as e:
            self._append_chat("Could not open calculator: " + str(e))

    def open_calendar(self):
        try:
            webbrowser.open("https://calendar.google.com/")
            self._append_chat("Opened Google Calendar in browser.")
        except Exception as e:
            self._append_chat("Could not open calendar: " + str(e))

    def open_files(self):
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
            self._append_chat("Could not open files: " + str(e))

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec_():
            self._append_chat("Settings saved.")

    # ---------------- Chat and Sentiment ----------------
    def on_send(self):
        text = self.input.text().strip()
        if not text:
            return
        self._append_chat("You: " + text)
        self.input.clear()
        # rudimentary command parsing
        lower = text.lower()
        if lower.startswith("analyze sentiment"):
            payload = text.partition("analyze sentiment")[2].strip()
            if payload:
                self._run_sentiment(payload)
            else:
                self._append_chat("Usage: analyze sentiment <text>")
        elif lower.startswith("start gesture mode"):
            self.toggle_gesture()
        else:
            # run sentiment on general text as well
            self._run_sentiment(text)

    def analyze_input_sentiment(self):
        # analyze last user input or prompt
        text, ok = QInputDialog.getText(self, "Sentiment", "Enter text to analyze:")
        if ok and text.strip():
            self._run_sentiment(text.strip())

    def _run_sentiment(self, text):
        if TextBlob is None:
            self._append_chat("Sentiment analysis unavailable (textblob not installed).")
            return
        try:
            blob = TextBlob(text)
            pol = blob.sentiment.polarity
            subj = blob.sentiment.subjectivity
            s = "Neutral"
            if pol > 0.2: s = "Positive"
            elif pol < -0.2: s = "Negative"
            self.sentiment_status.setText(f"Sentiment: {s} ({pol:.2f})")
            self._append_chat(f"Sentiment result: {s} (polarity={pol:.2f}, subjectivity={subj:.2f})")
        except Exception as e:
            self._append_chat("Sentiment analysis error: " + str(e))

    # ---------------- Voice listening ----------------
    def start_listening(self):
        if not self.voice_checkbox.isChecked():
            QMessageBox.information(self, "Voice", "Please enable 'Enable voice' checkbox first.")
            return
    def stop_listening(self):
        APP_STATE["listening"] = False
        self.start_listen_btn.setEnabled(True)
        self.stop_listen_btn.setEnabled(False)
        self._append_chat("Voice listening stopped.")

    def _listen_loop(self):
        recognizer = sr.Recognizer()
        mic = None
        try:
            mic = sr.Microphone()
        except Exception as e:
            self._append_chat("Microphone error: " + str(e))
            APP_STATE["listening"] = False
            return
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            while APP_STATE["listening"]:
                try:
                    audio = recognizer.listen(source, timeout=6, phrase_time_limit=8)
                    text = recognizer.recognize_google(audio)
                    self._append_chat("Voice: " + text)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    self._append_chat("Voice recognition error: " + str(e))

     # ---------------- Gesture detection ----------------
    def toggle_gesture(self):
        if cv2 is None or mp is None:
            QMessageBox.warning(self, "Missing", "OpenCV and MediaPipe are required for gesture detection.")
            return
        if APP_STATE["gesture_running"]:
            APP_STATE["gesture_running"] = False
            self.gesture_btn.setText("Start Gesture Mode")
            self.gesture_status.setText("Gesture: Idle")
            self._append_chat("Gesture mode stopping...")
        else:
            APP_STATE["gesture_running"] = True
            self.gesture_btn.setText("Stop Gesture Mode")
            self.gesture_status.setText("Gesture: Running")
            t = threading.Thread(target=self._gesture_loop, daemon=True)
            t.start()

    def _gesture_loop(self):
        try:
            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils
            hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self._append_chat("Gesture recognition started. Press 'q' in camera window to quit.")
            positions = []
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
                    # simple finger count using landmark y positions
                    try:
                        lm = results.multi_hand_landmarks[0].landmark
                        fingers = 0
                        # thumb approximation
                        if lm[4].x < lm[3].x:
                            fingers += 1
                        tips = [8,12,16,20]; pips=[6,10,14,18]
                        for t,p in zip(tips,pips):
                            if lm[t].y < lm[p].y:
                                fingers += 1
                        self._append_chat(f"Gesture detected: {fingers} fingers")
                        # hand energy: simple movement magnitude
                        wrist_x = int(lm[0].x * frame.shape[1])
                        positions.append(wrist_x)
                        if len(positions) > 8: positions.pop(0)
                        energy = int((max(positions) - min(positions)) / max(1, frame.shape[1]) * 100)
                        energy = min(max(0, energy), 100)
                        self.hand_energy_label.setText(f"Hand Energy: {energy}%")
                    except Exception:
                        pass
                cv2.imshow("Gesture Mode - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            APP_STATE["gesture_running"] = False
            cap.release()
            cv2.destroyAllWindows()
            self.gesture_status.setText("Gesture: Idle")
            self.gesture_btn.setText("Start Gesture Mode")
            self._append_chat("Gesture mode ended.")
        except Exception as e:
            self._append_chat("Gesture loop exception: " + str(e))
            traceback.print_exc()

    # ---------------- Presentation mode (simple toggle) ----------------
    def toggle_presentation_mode(self):
        if not hasattr(self, "presentation_active"):
            self.presentation_active = False

        if not self.presentation_active:
            try:
                # Choose PPT file (optional)
                ppt_path = QFileDialog.getOpenFileName(
                    self,
                    "Select PowerPoint File",
                    "",
                    "PowerPoint Files (*.pptx *.ppt)"
                )[0]

                # Start gesture + PowerPoint
                self.presentation_mode.start(ppt_path)
                self.presentation_active = True
                self.presentation_btn.setText("Stop Presentation Mode ▢")
                self.status_label.setText("Presentation: ON 🎤")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Unable to start Presentation Mode:\n{e}")

        else:
            try:
                self.presentation_mode.stop()
                self.presentation_active = False
                self.presentation_btn.setText("Start Presentation Mode ▶")
                self.status_label.setText("Presentation: OFF ⛔")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Unable to stop Presentation Mode:\n{e}")

class TasksDialog(QDialog):
    def __init__(self, tasks_file, parent=None):
        super().__init__(parent)
        self.tasks_file = Path(tasks_file)
        self.setWindowTitle("Tasks")
        self.resize(420, 320)
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
        self.tasks_file.write_text(json.dumps(items, indent=2), encoding='utf-8')

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

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(360,220)
        layout = QFormLayout(self)
        self.voice_default = QCheckBox("Enable voice by default")
        layout.addRow(self.voice_default)
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