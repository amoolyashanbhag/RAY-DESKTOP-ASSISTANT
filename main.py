import sys, os, json, subprocess, threading, datetime, random, ast, operator as op, importlib
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import math

# ===========================
# Library Availability Check
# ===========================
def check_and_import(module_name, pip_name=None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None

sr = check_and_import("speech_recognition")
pyttsx3 = check_and_import("pyttsx3")
cv2 = check_and_import("cv2", "opencv-python")
mp = check_and_import("mediapipe")
textblob_mod = check_and_import("textblob")

voice_available = sr is not None and pyttsx3 is not None
gesture_available = cv2 is not None and mp is not None
sentiment_available = textblob_mod is not None

# ===========================
# Safe Eval (Math)
# ===========================
operators = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.BitXor: op.xor, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg
}
ALLOWED_NAMES = {**{k: v for k, v in vars(math).items() if not k.startswith("__")}, "abs": abs, "round": round}

def safe_eval(expr):
    try:
        node = ast.parse(expr, mode="eval").body
        return _eval(node)
    except Exception:
        raise ValueError("Invalid expression")

def _eval(node):
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Num): return node.n
    if isinstance(node, ast.BinOp):
        left = _eval(node.left); right = _eval(node.right)
        return operators[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](_eval(node.operand))
    raise ValueError("Invalid expression type")

# ===========================
# Thread-safe Signals
# ===========================
class Communicate(QObject):
    append_RAY = pyqtSignal(str)
    append_user = pyqtSignal(str)

# ===========================
# Main Window
# ===========================
class RAYWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAY - Reassuring AI for You")
        self.resize(1100, 650)

        self.comm = Communicate()
        self.comm.append_RAY.connect(self._append_RAY)
        self.comm.append_user.connect(self._append_user)

        self.voice_listening = False
        self.engine = pyttsx3.init() if pyttsx3 else None

        self._build_ui()
        self._check_dependencies()
        self._greet()

    # -------------------------
    # UI Setup
    # -------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Status bar
        top = QHBoxLayout()
        self.status_label = QLabel("Checking dependencies...")
        top.addWidget(self.status_label)
        layout.addLayout(top)

        # Chat window
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        layout.addWidget(self.chat, 1)

        # Input bar
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a command or question...")
        self.send_btn = QPushButton("Send")
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        layout.addLayout(row)

        # Voice and Gesture options
        bottom = QHBoxLayout()
        self.voice_check = QCheckBox("Enable voice")
        self.gesture_btn = QPushButton("Start Gesture Mode")
        bottom.addWidget(self.voice_check)
        bottom.addWidget(self.gesture_btn)
        layout.addLayout(bottom)

        # Event bindings
        self.send_btn.clicked.connect(self._on_send)
        self.input.returnPressed.connect(self._on_send)
        self.gesture_btn.clicked.connect(self._gesture_start)

    # -------------------------
    # System Checks
    # -------------------------
    def _check_dependencies(self):
        missing = []
        if not sr: missing.append("speech_recognition")
        if not pyttsx3: missing.append("pyttsx3")
        if not cv2: missing.append("opencv-python")
        if not mp: missing.append("mediapipe")
        if not textblob_mod: missing.append("textblob")

        if missing:
            self.status_label.setText(f"⚠ Missing: {', '.join(missing)}. Please install using pip.")
            self._append_RAY(f"The following packages are missing: {', '.join(missing)}")
        else:
            self.status_label.setText("✅ All dependencies loaded successfully!")

    # -------------------------
    # Display Functions
    # -------------------------
    def _greet(self):
        self._append_RAY("Hello! I'm RAY — your smart assistant.")
        self._append_RAY("Try 'analyze sentiment I am happy' or 'start gesture mode'.")

    def _append_user(self, text):
        self.chat.append(f"<b>You:</b> {text}")

    def _append_RAY(self, text):
        self.chat.append(f"<span style='color:#00bfff'><b>RAY:</b></span> {text}")

    # -------------------------
    # Input Handling
    # -------------------------
    def _on_send(self):
        user_input = self.input.text().strip()
        if not user_input:
            return
        self.input.clear()
        self._append_user(user_input)
        resp = self.handle_command(user_input)
        self._append_RAY(resp)
        if self.voice_check.isChecked() and self.engine:
            self.engine.say(resp)
            self.engine.runAndWait()

    # -------------------------
    # Command Processing
    # -------------------------
    def handle_command(self, s: str) -> str:
        s_lower = s.lower()

        # Sentiment Analysis
        if "sentiment" in s_lower or "analyze sentiment" in s_lower:
            if not sentiment_available:
                return "TextBlob not installed. Run: pip install textblob"
            from textblob import TextBlob
            text = s.replace("analyze sentiment", "").replace("sentiment", "").strip()
            if not text:
                text = "I love coding"
            blob = TextBlob(text)
            score = blob.sentiment.polarity
            if score > 0:
                return f"Positive 😊 (score: {score:.2f})"
            elif score < 0:
                return f"Negative 😔 (score: {score:.2f})"
            else:
                return f"Neutral 😐 (score: {score:.2f})"

        # Time & Date
        if "time" in s_lower:
            return datetime.datetime.now().strftime("The time is %H:%M:%S")
        if "date" in s_lower:
            return datetime.datetime.now().strftime("Today's date: %A, %d %B %Y")

        # Math
        if any(ch.isdigit() for ch in s_lower) and any(op in s_lower for op in "+-*/"):
            try:
                result = safe_eval(s_lower)
                return f"The answer is {result}"
            except:
                return "Couldn't calculate that."

        # Greeting
        if "hello" in s_lower or "hi" in s_lower:
            return random.choice(["Hello!", "Hi there!", "Hey!"])

        # Gesture
        if "gesture" in s_lower:
            self._gesture_start()
            return "Starting gesture mode... (Press 'q' to quit)"

        return "Sorry, I didn't understand that."

    # -------------------------
    # Gesture Recognition
    # -------------------------
    def _gesture_start(self):
        if not gesture_available:
            QMessageBox.warning(self, "Error", "OpenCV or MediaPipe not installed.")
            return
        t = threading.Thread(target=self._gesture_thread, daemon=True)
        t.start()

    def _gesture_thread(self):
        import cv2, mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._append_RAY("Gesture recognition started. Show your hand!")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                    # Convert landmarks to (x, y)
                    h, w, _ = frame.shape
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

                    thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]
                    wrist = landmarks[0]
                    index_mcp = landmarks[5]
                    tip_idxs = [8, 12, 16, 20]
                    pip_idxs = [6, 10, 14, 18]

                    import time
                    now = time.time()
                    if not hasattr(self, "last_gesture_time"):
                        self.last_gesture_time = 0
                    cooldown = 2

                    def send_once(msg):
                        if now - self.last_gesture_time > cooldown:
                            self.comm.append_RAY.emit(msg)
                            self.last_gesture_time = now

                    # 👍 Thumbs Up
                    if thumb_tip[1] < index_mcp[1] and index_tip[1] > index_mcp[1]:
                        cv2.putText(frame, "👍 Thumbs Up", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        send_once("Gesture detected: 👍 Thumbs Up")

                    # 🖐️ Open Palm
                    open_palm = all(landmarks[tip_idxs[i]][1] < landmarks[pip_idxs[i]][1] for i in range(4))
                    if open_palm:
                        cv2.putText(frame, "🖐️ Open Palm", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                        send_once("Gesture detected: 🖐️ Open Palm")

                    # ✊ Fist
                    dists = [((landmarks[i][0]-wrist[0])**2 + (landmarks[i][1]-wrist[1])**2)**0.5 for i in tip_idxs]
                    if max(dists) < 80:
                        cv2.putText(frame, "✊ Fist", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        send_once("Gesture detected: ✊ Fist")

                    # ✌️ Peace
                    up = [landmarks[i][1] < landmarks[pip_idxs[i-1]][1] for i in range(1, 3)]
                    down = [landmarks[i][1] > landmarks[pip_idxs[i-1]][1] for i in range(3, 5)]
                    if all(up) and all(down):
                        cv2.putText(frame, "✌️ Peace", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                        send_once("Gesture detected: ✌️ Peace")

                    # Finger Counting
                    fingers = []
                    if landmarks[4][0] < landmarks[3][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    for i in range(1, 5):
                        fingers.append(1 if landmarks[tip_idxs[i-1]][1] < landmarks[pip_idxs[i-1]][1] else 0)
                    count = sum(fingers)
                    if count > 0:
                        cv2.putText(frame, f"✋ {count} Fingers", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        send_once(f"Gesture detected: Showing {count} fingers")

                    # 👋 Wave
                    if not hasattr(self, "wave_positions"):
                        self.wave_positions = []
                    self.wave_positions.append(wrist[0])
                    if len(self.wave_positions) > 5:
                        self.wave_positions.pop(0)
                        movement = max(self.wave_positions) - min(self.wave_positions)
                        if movement > 80:
                            cv2.putText(frame, "👋 Wave", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (128,0,255), 2)
                            send_once("Gesture detected: 👋 Wave")

            cv2.imshow("Gesture Mode - Press Q to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._append_RAY("Gesture mode ended.")

# ===========================
# App Entry
# ===========================
def main():
    app = QApplication(sys.argv)
    w = RAYWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
