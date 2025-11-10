
import threading, time, os, math
try:
    import cv2
    import mediapipe as mp
    import pyautogui
except Exception as e:
    cv2 = None; mp = None; pyautogui = None

class PresentationMode:
    """
    PresentationMode handles:
    - opening a selected .pptx file with the system default app (os.startfile)
    - launching camera window for gesture detection (OpenCV + MediaPipe)
    - mapping gestures to slide actions:
        * open palm (5 fingers) -> next slide
        * closed fist (0 fingers) -> previous slide
        * index-only (1 finger) -> start slideshow (F5)
        * steady palm for 3s -> pause gesture detection (toggle)
    """
    def __init__(self, camera_index=0, pinch_threshold=0.05):
        self.camera_index = camera_index
        self.pinch_threshold = pinch_threshold
        self._running = False
        self._slideshow_started = False
        self._pause_detection = False
        self._thread = None
        # debounce times
        self._last_action = 0
        self._action_cooldown = 0.6
        # mediapipe setup
        if mp:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
        else:
            self.mp_hands = None; self.mp_draw = None

    def start(self, ppt_path=None):
        if not (cv2 and mp and pyautogui):
            raise RuntimeError("Missing dependencies: require opencv-python, mediapipe and pyautogui")
        # open pptx file if provided (using system default)
        if ppt_path:
            if not os.path.exists(ppt_path):
                raise FileNotFoundError(f"PPT file not found: {ppt_path}")
            try:
                os.startfile(ppt_path)
            except Exception:
                # fallback: try subprocess open
                import subprocess
                subprocess.Popen(["cmd", "/c", "start", "", ppt_path])
            # small delay to let app open
            time.sleep(2.5)
        # start camera & processing thread
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, leave_slideshow=False):
        # stop detection loop
        self._running = False
        # if slideshow started and not asked to leave it, send ESC
        if self._slideshow_started and not leave_slideshow:
            try:
                pyautogui.press('esc')
            except Exception:
                pass
        self._slideshow_started = False

    def _loop(self):
        hands_mod = mp.solutions.hands
        hands = hands_mod.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2)
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        steady_timer = None
        last_palm_pos = None
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = None
            # determine gesture
            if results.multi_hand_landmarks:
                # analyze first hand primarily; if two hands present, may infer different actions
                num_hands = len(results.multi_hand_landmarks)
                # choose landmarks for first hand
                lm = results.multi_hand_landmarks[0].landmark
                # compute extended fingers using tip vs pip y coordinates
                tips = [4,8,12,16,20]
                pips = [3,6,10,14,18]
                extended = []
                for t,p in zip(tips,pips):
                    try:
                        extended.append(1 if lm[t].y < lm[p].y else 0)
                    except Exception:
                        extended.append(0)
                fingers_up = sum(extended[1:]) + (1 if extended[0] else 0)
                # open palm: >=4 fingers (including thumb heuristic)
                if fingers_up >= 4:
                    gesture = 'palm_open'
                # fist: 0 fingers
                elif fingers_up == 0:
                    gesture = 'fist'
                # index only: index extended and others not
                elif extended[1] == 1 and sum(extended[0:1] + extended[2:]) == 0:
                    gesture = 'index'
                else:
                    gesture = 'other'

                # steady palm detection: check wrist x movement low over time
                wrist = lm[0]
                wrist_x = wrist.x * w
                if last_palm_pos is None:
                    last_palm_pos = wrist_x
                    steady_timer = time.time()
                else:
                    if abs(wrist_x - last_palm_pos) < 8:  # small movement threshold in pixels
                        # if sustained for 3 seconds -> pause detection toggle
                        if time.time() - steady_timer > 3.0:
                            # toggle pause
                            self._pause_detection = not self._pause_detection
                            # reset timer so it doesn't toggle repeatedly
                            steady_timer = time.time() + 1.0
                            # provide visual feedback by printing
                            print("Gesture detection paused" if self._pause_detection else "Gesture detection resumed")
                    else:
                        last_palm_pos = wrist_x
                        steady_timer = time.time()

            else:
                gesture = None
                last_palm_pos = None
                steady_timer = None

            # handle gestures if not paused
            if not self._pause_detection and gesture:
                now = time.time()
                if now - self._last_action > self._action_cooldown:
                    if gesture == 'palm_open':
                        # next slide
                        try:
                            pyautogui.press('right')
                            print("➡ Next slide (palm_open)")
                            self._last_action = now
                        except Exception:
                            pass
                    elif gesture == 'fist':
                        try:
                            pyautogui.press('left')
                            print("⬅ Previous slide (fist)")
                            self._last_action = now
                        except Exception:
                            pass
                    elif gesture == 'index':
                        # start slideshow (F5)
                        try:
                            pyautogui.press('f5')
                            self._slideshow_started = True
                            print("▶ Slideshow started (index)")
                            self._last_action = now
                        except Exception:
                            pass
            # draw landmarks & show camera window
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Presentation Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC pressed in camera window - stop mode
                self.stop()
                break
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Open file dialog to choose the PPT file
    ppt_path = filedialog.askopenfilename(
        title="Select PowerPoint File",
        filetypes=[("PowerPoint Files", "*.pptx *.ppt"), ("All Files", "*.*")]
    )

    if not ppt_path:
        messagebox.showinfo("No file selected", "You did not select any PowerPoint file.")
        exit(0)

    try:
        mode = PresentationMode()
        mode.start(ppt_path)
        print("📽 Presentation mode started successfully.")
        print("✋ Gestures: Open palm = next | Fist = previous | Index finger = start slideshow")
        print("Press 'ESC' in the camera window to stop.")

        while mode._running:
            time.sleep(0.5)

    except Exception as e:
        messagebox.showerror("Error", f"Unable to start presentation mode:\n{e}")


