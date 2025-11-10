# voice_module.py
import threading
import speech_recognition as sr

class VoiceListener:
    def __init__(self, callback):
        """
        callback: function to call when a voice command is recognized.
        """
        self.callback = callback
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
        except Exception as e:
            print("Voice init problem:", e)
            self.recognizer = None
            self.microphone = None
        self.listening = False
        self.thread = None

    def _listen_loop(self):
        if not self.recognizer or not self.microphone:
            print("VoiceListener: microphone not available.")
            return
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=6)
                    command = self.recognizer.recognize_google(audio)
                    print("Recognized:", command)
                    self.callback(command)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    print("Voice listening error:", e)
                    break

    def start_listening(self):
        if self.listening:
            return
        self.listening = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print("VoiceListener started")

    def stop_listening(self):
        self.listening = False
        print("VoiceListener stopped")
