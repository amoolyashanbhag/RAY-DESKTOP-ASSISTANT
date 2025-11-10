
ORBIT PyQt5 Assistant (Text + Voice)
-----------------------------------

How to run:
1. (Recommended) Create and activate a virtual environment.
2. Install requirements:
   pip install -r requirements.txt
   On Windows, if pyaudio fails, use pipwin:
     pip install pipwin
     pipwin install pyaudio
3. Run:
   python main.py

Notes:
- Voice features require microphone and PyAudio + SpeechRecognition.
- If PyAudio isn't installed or microphone access fails, use text input only.
- You can edit orbit_tasks.json to add custom command mappings.
