import asyncio
import queue
import threading
import time
import io
import json

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import openai
from vosk import Model, KaldiRecognizer

# =============== CONFIGURATION ===============
# Your OpenAI API key (inserted as provided)
openai.api_key = "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"
MODEL_PATH = "vosk-model"  # This folder must contain your Vosk model files

# Configure WebRTC (using a public STUN server)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# =============== GLOBAL VARIABLES ===============
# Queue to store recognized text from audio frames
recognized_queue = queue.Queue()

# Initialize the Vosk model (download a small model and place it in your repo or host externally)
model = Model(MODEL_PATH)

# Audio frame queue for raw audio bytes
audio_frames = queue.Queue()

# =============== VOSK RECOGNITION WORKER ===============
def vosk_recognition_worker():
    recognizer = KaldiRecognizer(model, 16000)  # 16kHz sample rate
    recognizer.SetWords(True)
    buffer_data = b""
    last_partial = ""
    while True:
        frame = audio_frames.get()
        if frame is None:  # Stop signal
            break
        buffer_data += frame
        # Process roughly every second of audio (adjust as needed)
        if len(buffer_data) > 40000:
            if recognizer.AcceptWaveform(buffer_data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    recognized_queue.put(text)
            else:
                partial_res = json.loads(recognizer.PartialResult())
                partial_text = partial_res.get("partial", "").strip()
                if partial_text and partial_text != last_partial:
                    recognized_queue.put(partial_text + " ... (partial)")
                    last_partial = partial_text
            buffer_data = b""

# =============== STREAMLIT AUDIO CALLBACK ===============
def transcribe_audio_callback(frame):
    # Convert frame to bytes (assuming frame is float32; adjust if necessary)
    audio_data = frame.to_ndarray()
    audio_bytes = (audio_data * 32767).astype("int16").tobytes()
    audio_frames.put(audio_bytes)
    return frame

# =============== STREAMLIT APP INTERFACE ===============
st.title("Real-Time Jarvis Assistant Demo")
st.markdown("""
**Instructions:**
1. Allow microphone access when prompted.
2. Speak naturally; partial transcripts will display.
3. Final segments are sent to GPT for a response.
""")

# Start the WebRTC component to capture audio
webrtc_ctx = webrtc_streamer(
    key="jarvis-real-time",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=transcribe_audio_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# =============== BACKGROUND THREADS FOR PROCESSING ===============
# Start Vosk recognition in a separate thread
recognition_thread = threading.Thread(target=vosk_recognition_worker, daemon=True)
recognition_thread.start()

# GPT processing loop (async)
async def gpt_loop():
    conversation_history = []
    while True:
        await asyncio.sleep(1.0)  # Check the recognized_queue every second
        if not recognized_queue.empty():
            user_text = recognized_queue.get()
            st.write(f"**User said:** {user_text}")
            # If it's a partial result, skip sending to GPT
            if user_text.endswith("(partial)"):
                continue
            conversation_history.append({"role": "user", "content": user_text})
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # or use "gpt-4" if available
                    messages=conversation_history
                )
                assistant_text = response["choices"][0]["message"]["content"]
                st.write(f"**GPT:** {assistant_text}")
                conversation_history.append({"role": "assistant", "content": assistant_text})
            except Exception as e:
                st.error(f"OpenAI API error: {e}")

async def main_async():
    task = asyncio.create_task(gpt_loop())
    while True:
        await asyncio.sleep(0.1)
    task.cancel()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(main_async())
except asyncio.CancelledError:
    pass
finally:
    loop.close()
