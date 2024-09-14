import pyaudio
import numpy as np
import whisper
import threading
import torch
import keyboard
import time
import ollama
import sounddevice as sd
import logging
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Whisper model
model = whisper.load_model("small.en")

# PyAudio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Initialize PyAudio
p = pyaudio.PyAudio()

# Global variables
audio_buffer = np.array([], dtype=np.int16)
recording = False

import torch
import numpy as np
from TTS.api import TTS
import sounddevice as sd
import soundfile as sf

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/en/jenny/jenny").to(device) #tts_models/it/mai_female/vits
#tts = TTS("tts_models/it/mai_female/vits").to(device)
import time
import re
import threading

# Lock to ensure sentences are processed one at a time
tts_lock = threading.Lock()

def text_to_speech_with_lock(text):
    # Ignore very short text (e.g., punctuation or small fragments)
    if len(text.strip()) < 5:
        return

    with tts_lock:  # Ensure that only one sentence is spoken at a time
        try:
            text_to_speech(text)
        except Exception as e:
            logging.error(f"Error in text-to-speech: {e}")
            print("Error occurred while generating or playing speech.")

chat_history = []

def api_generate(text: str, max_retries=3, retry_delay=2):
    def process_tts_in_thread(sentence):
        tts_thread = threading.Thread(target=text_to_speech_with_lock, args=(sentence,))
        tts_thread.start()
        tts_thread.join()  # Wait for the current thread to finish before starting the next one

    for attempt in range(max_retries):
        try:
            # Add the new user message to chat history
            chat_history.append({'role': 'user', 'content': text})

            stream = ollama.chat(
                model='celine',
                messages=chat_history,  # Pass the chat history
                stream=True,
            )

            print('-----------------------------------------')
            buffer_sentence = ""
            sentence_endings = re.compile(r'([.!?])')  # Regex to detect sentence endings

            for chunk in stream:
                if not chunk['done']:
                    buffer_sentence += chunk['message']['content']
                    # Check for complete sentences using regex
                    if sentence_endings.search(buffer_sentence):
                        sentences = re.split(r'([.!?])', buffer_sentence)
                        for i in range(0, len(sentences) - 1, 2):
                            complete_sentence = sentences[i] + sentences[i+1]  # Combine sentence with its punctuation
                            process_tts_in_thread(complete_sentence.strip())
                        buffer_sentence = sentences[-1]  # Save the remaining part (if any)

            # Handle any leftover sentence at the end
            if buffer_sentence.strip():
                process_tts_in_thread(buffer_sentence.strip())

            print("\n-----------------------------------------")
            # Add the response to chat history
            chat_history.append({'role': 'assistant', 'content': buffer_sentence.strip()})
            return
        except Exception as e:
            logging.error(f"Error during streaming (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Unable to get response from Ollama.")
                process_tts_in_thread("I'm sorry, but I'm having trouble processing your request at the moment. Please try again later.")


def ollama_infer(result):
    try:
        # Optionally clear chat history when a new session starts
        # global chat_history
        # chat_history = []

        full_prompt = f"{result}\n"
        response = api_generate(full_prompt)
        if response:
            print(f"Ollama response: {response}")
            text_to_speech(response)
        else:
            logging.warning("No response received from Ollama.")
    except Exception as e:
        logging.error(f"Error in Ollama response: {e}")
        text_to_speech("I'm sorry, but I encountered an error while processing your request.")


def text_to_speech(text):
    try:
        wav = tts.tts(text=text)

        # Normalize the waveform (optional step to ensure proper sound levels)
        wav = np.array(wav, dtype=np.float32)

        # Play the audio
        sd.play(wav, samplerate=48000, blocking=True)
        sd.wait()

        logging.info("Audio played successfully")
    except Exception as e:
        logging.error(f"Error in text-to-speech: {e}")
        print("Error occurred while generating or playing speech.")

        
def ollama_infer_threaded(transcription):
    ollama_thread = threading.Thread(target=ollama_infer, args=(transcription,))
    ollama_thread.start()

def capture_audio(input_device_index):
    global audio_buffer, recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)

    print("Press and hold spacebar to start recording, release to stop and transcribe.")
    
    try:
        while True:
            if keyboard.is_pressed("space"):
                if not recording:
                    print("Recording started...")
                    recording = True
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                audio_buffer = np.concatenate((audio_buffer, audio_np))
            else:
                if recording:
                    print("Recording stopped. Transcribing...")
                    recording = False
                    process_buffer(audio_buffer)
                    audio_buffer = np.array([], dtype=np.int16)
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()

def process_buffer(audio_buffer):
    audio_float = audio_buffer.astype(np.float32) / 32767.0
    audio_tensor = torch.from_numpy(audio_float).float()

    try:
        result = model.transcribe(audio_tensor)
        transcription = result["text"].strip()

        if transcription:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] Transcription: {transcription}")
            ollama_infer_threaded(transcription)

    except Exception as e:                   
        logging.error(f"Error during transcription: {e}")

def list_microphones():
    print("Available microphones:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:
            print(f"{i}: {dev.get('name')}")

if __name__ == "__main__":
    try:
        list_microphones()
        capture_audio(3)
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        p.terminate()