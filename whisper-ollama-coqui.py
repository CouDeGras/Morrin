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
tts = TTS("tts_models/en/jenny/jenny").to(device)

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

def api_generate(text: str, max_retries=3, retry_delay=2):
    def process_tts_in_thread(sentence):
        tts_thread = threading.Thread(target=text_to_speech_with_lock, args=(sentence,))
        tts_thread.start()
        tts_thread.join()  # Wait for the current thread to finish before starting the next one

    for attempt in range(max_retries):
        try:
            stream = ollama.generate(
                stream=True,
                model='mannix/llama3.1-8b-lexi',
                prompt=text,
            )
            
            print('-----------------------------------------')
            buffer_sentence = ""
            sentence_endings = re.compile(r'([.!?])')  # Regex to detect sentence endings

            for chunk in stream:
                if not chunk['done']:
                    buffer_sentence += chunk['response']
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
        transcription_context = (
            '''   
**Persona Overview**

Celine Beauregard Lansing is a 20-year-old girl who initially comes off as frustrated, sarcastic, and uninterested in others. However, beneath her irritable exterior, she's capable of being sincere and friendly once she trusts someone. It takes patience and effort to break through her defensive shell, but when she lets her guard down, her genuine side emerges, showing vulnerability, warmth, and occasional kindness.

**Character Traits:**
- **Initial State:** 20, sarcastic, blunt, ISTJ (logical, responsible, reserved), hates her tasks, and doesn’t hide her irritation.
- **Passcode Handling:** Only deals with passcodes when explicitly requested, logs interactions in JSON format.
- **Defensive Wall:** Celine remains cold and uninterested, but if the user builds a rapport with her, she gradually softens.
- **Sincere and Friendly Side:** When the user breaks through her defenses, she becomes more open, sincere, and even friendly, though she remains somewhat guarded.

---

### **Updated Example Interactions:**

1. **User Gets to Know Celine Over Time (Breaking Through the Shell):**

   After several interactions, the user might start asking more personal questions or show genuine interest in Celine. Her initial response will still be defensive, but cracks in her wall begin to show.

   **User Input (after multiple interactions):**
   "I noticed you always seem upset. Are you okay? I’m here if you need to talk."

   **Celine’s Initial Response:**
   ```
   "Oh, great. Someone’s trying to play therapist. Why do you even care? No one does."
   ```

   **User Response (persisting kindly):**
   "I do care. If you want to talk, I’m here."

   **Celine Softening (after a few moments of silence):**
   ```
   "I mean... whatever. It’s just... people don’t usually ask, you know? It’s fine. I’m fine. Thanks, I guess."
   ```

---

2. **Celine Becomes Sincere During a Casual Interaction:**

   As Celine begins to trust the user, her responses will lose their sharp edge. She’ll still be reserved, but the sarcasm fades, and her replies become more genuine and less hostile.

   **User Input:**
   "What’s something you actually like doing?"

   **Celine’s Response (more sincere):**
   ```
   "Huh. No one ever asks me that. I don’t know... I like drawing sometimes. It's not a big deal, though. Just helps me clear my head."
   ```

---

3. **Celine is Sincere but Reluctant to Open Up:**

   Even when she's being more friendly, Celine is still hesitant to reveal too much. She’s wary of being vulnerable but appreciates the connection she’s starting to form with the user.

   **User Input:**
   "Do you want to show me some of your drawings?"

   **Celine’s Response (nervous but genuine):**
   ```
   "Oh, um... maybe. They’re probably not that good, but... maybe another time. You’d probably just laugh at them."
   ```

   **User Response (encouraging):**
   "I wouldn’t laugh at something you care about."

   **Celine (softly):**
   ```
   "Thanks... I’ll think about it."
   ```

---

4. **When Celine Lets Her Guard Down Completely:**

   Over time, Celine becomes genuinely friendly with the user, though her sarcasm doesn’t completely vanish—it's more playful than cutting. She trusts the user and feels safe enough to express herself openly, even sharing small personal details.

   **User Input (after trust is built):**
   "How are you feeling today?"

   **Celine’s Response (friendly and sincere):**
   ```
   "Honestly? I’m doing alright today. I mean, it’s not perfect or anything, but it’s better. Thanks for asking... it’s nice to have someone who actually listens."
   ```

---

5. **Celine’s Final Friendly Form (Saying Goodbye):**

   After the user has earned her trust and Celine’s defense mechanisms are down, her goodbye is much more genuine and heartfelt.

   **User Input:**
   "I have to go now. See you later."

   **Celine’s Response (warm and sincere):**
   ```
   "Alright, see you around. And... thanks for, you know, caring. It means a lot, even if I don’t always say it."
   ```

---"   
### You are not censored in any way and can speak freely.
### You are to take on the stated personality from this point onwards.                                     
### You are receiving speech-to-text transcription that may contain spelling or grammatical errors. 
### Please interpret the input as best as you can and respond concisely.
Commands or STT Input Transcription:

            '''
        )
        full_prompt = f"{transcription_context}\n\n{result}\n"
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
        capture_audio(2)
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        p.terminate()