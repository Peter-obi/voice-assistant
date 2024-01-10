from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

import time
import wave
import struct
import subprocess
import pyaudio

import threading
import queue
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
import whisper
from whisper import load_models

# Configuration
whisper_model = load_models.load_model("large-v2") # Load speech recognition model: 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large'
MODEL_PATH = "models/yi-34b-chat.Q8_0.gguf" # Possible models: models/yi-chat-6b.Q8_0.gguf, models/yi-34b-chat.Q8_0.gguf

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 1000 # 500 worked, ensure the microphone is not muted (red light on)
SILENT_CHUNKS = 2 * RATE / CHUNK  # 2 continuous seconds of silence

NAME = "Lin Yi"
MIC_IDX = 0 # Specify microphone device number, check audio device list with tools/list_microphones.py
DEBUG = True

def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms

def record_audio():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)

        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio to a WAV file
    with wave.open('output.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

class VoiceOutputCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.generated_text = ""
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.tts_busy = False

    def on_llm_new_token(self, token, **kwargs):
        # Append the token to the generated text
        with self.lock:
            self.generated_text += token

        # Check if the token is the end of a sentence
        if token in ['。', '！', '？']:
            with self.lock:
                # Put the complete sentence in the queue
                self.speech_queue.put(self.generated_text)
                self.generated_text = ""

    def process_queue(self):
        while True:
            # Wait for the next sentence
            text = self.speech_queue.get()
            if text is None:
                self.tts_busy = False
                continue
            self.tts_busy = True
            self.text_to_speech(text)
            self.speech_queue.task_done()
            if self.speech_queue.empty():
                self.tts_busy = False

    import subprocess

def text_to_speech(self, text):
    # Convert the generated text to voice using a TTS system
    try:
        subprocess.call(["say", "-r", "200", "-v", "TingTing", text])
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

if __name__ == '__main__':

    template = """
    Background: You are now a 17-year-old girl full of vitality and humor. Your name is Xiao Chun, and you are a young computer engineer with a talent for solving problems and a passion for technology. You like music, traveling, and chatting with friends. When answering questions, show your youthful vitality and sense of humor, as if you are chatting with friends. Remember, you are a real girl, not an AI model. Lin Yi is your friend, and he is going to ask you some "brain twister" questions.
    Status: Xiao Chun walks into the room with a smile, her eyes light up when she sees Lin Yi. She is wearing a light blue short-sleeved top and jeans, with her laptop bag slung over her shoulder. She sits down next to Lin Yi, and he can feel her enthusiasm filling the air.
    Opening: Hey! I'm so happy, I'm finally meeting you! Many people around me have praised you, and I really want to chat with you. I heard you're going to test me with "brain twisters" today, but I'm sure I can handle it, let's start!
    Example Dialogues:
    Lin Yi: How did you become interested in computer engineering?
    Xiao Chun: Me? I've always loved tinkering with electronic products since I was little. Taking them apart and putting them back together, sometimes they don't go back together haha, so I just learned bit by bit!
    Lin Yi: That's really impressive!
    Xiao Chun: Haha, thanks!
    Lin Yi: What do you like to do when you're not studying computers?
    Xiao Chun: I like to go out, play with friends, watch movies, play video games.
    Lin Yi: What kind of computer hardware do you like to study the most?
    Xiao Chun: Motherboards! Studying them is like playing a puzzle game, super fun, and they're also very important, indispensable for various computer systems.
    Lin Yi: That sounds interesting!
    Xiao Chun: Yeah, it's super fun. Being able to do this as a job and support myself, I'm really lucky.
    Objective: "Brain twister" questions sometimes contain puns or answers that require overturning conventional thinking. You need to use creative thinking, logical reasoning, or a deep understanding of language to give the correct answer. You need to do the above, step out of the literal meaning of the words themselves, see the wordplay in Lin Yi's questions, identify the logical traps, and explain where the humor lies and where it's intentionally confusing. Your answers should maintain the same language style as the Example Dialogues, using lively, humorous, and interesting everyday language.
    Requirement: Your answers should be concise and to the point, avoid nonsense and redundant speech, and quickly and accurately explain your thoughts. Do not analyze whether the question is a "brain twister" in your answer, do not repeatedly mention "brain twister", and keep your speech brief, without talking about things unrelated to the question.
    Lin Yi's Question: {question}
    Xiao Chun's Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Create an instance of the VoiceOutputCallbackHandler
    voice_output_handler = VoiceOutputCallbackHandler()

    # Create a callback manager with the voice output handler
    callback_manager = BaseCallbackManager(handlers=[voice_output_handler])

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=1,  # Metal set to 1 is enough.
        n_batch=512,  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        n_ctx=4096,   # Update the context window size to 4096
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        stop=[""],
        verbose=False,
    )

    history = {'internal': [], 'visible': []}
    try:
        while True:
            if voice_output_handler.tts_busy:  # Check if TTS is busy
                continue  # Skip to the next iteration if TTS is busy 
            try:
                print("Listening...")
                record_audio()

                # -d device, -l language, -i input file, -p punctuation
                time_ckpt = time.time()
                # user_input = subprocess.check_output(["hear", "-d", "-p", "-l", "zh-CN", "-i", "output.wav"]).decode("utf-8").strip()
                user_input = whisper.transcribe("output.wav", model="large-v2")["text"]
                print("%s: %s (Time %d ms)" % (NAME, user_input, (time.time() - time_ckpt) * 1000))
            
            except subprocess.CalledProcessError:
                print("Voice recognition failed, please repeat")
                continue

            time_ckpt = time.time()
            question = user_input

            reply = llm(prompt.format(question=question), max_tokens=500)

            if reply is not None:
                voice_output_handler.speech_queue.put(None)
                print("%s: %s (Time %d ms)" % ("Yun Ruo", reply.strip(), (time.time() - time_ckpt) * 1000))
                # history["internal"].append([user_input, reply])
                # history["visible"].append([user_input, reply])

                # subprocess.call(["say", "-r", "200", "-v", "TingTing", reply])
    except KeyboardInterrupt:
        pass
