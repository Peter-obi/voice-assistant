This is an english version of linyiLYi's implementaion of voice assistant using Apple's mlx. I tried to keep it as close as possible to the original version 

# Voice Assistant

A simple Python script that allows for voice interaction with a local large language model. In this project, the whisper implementation comes from mlx [official example library](https://github.com/ml-explore/mlx-examples/tree/main/whisper). The large language model is [Lingyi Wanwu](https://www.lingyiwanwu.com)'s Yi model, among which Yi-34B-Chat has stronger capabilities and is recommended for use if memory space allows.

### macOS Installation Guide

Below is the installation process for macOS. Windows and Linux can use speech_recognition and pyttsx3 to replace the macOS-specific hear/whisper and say commands in the text below.

#### Setting Up the Environment

```
conda create -n VoiceAI python=3.11
conda activate VoiceAI
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Install audio processing tools
brew install portaudio
pip install pyaudio
```

#### Installing the hear Voice Recognition Module

Download the installation package from the open source project [hear](https://github.com/sveinbjornt/hear) at [this link](https://sveinbjorn.org/files/software/hear.zip). After unzipping the folder, run `sudo bash install.sh` (administrator rights required). Once installed, the macOS voice recognition function can be called directly through console commands. Note that the keyboard dictation option in the computer settings must be enabled: Settings -> Keyboard -> Dictation (turn on the switch). The first time you use it on macOS, you also need to allow the hear module to run in "Settings -> Privacy & Security".

#### Model Files
The model files are stored in the `models/` folder and specified in the script via the variable `MODEL_PATH`.
It is recommended to download TheBloke and XeIaso's gguf format models, among which the 6B model occupies less memory:
- [TheBloke/Yi-34B-Chat-GGUF](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q8_0.gguf)
- [XeIaso/Yi-6B-Chat-GGUF](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/blob/main/yi-chat-6b.Q8_0.gguf)
