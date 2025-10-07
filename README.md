# sport-translations

translating live broadcasts into local languages!

# Real-Time Voice Translator

This project captures audio from your microphone, transcribes it, translates it to a target language, and speaks the translation in real-time. It uses the Google Cloud ecosystem for Speech-to-Text, Translation, and Text-to-Speech.

## Prerequisites

1.  **Python 3.7+**: Make sure Python is installed on your system.
2.  **Google Cloud Account**: You need a Google Cloud account with a project set up.
3.  **Enabled APIs**: In your Google Cloud project, enable the following APIs:
    - Cloud Speech-to-Text API
    - Cloud Translation API
    - Cloud Text-to-Speech API
4.  **Authentication**: You need to authenticate your environment so the Python client libraries can access your project. The simplest way is to use the Google Cloud CLI:
    - [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install).
    - Run `gcloud init` to initialize the SDK.
    - Run `gcloud auth application-default login` to set up your credentials.
5.  **FFmpeg**: This is a critical dependency for `pydub` to handle audio playback.
    - **macOS**: `brew install ffmpeg`
    - **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
    - **Windows**: Download from the [official site](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.

## Setup Instructions

1.  **Create a Project Directory**:

    ```bash
    mkdir real-time-translator
    cd real-time-translator
    ```

2.  **Create a Virtual Environment** (Recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Create the Files**: Create the files (`config.py`, `main.py`, `requirements.txt`) listed in this guide and copy the code into them.

4.  **Install Dependencies**: Use the `requirements.txt` file to install all the necessary Python libraries.

    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Languages**: Open the `config.py` file and set your desired `SOURCE_LANGUAGE` and `TARGET_LANGUAGE`. A list of supported language codes can be found in the Google Cloud documentation.

## How to Run

Once all the setup steps are complete, run the main script from your terminal:

```bash
python main.py
```

The application will start, and you will see a message indicating that the microphone is live. Start speaking, and after a short pause, you will hear the translated audio.

To stop the application, press `Ctrl+C` in the terminal.
