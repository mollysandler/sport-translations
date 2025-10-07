# Sport Translations

A real-time voice translation tool for live commentary, interviews, and broadcasts.

## Overview

This project captures audio from your microphone, transcribes it in real-time, translates it to a target language of your choice, and speaks the translation back to you. It is designed for interactive use and is built on the Google Cloud ecosystem for high-quality Speech-to-Text, Translation, and Text-to-Speech.

## Prerequisites

1.  **Python 3.7+**: Make sure Python is installed on your system.
2.  **Google Cloud Account**: You need a Google Cloud account with an active project.
3.  **Enabled APIs**: In your Google Cloud project, ensure the following APIs are enabled:
    - Cloud Speech-to-Text API
    - Cloud Translation API
    - Cloud Text-to-Speech API
4.  **Google Cloud Authentication**: The application uses a service account key for authentication.
    - You must [create a service account key](https://cloud.google.com/iam/docs/keys-create-delete) (a JSON file) in your Google Cloud project.
    - Download the JSON file and store it in a secure location on your computer. You will point to this file in the setup steps.
5.  **FFmpeg**: A required dependency for audio playback.
    - **macOS**: `brew install ffmpeg`
    - **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
    - **Windows**: Download from the [official site](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.

## Setup Instructions

1.  **Clone or Download the Project**:
    Get the project files (`main.py`, `config.py`, `requirements.txt`) onto your local machine.

2.  **Create a Virtual Environment** (Recommended):
    Navigate to the project directory in your terminal and run:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Configure Credentials (.env file)**:
    Create a new file in the project directory named `.env`. Inside this file, add the following line, replacing the path with the **full, absolute path** to your downloaded service account JSON file:

    ```env
    # .env
    GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-cloud-key.json"
    ```

4.  **Install Dependencies**:
    Install all the necessary Python libraries using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Default Languages (config.py)**:
    Open the `config.py` file.
    - Set your primary `SOURCE_LANGUAGE` (the language you will be speaking).
    - Set a default `TARGET_LANGUAGE`. This will be offered as a convenient default each time you start a translation session.

## Usage

This is an interactive command-line application.

1.  **Launch the Application**:
    Run the main script from your terminal:

    ```bash
    python main.py
    ```

2.  **Control the Translator**:
    You will see a `>` prompt. Use the following commands to control the application:

    - `start`

      - Starts a new translation session.
      - You will be prompted to enter a target language code (e.g., `fr-FR` for French, `ja-JP` for Japanese).
      - **Press Enter without typing anything** to use the default language you specified in `config.py`.
      - Once started, you can speak into your microphone to begin translation.

    - `stop`

      - Gracefully stops the current translation session and returns you to the `>` prompt.

    - `exit`
      - Stops any active session and closes the application.
