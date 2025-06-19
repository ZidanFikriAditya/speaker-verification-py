from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
from tempfile import NamedTemporaryFile
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from pydub import AudioSegment
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()

# Middleware untuk validasi API Key
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def save_temp_file(upload_file: UploadFile):
    temp = NamedTemporaryFile(delete=False, suffix=".webm")
    temp.write(upload_file.file.read())
    temp.close()
    return temp.name

def extract_audio(input_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(22050)
    output_path = input_path.replace(".webm", ".wav")
    audio.export(output_path, format="wav")
    return output_path

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    mean_pitch = np.mean(pitch[pitch > 0])
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean, mean_pitch

@app.post("/compare-voices")
async def compare_voices(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    _: None = Depends(verify_api_key)  # Validasi API 
):
    try:
        # Simpan dan ekstrak audio
        path1 = save_temp_file(file1)
        path2 = save_temp_file(file2)
        wav1 = extract_audio(path1)
        wav2 = extract_audio(path2)

        # Ekstraksi fitur
        mfcc1, pitch1 = extract_features(wav1)
        mfcc2, pitch2 = extract_features(wav2)

        # Bandingkan
        mfcc_distance = cosine(mfcc1, mfcc2)
        pitch_diff = abs(pitch1 - pitch2)

        # Logika sederhana
        same_speaker = mfcc_distance < 0.03

        print(f"MFCC Distance: {mfcc_distance}, Pitch Difference: {pitch_diff}, Same Speaker: {same_speaker}")

        # Bersihkan file
        os.remove(path1)
        os.remove(path2)
        os.remove(wav1)
        os.remove(wav2)

        return {
            "similarity_score": float(1 - mfcc_distance),
            "pitch_difference": float(pitch_diff),
            "same_speaker": bool(same_speaker)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
