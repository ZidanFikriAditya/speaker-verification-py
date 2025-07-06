from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
from tempfile import NamedTemporaryFile
import librosa
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from pydub import AudioSegment
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
import warnings
warnings.filterwarnings('ignore')

# Initialize SpeechBrain model (updated for SpeechBrain 1.0+)
try:
    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    SPEECHBRAIN_AVAILABLE = True
    print("SpeechBrain model loaded successfully (v1.0+)")
except Exception as e:
    print(f"SpeechBrain not available: {e}")
    SPEECHBRAIN_AVAILABLE = False

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
    
    # Pre-processing: normalisasi dan noise reduction
    y = librosa.util.normalize(y)
    
    # MFCC features (ciri khas suara) - lebih banyak coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    
    # Pitch features (tinggi nada) - lebih robust
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_values = f0[f0 > 0]
    if len(f0_values) > 0:
        mean_pitch = np.mean(f0_values)
        std_pitch = np.std(f0_values)
        pitch_range = np.max(f0_values) - np.min(f0_values)
    else:
        mean_pitch = 0
        std_pitch = 0
        pitch_range = 0
    
    # Spectral features (karakteristik frekuensi)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Chroma features (karakteristik harmonik)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Mel-frequency spectral coefficients
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_mean = np.mean(mel_spectrogram, axis=1)
    
    # Tempo dan rhythm
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Formant estimation (voice tract characteristics)
    stft = librosa.stft(y)
    magnitude = np.abs(stft)
    formants = []
    for frame in magnitude.T[:100]:  # Sample 100 frames
        peaks, _ = find_peaks(frame, height=np.max(frame)*0.1, distance=10)
        if len(peaks) >= 2:
            # Sort peaks by magnitude and take top 2
            peak_magnitudes = [(frame[peak], peak) for peak in peaks]
            peak_magnitudes.sort(reverse=True)
            formants.append([peak_magnitudes[0][1], peak_magnitudes[1][1]])
    
    if formants:
        formants = np.array(formants)
        f1_mean = np.mean(formants[:, 0]) if formants.shape[1] > 0 else 0
        f2_mean = np.mean(formants[:, 1]) if formants.shape[1] > 1 else 0
    else:
        f1_mean, f2_mean = 0, 0
    
    # Voice activity detection features
    energy = librosa.feature.rms(y=y)
    energy_mean = float(np.mean(energy))
    energy_std = float(np.std(energy))
    
    features = {
        'mfcc_mean': mfcc_mean,
        'mfcc_std': mfcc_std,
        'mfcc_delta': mfcc_delta_mean,
        'pitch_mean': float(mean_pitch),
        'pitch_std': float(std_pitch),
        'pitch_range': float(pitch_range),
        'spectral_centroid': float(np.mean(spectral_centroids)),
        'spectral_rolloff': float(np.mean(spectral_rolloff)),
        'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
        'spectral_contrast': np.mean(spectral_contrast, axis=1),
        'zero_crossing_rate': float(np.mean(zero_crossing_rate)),
        'chroma': np.mean(chroma, axis=1),
        'mel_features': mel_mean,
        'tempo': float(tempo),
        'formant_f1': float(f1_mean),
        'formant_f2': float(f2_mean),
        'energy_mean': energy_mean,
        'energy_std': energy_std
    }
    
    return features

def calculate_speechbrain_similarity(audio_path1, audio_path2):
    """
    Menggunakan model SpeechBrain ECAPA-VOXCELEB untuk speaker verification
    State-of-the-art model setara dengan Google Speaker ID
    """
    if not SPEECHBRAIN_AVAILABLE:
        return None, None
    
    try:
        print(f"Processing audio files: {audio_path1}, {audio_path2}")
        
        # Extract embeddings dari kedua audio
        embedding1 = speaker_model.encode_batch(audio_path1)
        embedding2 = speaker_model.encode_batch(audio_path2)
        
        print(f"Embedding shapes: {embedding1.shape}, {embedding2.shape}")
        
        # Hitung cosine similarity antara embeddings
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        similarity_score = float(similarity.item())
        
        # Hitung verification score (probability bahwa speaker sama)
        verification_score = speaker_model.verify_batch(audio_path1, audio_path2)
        verification_prob = float(verification_score.item())
        
        print(f"SpeechBrain raw scores - Similarity: {similarity_score}, Verification: {verification_prob}")
        
        return similarity_score, verification_prob
        
    except Exception as e:
        print(f"SpeechBrain detailed error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def calculate_similarity(features1, features2):
    """
    Menghitung similarity antara dua set fitur audio dengan multiple metrics yang lebih strict
    """
    similarities = {}
    weights = {}
    
    # MFCC similarity - lebih strict dengan multiple distance metrics
    mfcc_cosine_sim = 1 - cosine(features1['mfcc_mean'], features2['mfcc_mean'])
    mfcc_std_sim = 1 - cosine(features1['mfcc_std'], features2['mfcc_std'])
    mfcc_delta_sim = 1 - cosine(features1['mfcc_delta'], features2['mfcc_delta'])
    
    # Euclidean distance untuk MFCC (normalized)
    mfcc_euclidean = euclidean(features1['mfcc_mean'], features2['mfcc_mean'])
    mfcc_euclidean_sim = max(0, 1 - (mfcc_euclidean / 10))  # Normalize
    
    similarities['mfcc'] = float((mfcc_cosine_sim + mfcc_std_sim + mfcc_delta_sim + mfcc_euclidean_sim) / 4)
    weights['mfcc'] = 0.35
    
    # Pitch similarity - lebih strict
    pitch_diff = abs(float(features1['pitch_mean']) - float(features2['pitch_mean']))
    pitch_std_diff = abs(float(features1['pitch_std']) - float(features2['pitch_std']))
    pitch_range_diff = abs(float(features1['pitch_range']) - float(features2['pitch_range']))
    
    # Stricter pitch thresholds
    pitch_sim = max(0, 1 - (pitch_diff / 50))  # Reduced from 100 to 50
    pitch_std_sim = max(0, 1 - (pitch_std_diff / 25))  # Reduced from 50 to 25
    pitch_range_sim = max(0, 1 - (pitch_range_diff / 75))
    
    similarities['pitch'] = float((pitch_sim + pitch_std_sim + pitch_range_sim) / 3)
    weights['pitch'] = 0.25
    
    # Spectral similarity - lebih comprehensive
    spectral_centroid_diff = abs(float(features1['spectral_centroid']) - float(features2['spectral_centroid']))
    spectral_rolloff_diff = abs(float(features1['spectral_rolloff']) - float(features2['spectral_rolloff']))
    spectral_bandwidth_diff = abs(float(features1['spectral_bandwidth']) - float(features2['spectral_bandwidth']))
    
    spectral_contrast_sim = 1 - cosine(features1['spectral_contrast'], features2['spectral_contrast'])
    
    spectral_sim = max(0, 1 - (spectral_centroid_diff / 3000))  # More strict
    rolloff_sim = max(0, 1 - (spectral_rolloff_diff / 7000))   # More strict
    bandwidth_sim = max(0, 1 - (spectral_bandwidth_diff / 2000))
    
    similarities['spectral'] = float((spectral_sim + rolloff_sim + bandwidth_sim + spectral_contrast_sim) / 4)
    weights['spectral'] = 0.15
    
    # Mel-frequency features
    mel_sim = 1 - cosine(features1['mel_features'], features2['mel_features'])
    similarities['mel'] = float(mel_sim)
    weights['mel'] = 0.1
    
    # Formant similarity (voice tract characteristics)
    f1_diff = abs(float(features1['formant_f1']) - float(features2['formant_f1']))
    f2_diff = abs(float(features1['formant_f2']) - float(features2['formant_f2']))
    formant_sim = max(0, 1 - ((f1_diff + f2_diff) / 200))  # Strict formant comparison
    similarities['formant'] = float(formant_sim)
    weights['formant'] = 0.08
    
    # Energy characteristics
    energy_mean_diff = abs(float(features1['energy_mean']) - float(features2['energy_mean']))
    energy_std_diff = abs(float(features1['energy_std']) - float(features2['energy_std']))
    energy_sim = max(0, 1 - ((energy_mean_diff + energy_std_diff) / 0.2))
    similarities['energy'] = float(energy_sim)
    weights['energy'] = 0.05
    
    # Zero crossing rate similarity
    zcr_diff = abs(float(features1['zero_crossing_rate']) - float(features2['zero_crossing_rate']))
    zcr_sim = max(0, 1 - (zcr_diff / 0.3))  # More strict
    similarities['zcr'] = float(zcr_sim)
    weights['zcr'] = 0.02
    
    # Weighted average
    total_similarity = sum(similarities[key] * weights[key] for key in similarities)
    
    return float(total_similarity), similarities

def determine_same_speaker_advanced(similarity_score, detailed_similarities, speechbrain_similarity=None, speechbrain_verification=None):
    """
    Menentukan apakah dua audio dari speaker yang sama dengan:
    1. Model pre-trained SpeechBrain (primary)
    2. Multiple audio features (fallback/validation)
    
    Threshold disesuaikan agar tidak terlalu strict untuk speaker yang sama
    """
    
    # Primary method: SpeechBrain model (state-of-the-art)
    if speechbrain_similarity is not None and speechbrain_verification is not None:
        # SpeechBrain verification threshold (lebih fleksibel)
        if speechbrain_verification > 0.25:  # Lowered from 0.3
            # Additional validation dengan cosine similarity
            if speechbrain_similarity > 0.55:  # Lowered from 0.65
                return True, "speechbrain_high_confidence"
            elif speechbrain_similarity > 0.45:  # Lowered from 0.55
                # Cross-validate dengan traditional features (lebih lenieh)
                core_features_good = (
                    detailed_similarities['mfcc'] > 0.60 and  # Lowered from 0.65
                    detailed_similarities['pitch'] > 0.45 and  # Lowered from 0.6
                    detailed_similarities['spectral'] > 0.50  # Lowered from 0.55
                )
                if core_features_good:
                    return True, "speechbrain_medium_with_validation"
        
        # Additional fallback untuk SpeechBrain dengan threshold lebih rendah
        if speechbrain_verification > 0.15 and speechbrain_similarity > 0.4:
            # Check traditional features sebagai backup
            backup_features_acceptable = (
                detailed_similarities['mfcc'] > 0.65 or
                (detailed_similarities['pitch'] > 0.5 and detailed_similarities['spectral'] > 0.55)
            )
            if backup_features_acceptable:
                return True, "speechbrain_low_confidence_with_backup"
        
        # SpeechBrain says different speaker (lebih konservatif dalam rejection)
        if speechbrain_verification < 0.1 and speechbrain_similarity < 0.3:
            return False, "speechbrain_different_speaker"
    
    # Fallback: Traditional feature-based analysis (less strict)
    # Threshold utama - lebih fleksibel
    if similarity_score > 0.75:  # Lowered from 0.78
        return True, "traditional_high_confidence"
    
    # Multi-layer validation dengan threshold yang lebih realistis
    core_features_good = (
        detailed_similarities['mfcc'] > 0.68 and  # Lowered from 0.72
        detailed_similarities['pitch'] > 0.30 and  # Significantly lowered from 0.65 (pitch variance is high)
        detailed_similarities['spectral'] > 0.55  # Lowered from 0.60
    )
    
    # Advanced features validation (lebih fleksibel)
    advanced_features_good = (
        detailed_similarities.get('mel', 0) > 0.55 and  # Lowered from 0.60
        detailed_similarities.get('formant', 0) > 0.50 and  # Lowered from 0.55
        detailed_similarities.get('energy', 0) > 0.40  # Lowered from 0.45
    )
    
    # Decision making dengan threshold yang lebih realistis
    if core_features_good and advanced_features_good and similarity_score > 0.65:  # Lowered from 0.70
        return True, "traditional_multi_feature_validation"
    
    # Fallback dengan MFCC yang kuat
    if detailed_similarities['mfcc'] > 0.70 and similarity_score > 0.65:  # Lowered thresholds
        return True, "traditional_mfcc_strong"
    
    # Voice characteristics consistency check (lebih lenieh)
    voice_characteristics_good = (
        detailed_similarities['pitch'] > 0.25 and  # Much lower tolerance for pitch
        detailed_similarities.get('formant', 0) > 0.55 and
        detailed_similarities['mfcc'] > 0.65  # Lowered from 0.75
    )
    
    if voice_characteristics_good and similarity_score > 0.60:  # Lowered from 0.45
        return True, "traditional_voice_characteristics"
    
    # Additional lenient check untuk edge cases
    # Jika MFCC atau spectral features sangat tinggi, boleh lolos dengan similarity rendah
    exceptional_single_feature = (
        (detailed_similarities['mfcc'] > 0.70 and similarity_score > 0.60) or
        (detailed_similarities['spectral'] > 0.80 and detailed_similarities.get('mel', 0) > 0.80 and similarity_score > 0.55) or
        (detailed_similarities.get('formant', 0) > 0.85 and detailed_similarities.get('mel', 0) > 0.85 and similarity_score > 0.58)
    )
    
    if exceptional_single_feature:
        return True, "traditional_exceptional_single_feature"
    
    # Special case: High spectral + formant + mel (compensate for low pitch)
    high_non_pitch_features = (
        detailed_similarities['spectral'] > 0.85 and
        detailed_similarities.get('mel', 0) > 0.85 and
        detailed_similarities.get('formant', 0) > 0.85 and
        detailed_similarities['mfcc'] > 0.65
    )
    
    if high_non_pitch_features and similarity_score > 0.55:
        return True, "traditional_high_non_pitch_features"
    
    return False, "insufficient_similarity"

def determine_same_speaker(similarity_score, detailed_similarities):
    """
    Legacy function - kept for backward compatibility
    """
    result, _ = determine_same_speaker_advanced(similarity_score, detailed_similarities)
    return result

def calculate_confidence_advanced(similarity_score, detailed_similarities, same_speaker, decision_method, speechbrain_similarity=None, speechbrain_verification=None):
    """
    Menghitung confidence score berdasarkan metode yang digunakan dan konsistensi features
    Disesuaikan untuk threshold yang lebih fleksibel
    """
    if not same_speaker:
        # Confidence in rejection - lebih hati-hati dalam rejection
        if decision_method == "speechbrain_different_speaker":
            # Lebih konservatif dalam rejection confidence
            max_score = max(speechbrain_similarity or 0, speechbrain_verification or 0)
            return float(min(0.90, 0.70 + (1 - max_score) * 0.20))  # Reduced confidence in rejection
        else:
            return float(min(0.85, (1 - similarity_score) * 1.1))  # Less confident in traditional rejection
    
    # Confidence in acceptance
    if decision_method.startswith("speechbrain"):
        # SpeechBrain-based confidence
        if speechbrain_verification is not None and speechbrain_similarity is not None:
            if decision_method == "speechbrain_high_confidence":
                return float(min(0.95, 0.80 + speechbrain_verification * 0.15))
            elif decision_method == "speechbrain_medium_with_validation":
                # Kombinasi SpeechBrain + traditional features
                feature_scores = [
                    detailed_similarities['mfcc'],
                    detailed_similarities['pitch'],
                    detailed_similarities['spectral']
                ]
                feature_consistency = 1 - np.std(feature_scores) if len(feature_scores) > 1 else 0.75
                base_confidence = (speechbrain_verification + speechbrain_similarity) / 2
                return float(min(0.92, base_confidence * 0.6 + feature_consistency * 0.4))
            elif decision_method == "speechbrain_low_confidence_with_backup":
                # Lower confidence tapi masih reasonable
                return float(min(0.85, 0.65 + (speechbrain_verification + speechbrain_similarity) / 2 * 0.20))
        
        # Fallback jika SpeechBrain data tidak tersedia
        return float(min(0.88, similarity_score * 0.90))
    
    else:
        # Traditional feature-based confidence (disesuaikan dengan threshold baru)
        feature_scores = [
            detailed_similarities['mfcc'],
            detailed_similarities['pitch'],
            detailed_similarities['spectral'],
            detailed_similarities.get('mel', 0),
            detailed_similarities.get('formant', 0)
        ]
        
        # Remove zero scores for confidence calculation
        valid_scores = [score for score in feature_scores if score > 0]
        
        if valid_scores:
            score_std = np.std(valid_scores)
            consistency_bonus = max(0, 0.10 - score_std)  # Increased bonus for consistency
            base_confidence = np.mean(valid_scores)
            
            # Confidence disesuaikan dengan decision method
            if decision_method == "traditional_high_confidence":
                confidence = min(0.92, (base_confidence + consistency_bonus) * 0.90)
            elif decision_method == "traditional_exceptional_single_feature":
                confidence = min(0.85, base_confidence * 0.85 + consistency_bonus)
            else:
                confidence = min(0.88, (base_confidence + consistency_bonus) * 0.88)
        else:
            confidence = similarity_score * 0.80
        
        return float(confidence)

def calculate_confidence(similarity_score, detailed_similarities, same_speaker):
    """
    Legacy function - kept for backward compatibility
    """
    return calculate_confidence_advanced(similarity_score, detailed_similarities, same_speaker, "traditional_legacy")

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

        # Method 1: SpeechBrain model (state-of-the-art speaker verification)
        speechbrain_similarity = None
        speechbrain_verification = None
        speechbrain_available = SPEECHBRAIN_AVAILABLE
        
        if SPEECHBRAIN_AVAILABLE:
            try:
                speechbrain_similarity, speechbrain_verification = calculate_speechbrain_similarity(wav1, wav2)
                print(f"SpeechBrain Similarity: {speechbrain_similarity:.4f}" if speechbrain_similarity else "SpeechBrain: Failed")
                print(f"SpeechBrain Verification: {speechbrain_verification:.4f}" if speechbrain_verification else "SpeechBrain: Failed")
            except Exception as e:
                print(f"SpeechBrain error: {e}")
                speechbrain_available = False

        # Method 2: Traditional feature extraction (fallback/validation)
        features1 = extract_features(wav1)
        features2 = extract_features(wav2)

        # Hitung similarity dengan multiple metrics
        overall_similarity, detailed_similarities = calculate_similarity(features1, features2)
        
        # Tentukan apakah speaker sama dengan advanced method
        same_speaker, decision_method = determine_same_speaker_advanced(
            overall_similarity, 
            detailed_similarities, 
            speechbrain_similarity, 
            speechbrain_verification
        )

        # Hitung confidence score yang lebih akurat
        confidence = calculate_confidence_advanced(
            overall_similarity, 
            detailed_similarities, 
            same_speaker,
            decision_method,
            speechbrain_similarity, 
            speechbrain_verification
        )

        # Logging untuk debugging
        print(f"Overall Similarity: {float(overall_similarity):.4f}")
        print(f"MFCC Similarity: {float(detailed_similarities['mfcc']):.4f}")
        print(f"Pitch Similarity: {float(detailed_similarities['pitch']):.4f}")
        print(f"Spectral Similarity: {float(detailed_similarities['spectral']):.4f}")
        print(f"Mel Similarity: {float(detailed_similarities.get('mel', 0)):.4f}")
        print(f"Formant Similarity: {float(detailed_similarities.get('formant', 0)):.4f}")
        print(f"Decision Method: {decision_method}")
        print(f"Same Speaker: {same_speaker} (Confidence: {float(confidence):.2%})")

        # Bersihkan file
        os.remove(path1)
        os.remove(path2)
        os.remove(wav1)
        os.remove(wav2)

        response_data = {
            "similarity_score": float(overall_similarity),
            "confidence": float(confidence),
            "same_speaker": bool(same_speaker),
            "method_used": decision_method,
            "detailed_analysis": {
                "mfcc_similarity": float(detailed_similarities['mfcc']),
                "pitch_similarity": float(detailed_similarities['pitch']),
                "spectral_similarity": float(detailed_similarities['spectral']),
                "mel_similarity": float(detailed_similarities.get('mel', 0)),
                "formant_similarity": float(detailed_similarities.get('formant', 0)),
                "energy_similarity": float(detailed_similarities.get('energy', 0)),
                "rhythm_similarity": float(detailed_similarities['zcr'])
            },
            "advanced_analysis": {
                "speechbrain_available": speechbrain_available,
                "speechbrain_similarity": float(speechbrain_similarity) if speechbrain_similarity is not None else None,
                "speechbrain_verification": float(speechbrain_verification) if speechbrain_verification is not None else None
            },
            "threshold_info": {
                "primary_method": "speechbrain_ecapa_voxceleb" if speechbrain_available else "traditional_features",
                "speechbrain_similarity_threshold": 0.45,
                "speechbrain_verification_threshold": 0.25,
                "traditional_similarity_threshold": 0.75,
                "pitch_tolerance": "low (compensated by other features)",
                "strict_mode": False,
                "false_positive_protection": "balanced"
            }
        }

        return response_data

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "detail": "Error in speaker comparison processing"
        })
    
@app.get("/system-info")
async def get_system_info():
    """Endpoint untuk melihat konfigurasi sistem"""
    return {
        "version": "3.0",
        "primary_method": "speechbrain_ecapa_voxceleb" if SPEECHBRAIN_AVAILABLE else "traditional_features",
        "models": {
            "speechbrain": {
                "available": SPEECHBRAIN_AVAILABLE,
                "model": "speechbrain/spkrec-ecapa-voxceleb",
                "description": "State-of-the-art speaker verification model",
                "accuracy": "Comparable to Google Speaker ID"
            },
            "traditional_features": {
                "available": True,
                "description": "Fallback method using audio signal processing",
                "features_count": 8
            }
        },
        "features": {
            "mfcc_coefficients": 20,
            "pitch_algorithm": "YIN",
            "formant_analysis": True,
            "energy_analysis": True,
            "mel_features": 40,
            "speaker_embeddings": SPEECHBRAIN_AVAILABLE
        },
        "thresholds": {
            "speechbrain": {
                "similarity_threshold": 0.45,
                "verification_threshold": 0.25,
                "high_confidence_threshold": 0.55
            },
            "traditional": {
                "overall_similarity": 0.75,
                "mfcc_similarity": 0.68,
                "pitch_tolerance_hz": "flexible (compensated)",
                "spectral_tolerance_khz": 3
            }
        },
        "improvements": {
            "false_positive_protection": "balanced",
            "multi_layer_validation": True,
            "advanced_features": True,
            "conservative_mode": False,
            "state_of_the_art_model": SPEECHBRAIN_AVAILABLE,
            "dual_method_validation": True,
            "flexible_thresholds": True
        },
        "accuracy_notes": {
            "primary_method": "Uses pre-trained deep learning model trained on VoxCeleb dataset",
            "fallback_reliability": "Traditional features with balanced thresholds",
            "false_positive_rate": "Balanced to allow same speakers while preventing false positives",
            "threshold_philosophy": "Prioritizes speaker acceptance while maintaining accuracy"
        }
    }
