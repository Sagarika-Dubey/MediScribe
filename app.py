from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import uvicorn
import os
import tempfile
import numpy as np
import wave
import struct
from collections import deque
import webrtcvad
from pydub import AudioSegment, effects
import requests
import aiohttp
import base64
import time
from dotenv import load_dotenv
load_dotenv()


class SpeakerDiarizer:
    def __init__(self):
        print("Initializing Speaker Diarization with Deepgram")
        
        try:
            self.api_key = os.environ.get("DEEPGRAM_API_KEY", "")
            if not self.api_key:
                print("Warning: DEEPGRAM_API_KEY not found in environment variables")
            
            self.known_speakers = {
                "doctor": None,
                "patient": None
            }
            
            self.is_calibrated = False
            self.calibration_samples = {"doctor": [], "patient": []}
            
            print("Speaker diarization initialized successfully")
        except Exception as e:
            print(f"Error initializing diarization: {e}")
            print("Falling back to simple energy-based speaker separation")
            self.api_key = None
    
    async def process_audio(self, audio_path: str) -> Dict[str, List[Tuple[float, float]]]:
        try:
            if not self.api_key:
                return self._simple_diarization(audio_path)
            
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            url = "https://api.deepgram.com/v1/listen?diarize=true&punctuate=true&model=nova"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav"
                }
                
                async with session.post(url, headers=headers, data=audio_data) as response:
                    if response.status != 200:
                        print(f"Deepgram API error: {response.status}")
                        return self._simple_diarization(audio_path)
                    
                    result = await response.json()
            
            speaker_segments = {}
            
            if "results" in result and "channels" in result["results"]:
                for channel in result["results"]["channels"]:
                    for alternative in channel.get("alternatives", []):
                        for word in alternative.get("words", []):
                            if "speaker" in word:
                                speaker_id = f"speaker_{word['speaker']}"
                                start_time = word["start"]
                                end_time = word["end"]
                                
                                if speaker_id not in speaker_segments:
                                    speaker_segments[speaker_id] = []
                                
                                if (speaker_segments[speaker_id] and 
                                    speaker_segments[speaker_id][-1][1] == start_time):
                                    speaker_segments[speaker_id][-1] = (
                                        speaker_segments[speaker_id][-1][0], end_time
                                    )
                                else:
                                    speaker_segments[speaker_id].append((start_time, end_time))
            
            if speaker_segments:
                speaker_segments = self._map_to_known_speakers(speaker_segments, audio_path)
            else:
                speaker_segments = self._simple_diarization(audio_path)
            
            return speaker_segments
        
        except Exception as e:
            print(f"Error in diarization: {e}")
            return self._simple_diarization(audio_path)
    
    def _simple_diarization(self, audio_path: str) -> Dict[str, List[Tuple[float, float]]]:
        try:
            audio = AudioSegment.from_wav(audio_path)
            
            chunk_size = 500
            chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
            
            energies = [chunk.dBFS for chunk in chunks]
            
            threshold = np.mean(energies)
            
            speaker_segments = {"SPEAKER_1": [], "SPEAKER_2": []}
            current_speaker = "SPEAKER_1" if energies[0] > threshold else "SPEAKER_2"
            start_time = 0
            
            for i, energy in enumerate(energies[1:], 1):
                speaker = "SPEAKER_1" if energy > threshold else "SPEAKER_2"
                
                if speaker != current_speaker:
                    end_time = i * chunk_size / 1000
                    speaker_segments[current_speaker].append((start_time, end_time))
                    start_time = end_time
                    current_speaker = speaker
            
            end_time = len(chunks) * chunk_size / 1000
            speaker_segments[current_speaker].append((start_time, end_time))
            
            if np.mean([e for i, e in enumerate(energies) if i % 2 == 0]) > np.mean([e for i, e in enumerate(energies) if i % 2 == 1]):
                return {
                    "doctor": speaker_segments["SPEAKER_1"],
                    "patient": speaker_segments["SPEAKER_2"]
                }
            else:
                return {
                    "doctor": speaker_segments["SPEAKER_2"],
                    "patient": speaker_segments["SPEAKER_1"]
                }
            
        except Exception as e:
            print(f"Error in simple diarization: {e}")
            return {"doctor": [], "patient": []}
    
    def _map_to_known_speakers(self, speaker_segments, audio_path):
        durations = {}
        for speaker, segments in speaker_segments.items():
            durations[speaker] = sum(end - start for start, end in segments)
        
        sorted_speakers = sorted(durations.items(), key=lambda x: x[1])
        
        if len(sorted_speakers) >= 2:
            doctor_id = sorted_speakers[0][0]
            patient_id = sorted_speakers[1][0]
            
            mapped_segments = {
                "doctor": speaker_segments[doctor_id],
                "patient": speaker_segments[patient_id]
            }
            
            for i in range(2, len(sorted_speakers)):
                mapped_segments["patient"].extend(speaker_segments[sorted_speakers[i][0]])
            
            return mapped_segments
        elif len(sorted_speakers) == 1:
            only_speaker = sorted_speakers[0][0]
            return {
                "doctor": speaker_segments[only_speaker],
                "patient": []
            }
        else:
            return {"doctor": [], "patient": []}
    
    def calibrate(self, speaker_type: str, audio_path: str) -> bool:
        try:
            if speaker_type not in ["doctor", "patient"]:
                return False
            
            self.calibration_samples[speaker_type].append(audio_path)
            
            if (len(self.calibration_samples["doctor"]) > 0 and 
                len(self.calibration_samples["patient"]) > 0):
                
                self.known_speakers["doctor"] = "doctor_model"
                self.known_speakers["patient"] = "patient_model"
                self.is_calibrated = True
                
                print("Speaker models calibrated")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error in calibration: {e}")
            return False
    
    def get_speaker_at_time(self, timestamp: float, diarization_result: Dict[str, List[Tuple[float, float]]]) -> str:
        for speaker, segments in diarization_result.items():
            for start, end in segments:
                if start <= timestamp <= end:
                    return speaker
        
        return ""


class MedicalConversationAnalyzer:
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        self.symptoms_patterns = [
            r"(having|suffering from|experiencing|feel|feeling|with|has|got|getting)\s+(a|an)?\s*([a-zA-Z\s]+)(pain|ache|fever|cough|cold|nausea|fatigue|weakness|infection|issue|problem)",
            r"(headache|migraine|fever|cough|cold|sore throat|nausea|vomiting|diarrhea|constipation|back pain|chest pain|shortness of breath|difficulty breathing)",
            r"(been|having|had)\s+(a|an)?\s*([a-zA-Z\s]+)(for|since)\s+(\d+)\s+(day|days|week|weeks|month|months)",
            r"(pain|ache|discomfort)\s+in\s+(my|the)\s+([a-zA-Z\s]+)",
            r"I\s+have\s+(a|an)?\s+([a-zA-Z\s]+)(pain|ache|fever|cough|cold|nausea|problem|issue)"
        ]
        
        self.medication_patterns = [
            r"(take|prescribe|recommend|give|suggest|try)\s+([A-Za-z0-9\s\-]+)\s+(once|twice|thrice|\d+\s+times)\s+(?:a|per|every|each)?\s*(?:day|daily|week|month)?",
            r"([A-Za-z0-9\s\-]+)\s+(\d+\s*mg|\d+\s*ml|\d+\s*tab|\d+\s*pill|\d+\s*capsule)",
            r"([A-Za-z0-9\s\-]+)\s+(\d+\s+times)(?:\s+a\s+day)?(?:\s+for\s+(\d+)\s+days)?",
            r"would\s+(recommend|suggest|prescribe|advise)\s+([A-Za-z0-9\s\-]+)"
        ]
        
        self.duration_patterns = [
            r"for\s+(\d+)\s+(day|days|week|weeks|month|months)",
            r"(before|after)\s+(meals?|food|eating)",
            r"(with|without)\s+(water|food|meals?)",
            r"(in the|at)\s+(morning|afternoon|evening|night|bedtime)"
        ]
        
        self.diagnosis_patterns = [
            r"(you have|it's|it is|looks like|seems to be|might be|could be|diagnosis is|diagnosing)\s+(a|an)?\s*([A-Za-z\s\-]+)",
            r"(suggesting|pointing to|indicative of|consistent with)\s+(a|an)?\s*([A-Za-z\s\-]+)",
            r"(test results|tests|x-ray|scan|mri|ct scan)\s+(show|indicate|reveal|confirm)\s+(a|an)?\s*([A-Za-z\s\-]+)"
        ]
    
    def analyze(self, text: str, speaker_segments: Dict[str, List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        normalized_text = ' '.join(text.lower().split())
        
        if speaker_segments is None:
            problems = self._extract_problems(normalized_text)
            medications = self._extract_medications(normalized_text)
            durations = self._extract_durations(normalized_text)
            diagnoses = self._extract_diagnoses(normalized_text)
        else:
            problems = self._extract_problems_by_speaker(speaker_segments, "patient")
            medications = self._extract_medications_by_speaker(speaker_segments, "doctor")
            durations = self._extract_durations_by_speaker(speaker_segments, "doctor")
            diagnoses = self._extract_diagnoses_by_speaker(speaker_segments, "doctor")
        
        soap = {
            "subjective": self._format_subjective(problems),
            "objective": "",
            "assessment": self._format_assessment(diagnoses, problems),
            "plan": self._format_plan(medications, durations)
        }
        
        return soap
    
    def _extract_problems_by_speaker(self, speaker_segments: Dict[str, List[Dict[str, Any]]], target_speaker: str) -> List[str]:
        problems = []
        
        speaker_text = ""
        for segment in speaker_segments.get(target_speaker, []):
            speaker_text += " " + segment["text"]
        
        for pattern in self.symptoms_patterns:
            matches = re.finditer(pattern, speaker_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 4 and match.group(3) and match.group(4):
                    symptom = (match.group(3) + match.group(4)).strip()
                    problems.append(symptom)
                elif len(match.groups()) >= 1:
                    symptom = match.group(0).strip()
                    problems.append(symptom)
        
        return list(set(self._normalize_entities(problems)))
    
    def _extract_medications_by_speaker(self, speaker_segments: Dict[str, List[Dict[str, Any]]], target_speaker: str) -> List[Dict[str, str]]:
        medications = []
        
        speaker_text = ""
        for segment in speaker_segments.get(target_speaker, []):
            speaker_text += " " + segment["text"]
        
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, speaker_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    med_info = {}
                    
                    if "prescribe" in match.group(1).lower() or "take" in match.group(1).lower():
                        med_info["name"] = match.group(2).strip()
                    else:
                        med_info["name"] = match.group(1).strip()
                    
                    if len(match.groups()) >= 3 and match.group(3):
                        if "mg" in match.group(2) or "ml" in match.group(2) or "tab" in match.group(2):
                            med_info["dosage"] = match.group(2).strip()
                            med_info["frequency"] = match.group(3).strip() if match.group(3) else ""
                        else:
                            med_info["frequency"] = match.group(2).strip() if match.group(2) else ""
                    
                    medications.append(med_info)
        
        return medications
    
    def _extract_durations_by_speaker(self, speaker_segments: Dict[str, List[Dict[str, Any]]], target_speaker: str) -> List[str]:
        durations = []
        
        speaker_text = ""
        for segment in speaker_segments.get(target_speaker, []):
            speaker_text += " " + segment["text"]
        
        for pattern in self.duration_patterns:
            matches = re.finditer(pattern, speaker_text, re.IGNORECASE)
            for match in matches:
                instruction = match.group(0).strip()
                durations.append(instruction)
        
        return list(set(durations))
    
    def _extract_diagnoses_by_speaker(self, speaker_segments: Dict[str, List[Dict[str, Any]]], target_speaker: str) -> List[str]:
        diagnoses = []
        
        speaker_text = ""
        for segment in speaker_segments.get(target_speaker, []):
            speaker_text += " " + segment["text"]
        
        for pattern in self.diagnosis_patterns:
            matches = re.finditer(pattern, speaker_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 3 and match.group(3):
                    diagnosis = match.group(3).strip()
                    diagnoses.append(diagnosis)
                elif len(match.groups()) >= 4 and match.group(4):
                    diagnosis = match.group(4).strip()
                    diagnoses.append(diagnosis)
        
        return list(set(self._normalize_entities(diagnoses)))
    
    def _extract_problems(self, text: str) -> List[str]:
        problems = []
        
        for pattern in self.symptoms_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 4 and match.group(3) and match.group(4):
                    symptom = (match.group(3) + match.group(4)).strip()
                    problems.append(symptom)
                elif len(match.groups()) >= 1:
                    symptom = match.group(0).strip()
                    problems.append(symptom)
        
        return list(set(self._normalize_entities(problems)))
    
    def _extract_medications(self, text: str) -> List[Dict[str, str]]:
        medications = []
        
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    med_info = {}
                    
                    if "prescribe" in match.group(1).lower() or "take" in match.group(1).lower():
                        med_info["name"] = match.group(2).strip()
                    else:
                        med_info["name"] = match.group(1).strip()
                    
                    if len(match.groups()) >= 3 and match.group(3):
                        if "mg" in match.group(2) or "ml" in match.group(2) or "tab" in match.group(2):
                            med_info["dosage"] = match.group(2).strip()
                            med_info["frequency"] = match.group(3).strip() if match.group(3) else ""
                        else:
                            med_info["frequency"] = match.group(2).strip() if match.group(2) else ""
                    
                    medications.append(med_info)
        
        return medications
    
    def _extract_durations(self, text: str) -> List[str]:
        durations = []
        
        for pattern in self.duration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                instruction = match.group(0).strip()
                durations.append(instruction)
        
        return list(set(durations))
    
    def _extract_diagnoses(self, text: str) -> List[str]:
        diagnoses = []
        
        for pattern in self.diagnosis_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 3 and match.group(3):
                    diagnosis = match.group(3).strip()
                    diagnoses.append(diagnosis)
                elif len(match.groups()) >= 4 and match.group(4):
                    diagnosis = match.group(4).strip()
                    diagnoses.append(diagnosis)
        
        return list(set(self._normalize_entities(diagnoses)))
    
    def _normalize_entities(self, entities: List[str]) -> List[str]:
        normalized = []
        for entity in entities:
            entity = re.sub(r'\b(a|an|the|some|this|that)\b', '', entity)
            entity = re.sub(r'\s+', ' ', entity).strip()
            if entity:
                normalized.append(entity)
        
        return normalized
    
    def _format_subjective(self, problems: List[str]) -> str:
        if not problems:
            return ""
        
        return "Patient reports: " + ", ".join(problems)
    
    def _format_assessment(self, diagnoses: List[str], problems: List[str]) -> str:
        if diagnoses:
            return "Assessment: " + ", ".join(diagnoses)
        elif problems:
            return "Potential diagnosis based on symptoms: " + ", ".join(problems)
        else:
            return ""
    
    def _format_plan(self, medications: List[Dict[str, str]], durations: List[str]) -> str:
        if not medications and not durations:
            return ""
        
        plan_text = ""
        
        if medications:
            plan_text += "Medications:\n"
            for med in medications:
                med_line = f"- {med['name']}"
                if 'dosage' in med and med['dosage']:
                    med_line += f" ({med['dosage']})"
                if 'frequency' in med and med['frequency']:
                    med_line += f" {med['frequency']}"
                plan_text += med_line + "\n"
        
        if durations:
            plan_text += "\nInstructions:\n"
            for instruction in durations:
                plan_text += f"- {instruction}\n"
        
        return plan_text


class AudioProcessor:
    def __init__(self):
        self.api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            print("Warning: DEEPGRAM_API_KEY not found in environment variables")
        
        print(f"Initializing Deepgram Speech Processing")
        
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)
        
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.buffer_duration_secs = 2.0
        
        self.audio_buffer = deque()
        self.accumulated_bytes = 0
        self.buffer_threshold = int(self.sample_rate * 2 * self.buffer_duration_secs)
        
        self.temp_dir = tempfile.mkdtemp()
        print(f"Audio buffer threshold set to {self.buffer_threshold} bytes ({self.buffer_duration_secs}s)")
        
        self.diarizer = SpeakerDiarizer()
    
    def add_audio_chunk(self, audio_data: bytes) -> bool:
        self.audio_buffer.append(audio_data)
        self.accumulated_bytes += len(audio_data)
        
        return self.accumulated_bytes >= self.buffer_threshold
    
    def has_voice_activity(self, audio_data: bytes) -> bool:
        samples_per_frame = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        try:
            if len(audio_data) >= samples_per_frame * 2:
                for i in range(0, len(audio_data) - samples_per_frame * 2, samples_per_frame * 2):
                    frame = audio_data[i:i + samples_per_frame * 2]
                    if self.vad.is_speech(frame, self.sample_rate):
                        return True
        except Exception as e:
            print(f"VAD error: {e}")
        
        return False
    
    def normalize_audio(self, wav_path: str) -> None:
        try:
            audio = AudioSegment.from_wav(wav_path)
            
            normalized = effects.normalize(audio)
            
            if audio.dBFS < -25:
                normalized = normalized + 8
            
            normalized.export(wav_path, format="wav")
            
            print(f"Audio normalized from {audio.dBFS:.2f}dB to {normalized.dBFS:.2f}dB")
        
        except Exception as e:
            print(f"Error normalizing audio: {e}")
    
    async def process_buffer(self) -> Dict[str, Any]:
        if not self.audio_buffer:
            return {"text": "", "speaker": "", "speaker_segments": []}
        
        try:
            temp_file = os.path.join(self.temp_dir, f"audio_chunk_{int(time.time())}.wav")
            
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_buffer))
            
            self.normalize_audio(temp_file)
            
            if self.api_key:
                url = "https://api.deepgram.com/v1/listen?diarize=true&punctuate=true&model=nova"
                
                with open(temp_file, 'rb') as f:
                    audio_data = f.read()
                
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "audio/wav"
                    }
                    
                    async with session.post(url, headers=headers, data=audio_data) as response:
                        if response.status != 200:
                            print(f"Deepgram API error: {response.status}")
                            return {"text": "", "error": f"API error: {response.status}", "speaker_segments": []}
                        
                        result = await response.json()
                
                transcript = ""
                speaker_segments = []
                
                if "results" in result and "channels" in result["results"]:
                    for channel in result["results"]["channels"]:
                        for alternative in channel.get("alternatives", []):
                            transcript = alternative.get("transcript", "")
                            
                            if "words" in alternative:
                                current_speaker = None
                                current_segment = {"speaker": "", "text": "", "start": 0, "end": 0}
                                
                                for word in alternative["words"]:
                                    speaker_id = word.get("speaker", "unknown")
                                    
                                    if speaker_id == 0:
                                        speaker_role = "doctor"
                                    else:
                                        speaker_role = "patient"
                                    
                                    if current_speaker is None:
                                        current_speaker = speaker_role
                                        current_segment = {
                                            "speaker": speaker_role,
                                            "text": word["word"],
                                            "start": word["start"],
                                            "end": word["end"]
                                        }
                                    elif speaker_role == current_speaker:
                                        current_segment["text"] += " " + word["word"]
                                        current_segment["end"] = word["end"]
                                    else:
                                        speaker_segments.append(current_segment)
                                        current_speaker = speaker_role
                                        current_segment = {
                                            "speaker": speaker_role,
                                            "text": word["word"],
                                            "start": word["start"],
                                            "end": word["end"]
                                        }
                                
                                if current_segment["text"]:
                                    speaker_segments.append(current_segment)
                
                diarization_result = await self.diarizer.process_audio(temp_file)
                
                for segment in speaker_segments:
                    mid_time = (segment["start"] + segment["end"]) / 2
                    diarized_speaker = self.diarizer.get_speaker_at_time(mid_time, diarization_result)
                    if diarized_speaker:
                        segment["speaker"] = diarized_speaker
                
                result = {
                    "text": transcript,
                    "speaker_segments": speaker_segments
                }
                
                os.remove(temp_file)
                
                self.audio_buffer.clear()
                self.accumulated_bytes = 0
                
                return result
            
            else:
                print("No Deepgram API key available")
                return {"text": "", "error": "No API key", "speaker_segments": []}
        
        except Exception as e:
            print(f"Error processing audio buffer: {e}")
            return {"text": "", "error": str(e), "speaker_segments": []}


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections: List[WebSocket] = []

audio_processor = AudioProcessor()
conversation_analyzer = MedicalConversationAnalyzer()

class TranscriptionRequest(BaseModel):
    audio_base64: str

class SOAPRequest(BaseModel):
    conversation_text: str
    speaker_segments: Optional[List[Dict[str, Any]]] = None


@app.get("/")
async def read_root():
    return {"status": "Medical Transcription Service is running"}


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            if audio_processor.has_voice_activity(data):
                should_process = audio_processor.add_audio_chunk(data)
                
                if should_process:
                    result = await audio_processor.process_buffer()
                    
                    await websocket.send_json(result)
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.post("/api/transcribe")
async def transcribe_audio(request: TranscriptionRequest):
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
        
        audio_processor.normalize_audio(temp_path)
        
        result = {"text": "", "speaker_segments": []}
        
        if audio_processor.api_key:
            url = "https://api.deepgram.com/v1/listen?diarize=true&punctuate=true&model=nova"
            
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Token {audio_processor.api_key}",
                    "Content-Type": "audio/wav"
                }
                
                async with session.post(url, headers=headers, data=audio_data) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail=f"Deepgram API error: {response.status}")
                    
                    deepgram_result = await response.json()
            
            transcript = ""
            speaker_segments = []
            
            if "results" in deepgram_result and "channels" in deepgram_result["results"]:
                    for channel in deepgram_result["results"]["channels"]:
                        for alternative in channel.get("alternatives", []):
                            transcript = alternative.get("transcript", "")
                            
                            # Extract speaker segments
                            if "words" in alternative:
                                current_speaker = None
                                current_segment = {"speaker": "", "text": "", "start": 0, "end": 0}
                                
                                for word in alternative["words"]:
                                    speaker_id = word.get("speaker", "unknown")
                                    
                                    # Map speaker ID to role
                                    if speaker_id == 0:  # Assuming first speaker is doctor
                                        speaker_role = "doctor"
                                    else:
                                        speaker_role = "patient"
                                    
                                    if current_speaker is None:
                                        current_speaker = speaker_role
                                        current_segment = {
                                            "speaker": speaker_role,
                                            "text": word["word"],
                                            "start": word["start"],
                                            "end": word["end"]
                                        }
                                    elif speaker_role == current_speaker:
                                        current_segment["text"] += " " + word["word"]
                                        current_segment["end"] = word["end"]
                                    else:
                                        speaker_segments.append(current_segment)
                                        current_speaker = speaker_role
                                        current_segment = {
                                            "speaker": speaker_role,
                                            "text": word["word"],
                                            "start": word["start"],
                                            "end": word["end"]
                                        }
                                
                                # Add the last segment
                                if current_segment["text"]:
                                    speaker_segments.append(current_segment)
                
            # Process with diarization for improved speaker identification
            diarization_result = await audio_processor.diarizer.process_audio(temp_path)
                
            # Map speakers based on diarization
            for segment in speaker_segments:
                mid_time = (segment["start"] + segment["end"]) / 2
                diarized_speaker = audio_processor.diarizer.get_speaker_at_time(mid_time, diarization_result)
                if diarized_speaker:
                    segment["speaker"] = diarized_speaker
                
            result = {
                "text": transcript,
                "speaker_segments": speaker_segments
                }
            
            # Clean up
            os.unlink(temp_path)
            
            return result
        else:
            print("No Deepgram API key available")
            return {"text": "", "error": "No API key", "speaker_segments": []}
        
    except Exception as e:
            raise HTTPException(status_code=400, detail=f"Transcription error: {str(e)}")


@app.post("/api/analyze")
async def analyze_conversation(request: SOAPRequest):
    """
    Analyze medical conversation and extract SOAP elements
    """
    try:
        # Prepare speaker segments in the format needed by the analyzer
        speaker_data = {}
        
        if request.speaker_segments:
            # Group segments by speaker
            for segment in request.speaker_segments:
                speaker = segment.get("speaker", "unknown")
                if speaker not in speaker_data:
                    speaker_data[speaker] = []
                
                speaker_data[speaker].append(segment)
        
        # Analyze the conversation
        soap_result = conversation_analyzer.analyze(request.conversation_text, speaker_data)
        
        return soap_result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis error: {str(e)}")


@app.post("/api/calibrate_speaker")
async def calibrate_speaker(speaker_type: str, request: TranscriptionRequest):
    """
    Calibrate speaker model with sample audio
    """
    try:
        if speaker_type not in ["doctor", "patient"]:
            raise HTTPException(status_code=400, detail="Speaker type must be 'doctor' or 'patient'")
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
        
        # Calibrate the speaker model
        success = audio_processor.diarizer.calibrate(speaker_type, temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        if success:
            return {"status": "success", "message": f"{speaker_type} calibration completed"}
        else:
            return {"status": "failure", "message": "Calibration failed"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Calibration error: {str(e)}")


@app.post("/api/generate_soap")
async def generate_soap(request: SOAPRequest):
    """
    Generate complete SOAP note from conversation
    """
    try:
        # Prepare speaker segments in the format needed by the analyzer
        speaker_data = {}
        
        if request.speaker_segments:
            # Group segments by speaker
            for segment in request.speaker_segments:
                speaker = segment.get("speaker", "unknown")
                if speaker not in speaker_data:
                    speaker_data[speaker] = []
                
                speaker_data[speaker].append(segment)
        
        # Analyze the conversation to extract SOAP elements
        soap_result = conversation_analyzer.analyze(request.conversation_text, speaker_data)
        
        # Format the SOAP note in a structured way
        formatted_soap = {
            "title": "Medical SOAP Note",
            "date": time.strftime("%Y-%m-%d"),
            "time": time.strftime("%H:%M:%S"),
            "soap": soap_result,
            "raw_transcript": request.conversation_text
        }
        
        return formatted_soap
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SOAP generation error: {str(e)}")


if __name__ == "__main__":
    # Run the app with uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)