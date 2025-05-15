from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import re
import asyncio
from typing import List, Dict, Any, Optional
import uvicorn
from vosk import Model, KaldiRecognizer
import os


class MedicalConversationAnalyzer:
    """
    Advanced analyzer for medical conversations to extract SOAP elements
    from doctor-patient interactions.
    """
    
    def __init__(self):
        # Initialize patterns for different medical entities
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for medical entity extraction"""
        
        # Symptoms and problems patterns
        self.symptoms_patterns = [
            # General symptom patterns
            r"(having|suffering from|experiencing|feel|feeling|with|has|got|getting)\s+(a|an)?\s*([a-zA-Z\s]+)(pain|ache|fever|cough|cold|nausea|fatigue|weakness|infection|issue|problem)",
            # Direct mentions of symptoms
            r"(headache|migraine|fever|cough|cold|sore throat|nausea|vomiting|diarrhea|constipation|back pain|chest pain|shortness of breath|difficulty breathing)",
            # Duration-based symptoms
            r"(been|having|had)\s+(a|an)?\s*([a-zA-Z\s]+)(for|since)\s+(\d+)\s+(day|days|week|weeks|month|months)",
            # Pain localization
            r"(pain|ache|discomfort)\s+in\s+(my|the)\s+([a-zA-Z\s]+)",
            # Patient reporting "I have X"
            r"I\s+have\s+(a|an)?\s+([a-zA-Z\s]+)(pain|ache|fever|cough|cold|nausea|problem|issue)"
        ]
        
        # Medication patterns
        self.medication_patterns = [
            # Direct prescriptions
            r"(take|prescribe|recommend|give|suggest|try)\s+([A-Za-z0-9\s\-]+)\s+(once|twice|thrice|\d+\s+times)\s+(?:a|per|every|each)?\s*(?:day|daily|week|month)?",
            # Medication with dosage
            r"([A-Za-z0-9\s\-]+)\s+(\d+\s*mg|\d+\s*ml|\d+\s*tab|\d+\s*pill|\d+\s*capsule)",
            # Medication with frequency
            r"([A-Za-z0-9\s\-]+)\s+(\d+\s+times)(?:\s+a\s+day)?(?:\s+for\s+(\d+)\s+days)?",
            # Advising medication
            r"would\s+(recommend|suggest|prescribe|advise)\s+([A-Za-z0-9\s\-]+)"
        ]
        
        # Duration and instructions patterns
        self.duration_patterns = [
            # Treatment duration
            r"for\s+(\d+)\s+(day|days|week|weeks|month|months)",
            # Before/after meals
            r"(before|after)\s+(meals?|food|eating)",
            # Special instructions
            r"(with|without)\s+(water|food|meals?)",
            # Time of day
            r"(in the|at)\s+(morning|afternoon|evening|night|bedtime)"
        ]
        
        # Diagnosis patterns
        self.diagnosis_patterns = [
            # Direct diagnosis
            r"(you have|it's|it is|looks like|seems to be|might be|could be|diagnosis is|diagnosing)\s+(a|an)?\s*([A-Za-z\s\-]+)",
            # Suggesting condition
            r"(suggesting|pointing to|indicative of|consistent with)\s+(a|an)?\s*([A-Za-z\s\-]+)",
            # Test results showing
            r"(test results|tests|x-ray|scan|mri|ct scan)\s+(show|indicate|reveal|confirm)\s+(a|an)?\s*([A-Za-z\s\-]+)"
        ]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze medical conversation text to extract SOAP elements
        
        Args:
            text: The conversation text to analyze
            
        Returns:
            Dictionary with SOAP note elements
        """
        # Normalize text: lowercase and remove extra whitespace
        normalized_text = ' '.join(text.lower().split())
        
        # Extract medical entities
        problems = self._extract_problems(normalized_text)
        medications = self._extract_medications(normalized_text)
        durations = self._extract_durations(normalized_text)
        diagnoses = self._extract_diagnoses(normalized_text)
        
        # Build SOAP structure
        soap = {
            "subjective": self._format_subjective(problems),
            "objective": "",  # Usually filled manually by doctor
            "assessment": self._format_assessment(diagnoses, problems),
            "plan": self._format_plan(medications, durations)
        }
        
        return soap
    
    def _extract_problems(self, text: str) -> List[str]:
        """Extract patient problems and symptoms from text"""
        problems = []
        
        for pattern in self.symptoms_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract the symptom based on match groups
                # This requires handling different pattern structures
                if len(match.groups()) >= 4 and match.group(3) and match.group(4):
                    # Pattern with verb + symptom + type
                    symptom = (match.group(3) + match.group(4)).strip()
                    problems.append(symptom)
                elif len(match.groups()) >= 1:
                    # Direct symptom mention
                    symptom = match.group(0).strip()
                    problems.append(symptom)
        
        # Remove duplicates and normalize
        return list(set(self._normalize_entities(problems)))
    
    def _extract_medications(self, text: str) -> List[Dict[str, str]]:
        """Extract medication information from text"""
        medications = []
        
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract different components based on pattern
                if len(match.groups()) >= 2:
                    med_info = {}
                    
                    # Get the medication name (usually in group 1 or 2)
                    if "prescribe" in match.group(1).lower() or "take" in match.group(1).lower():
                        med_info["name"] = match.group(2).strip()
                    else:
                        med_info["name"] = match.group(1).strip()
                    
                    # Try to extract dosage or frequency
                    if len(match.groups()) >= 3 and match.group(3):
                        if "mg" in match.group(2) or "ml" in match.group(2) or "tab" in match.group(2):
                            med_info["dosage"] = match.group(2).strip()
                            med_info["frequency"] = match.group(3).strip() if match.group(3) else ""
                        else:
                            med_info["frequency"] = match.group(2).strip() if match.group(2) else ""
                    
                    medications.append(med_info)
        
        return medications
    
    def _extract_durations(self, text: str) -> List[str]:
        """Extract treatment durations and instructions from text"""
        durations = []
        
        for pattern in self.duration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                instruction = match.group(0).strip()
                durations.append(instruction)
        
        # Remove duplicates
        return list(set(durations))
    
    def _extract_diagnoses(self, text: str) -> List[str]:
        """Extract potential diagnoses from doctor's statements"""
        diagnoses = []
        
        for pattern in self.diagnosis_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract diagnosis based on pattern structure
                if len(match.groups()) >= 3 and match.group(3):
                    diagnosis = match.group(3).strip()
                    diagnoses.append(diagnosis)
                elif len(match.groups()) >= 4 and match.group(4):
                    diagnosis = match.group(4).strip()
                    diagnoses.append(diagnosis)
        
        # Remove duplicates and normalize
        return list(set(self._normalize_entities(diagnoses)))
    
    def _normalize_entities(self, entities: List[str]) -> List[str]:
        """Clean and normalize extracted entities"""
        normalized = []
        for entity in entities:
            # Remove articles and common fillers
            entity = re.sub(r'\b(a|an|the|some|this|that)\b', '', entity)
            # Remove extra whitespace
            entity = re.sub(r'\s+', ' ', entity).strip()
            if entity:
                normalized.append(entity)
        
        return normalized
    
    def _format_subjective(self, problems: List[str]) -> str:
        """Format problems into subjective section"""
        if not problems:
            return ""
        
        return "Patient reports: " + ", ".join(problems)
    
    def _format_assessment(self, diagnoses: List[str], problems: List[str]) -> str:
        """Format diagnoses into assessment section"""
        if diagnoses:
            return "Assessment: " + ", ".join(diagnoses)
        elif problems:
            # If no explicit diagnosis, use problems as assessment
            return "Potential diagnosis based on symptoms: " + ", ".join(problems)
        else:
            return ""
    
    def _format_plan(self, medications: List[Dict[str, str]], durations: List[str]) -> str:
        """Format medications and durations into plan section"""
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


# Initialize FastAPI app
app = FastAPI(title="Medical Conversation to SOAP Prescription API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class SOAPNote(BaseModel):
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""

# Load Vosk model - update path as needed
model_path = "vosk-model"
if not os.path.exists(model_path):
    print(f"Please download the model from https://alphacephei.com/vosk/models and unpack to {model_path}")
    exit(1)

model = Model(model_path)
print("Vosk model loaded")

# Initialize medical conversation analyzer
analyzer = MedicalConversationAnalyzer()
print("Medical conversation analyzer initialized")

# Store active connections
active_connections: List[WebSocket] = []

# In-memory conversation storage
conversation_buffer = []
current_soap = SOAPNote()

# WebSocket endpoint for real-time audio streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    # Initialize recognizer for this connection
    rec = KaldiRecognizer(model, 16000)  # 16kHz sample rate (common for speech)
    
    try:
        conversation_text = ""
        
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Process with Vosk
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if "text" in result and result["text"].strip():
                    text = result["text"]
                    conversation_text += " " + text
                    conversation_buffer.append(text)
                    
                    # Send recognized text back to client
                    await websocket.send_json({
                        "type": "transcript",
                        "text": text
                    })
                    
                    # Analyze the ongoing conversation
                    soap_data = analyzer.analyze(conversation_text)
                    
                    # Send the updated SOAP note back to the client
                    await websocket.send_json({
                        "type": "soap_update",
                        "soap": {
                            "subjective": soap_data["subjective"],
                            "objective": soap_data["objective"],
                            "assessment": soap_data["assessment"],
                            "plan": soap_data["plan"]
                        }
                    })
            else:
                # Partial result
                partial = json.loads(rec.PartialResult())
                if "partial" in partial and partial["partial"].strip():
                    await websocket.send_json({
                        "type": "partial",
                        "text": partial["partial"]
                    })
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"Error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# API endpoint to manually update the SOAP note
@app.post("/update_soap")
async def update_soap(soap: SOAPNote):
    # Update the current SOAP note
    global current_soap
    current_soap = soap
    return {"status": "success", "soap": current_soap}

# API endpoint to get the current SOAP note
@app.get("/get_soap")
async def get_soap():
    return current_soap

# API endpoint to clear the conversation and SOAP note
@app.post("/clear")
async def clear_conversation():
    global conversation_buffer, current_soap
    conversation_buffer = []
    current_soap = SOAPNote()
    return {"status": "success", "message": "Conversation and SOAP note cleared"}

# Run the server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)