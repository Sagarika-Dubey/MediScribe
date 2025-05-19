// Global variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recognitionActive = false;
let recognition;

// DOM Elements
document.addEventListener('DOMContentLoaded', () => {
    // Initialize SpeechRecognition if available
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        
        setupSpeechRecognition();
    } else {
        alert("Speech recognition is not supported in your browser. Please use Chrome or Edge.");
    }
    
    // Set up event listeners
    setupEventListeners();
});

// Set up Speech Recognition
function setupSpeechRecognition() {
    let currentSpeaker = 'doctor'; // Default speaker
    let currentTranscript = '';
    
    recognition.onstart = () => {
        recognitionActive = true;
        updateStatus('recording');
    };
    
    recognition.onend = () => {
        recognitionActive = false;
        if (isRecording) {
            recognition.start(); // Restart if recording is still active
        } else {
            updateStatus('ready');
        }
    };
    
    recognition.onresult = (event) => {
        let interimTranscript = '';
        
        // Process results
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            
            if (event.results[i].isFinal) {
                currentTranscript += transcript + ' ';
                // Add entry to transcript display
                addTranscriptEntry(currentSpeaker, currentTranscript.trim());
                currentTranscript = '';
            } else {
                interimTranscript += transcript;
            }
        }
        
        // Show interim results
        if (interimTranscript) {
            // Update the last transcript entry with interim results if it exists
            const transcriptEl = document.getElementById('transcript');
            const lastEntry = transcriptEl.lastElementChild;
            
            if (lastEntry && lastEntry.querySelector(`.speaker-${currentSpeaker}`)) {
                const textEl = lastEntry.querySelector('.transcript-text');
                textEl.textContent = currentTranscript + interimTranscript;
            } else {
                // Create a new interim entry
                const interimEntry = document.createElement('div');
                interimEntry.className = 'transcript-entry interim';
                interimEntry.innerHTML = `
                    <div class="speaker-${currentSpeaker}">${currentSpeaker === 'doctor' ? 'Doctor' : 'Patient'}</div>
                    <div class="transcript-text">${interimTranscript}</div>
                `;
                transcriptEl.appendChild(interimEntry);
            }
        }
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'not-allowed') {
            alert('Microphone access denied. Please enable microphone permissions.');
        }
        updateStatus('error');
    };
}

// Add a transcript entry to the display
function addTranscriptEntry(speaker, text) {
    const transcriptEl = document.getElementById('transcript');
    
    // Remove any interim entry
    const interimEntry = transcriptEl.querySelector('.interim');
    if (interimEntry) {
        transcriptEl.removeChild(interimEntry);
    }
    
    // Create new entry
    const entry = document.createElement('div');
    entry.className = 'transcript-entry';
    entry.innerHTML = `
        <div class="speaker-${speaker}">${speaker === 'doctor' ? 'Doctor' : 'Patient'}</div>
        <div class="transcript-text">${text}</div>
    `;
    
    transcriptEl.appendChild(entry);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// Set up event listeners
function setupEventListeners() {
    // Recording buttons
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    
    startButton.addEventListener('click', startRecording);
    stopButton.addEventListener('click', stopRecording);
    
    // Audio upload
    const audioUpload = document.getElementById('audioUpload');
    audioUpload.addEventListener('change', handleAudioUpload);
    
    // Calibration buttons
    const calibrateDoctor = document.getElementById('calibrateDoctor');
    const calibratePatient = document.getElementById('calibratePatient');
    
    calibrateDoctor.addEventListener('click', () => setCurrentSpeaker('doctor'));
    calibratePatient.addEventListener('click', () => setCurrentSpeaker('patient'));
    
    // Generate SOAP Note button
    const generateSoapButton = document.getElementById('generateSoapButton');
    generateSoapButton.addEventListener('click', generateSoapNote);
    
    // Edit prescription button
    const editPlanButton = document.getElementById('editPlanButton');
    editPlanButton.addEventListener('click', openPrescriptionModal);
    
    // Modal buttons
    const closeModal = document.querySelector('.close-modal');
    closeModal.addEventListener('click', closePrescriptionModal);
    
    const addMedicationBtn = document.getElementById('addMedicationBtn');
    addMedicationBtn.addEventListener('click', addMedication);
    
    const savePrescriptionBtn = document.getElementById('savePrescriptionBtn');
    savePrescriptionBtn.addEventListener('click', savePrescription);
    
    // Print button
    const printButton = document.getElementById('printButton');
    printButton.addEventListener('click', () => window.print());
}

// Update status indicator
function updateStatus(status) {
    const statusLight = document.getElementById('statusLight');
    const statusText = document.getElementById('statusText');
    
    statusLight.className = 'status-light';
    
    switch (status) {
        case 'recording':
            statusLight.classList.add('recording');
            statusText.textContent = 'Recording';
            document.getElementById('startButton').disabled = true;
            document.getElementById('stopButton').disabled = false;
            break;
        case 'processing':
            statusLight.classList.add('processing');
            statusText.textContent = 'Processing';
            break;
        case 'error':
            statusText.textContent = 'Error';
            document.getElementById('startButton').disabled = false;
            document.getElementById('stopButton').disabled = true;
            break;
        default:
            statusText.textContent = 'Ready';
            document.getElementById('startButton').disabled = false;
            document.getElementById('stopButton').disabled = true;
    }
}

// Start recording
function startRecording() {
    isRecording = true;
    audioChunks = [];
    
    // Start browser speech recognition
    if (recognition && !recognitionActive) {
        recognition.start();
    }
    
    // Start media recorder if available
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.start();
                updateStatus('recording');
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                alert('Unable to access the microphone. Please check your permissions.');
                updateStatus('error');
            });
    } else {
        console.warn('MediaRecorder API not supported in this browser');
        // Fall back to just speech recognition
        updateStatus('recording');
    }
}

// Stop recording
function stopRecording() {
    isRecording = false;
    
    // Stop speech recognition
    if (recognition && recognitionActive) {
        recognition.stop();
    }
    
    // Stop media recorder if active
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        
        mediaRecorder.onstop = () => {
            updateStatus('processing');
            
            // Process audio chunks if needed
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            
            // Here you could implement sending the audio to a backend service
            // for more accurate transcription if needed
            
            setTimeout(() => {
                updateStatus('ready');
            }, 1000);
        };
    } else {
        updateStatus('ready');
    }
}

// Handle audio file upload
function handleAudioUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const uploadFileName = document.getElementById('uploadFileName');
    uploadFileName.textContent = file.name;
    
    updateStatus('processing');
    
    // Here you would process the uploaded audio file
    // For demo purposes, we'll just simulate processing
    
    // In a real implementation, you would send this file to a backend service
    // for transcription and then update the transcript display
    
    setTimeout(() => {
        alert(`Audio file "${file.name}" uploaded. In a production system, this would be processed for transcription.`);
        updateStatus('ready');
    }, 2000);
}

// Set current speaker for transcription
function setCurrentSpeaker(speaker) {
    currentSpeaker = speaker;
    alert(`Now calibrated for ${speaker === 'doctor' ? 'Doctor' : 'Patient'} speaking`);
}

// Generate SOAP Note from transcript
function generateSoapNote() {
    updateStatus('processing');
    
    // Get transcript content
    const transcriptEl = document.getElementById('transcript');
    const transcriptText = transcriptEl.innerText;
    
    // In a real implementation, you would send the transcript to an AI service
    // to generate the SOAP note. Here we'll simulate with some example content.
    
    setTimeout(() => {
        // Sample SOAP note for demonstration
        document.getElementById('subjective').innerHTML = 
            `Patient reports experiencing intermittent headaches for the past 3 weeks. 
            Describes pain as "throbbing" primarily on the right side of the head. 
            Pain rated as 6/10 at worst. Associated symptoms include mild nausea but no vomiting. 
            Denies visual disturbances or aura. States that stress at work may be a trigger. 
            Has been taking over-the-counter ibuprofen with minimal relief.`;
            
        document.getElementById('assessment').innerHTML = 
            `1. Tension headache - likely related to reported work stress
            2. Possible migraine - considering unilateral nature and associated nausea
            3. Dehydration - patient admits to poor fluid intake during work hours`;
            
        document.getElementById('plan').innerHTML = 
            `1. Sumatriptan 50mg oral, take at onset of headache, may repeat after 2 hours if no relief, max 200mg/day
            2. Stress management techniques discussed
            3. Increase water intake to minimum 2L daily
            4. Follow-up in 3 weeks
            5. If symptoms worsen or change in character, return sooner`;
            
        updateStatus('ready');
    }, 2000);
}

// Prescription modal functions
function openPrescriptionModal() {
    const modal = document.getElementById('prescriptionModal');
    const medicationsList = document.getElementById('medicationsList');
    
    // Clear existing medications
    medicationsList.innerHTML = `
        <div class="medication-header medication-item">
            <div>Medication</div>
            <div>Dosage</div>
            <div>Frequency</div>
            <div></div>
        </div>
    `;
    
    // Parse existing medications from plan
    const planText = document.getElementById('plan').innerText;
    const medicationRegex = /(\d+\.\s*)([^,-]+),?\s*([^,]+)(?:,\s*([^,]+))?/g;
    let match;
    let found = false;
    
    while ((match = medicationRegex.exec(planText)) !== null) {
        found = true;
        const medName = match[2].trim();
        const dosage = match[3].trim();
        const frequency = match[4] ? match[4].trim() : '';
        
        addMedicationToList(medName, dosage, frequency);
    }
    
    // If no medications were found, add an empty one
    if (!found) {
        addMedication();
    }
    
    modal.style.display = 'block';
}

function closePrescriptionModal() {
    const modal = document.getElementById('prescriptionModal');
    modal.style.display = 'none';
}

function addMedication() {
    const medicationsList = document.getElementById('medicationsList');
    
    const medItem = document.createElement('div');
    medItem.className = 'medication-item';
    medItem.innerHTML = `
        <input type="text" class="med-name" placeholder="Medication name">
        <input type="text" class="med-dosage" placeholder="Dosage">
        <input type="text" class="med-frequency" placeholder="Frequency">
        <button class="delete-med"><i class="fas fa-trash"></i></button>
    `;
    
    medicationsList.appendChild(medItem);
    
    // Add event listener to delete button
    const deleteBtn = medItem.querySelector('.delete-med');
    deleteBtn.addEventListener('click', () => {
        medicationsList.removeChild(medItem);
    });
}

function addMedicationToList(name, dosage, frequency) {
    const medicationsList = document.getElementById('medicationsList');
    
    const medItem = document.createElement('div');
    medItem.className = 'medication-item';
    medItem.innerHTML = `
        <input type="text" class="med-name" value="${name}" placeholder="Medication name">
        <input type="text" class="med-dosage" value="${dosage}" placeholder="Dosage">
        <input type="text" class="med-frequency" value="${frequency}" placeholder="Frequency">
        <button class="delete-med"><i class="fas fa-trash"></i></button>
    `;
    
    medicationsList.appendChild(medItem);
    
    // Add event listener to delete button
    const deleteBtn = medItem.querySelector('.delete-med');
    deleteBtn.addEventListener('click', () => {
        medicationsList.removeChild(medItem);
    });
}

function savePrescription() {
    const medicationsList = document.getElementById('medicationsList');
    const medications = medicationsList.querySelectorAll('.medication-item:not(.medication-header)');
    
    let prescriptionText = '';
    
    medications.forEach((med, index) => {
        const name = med.querySelector('.med-name').value.trim();
        const dosage = med.querySelector('.med-dosage').value.trim();
        const frequency = med.querySelector('.med-frequency').value.trim();
        
        if (name) {
            prescriptionText += `${index + 1}. ${name}`;
            if (dosage) prescriptionText += `, ${dosage}`;
            if (frequency) prescriptionText += `, ${frequency}`;
            prescriptionText += '<br>';
        }
    });
    
    // Update the plan section with the new prescription
    const planEl = document.getElementById('plan');
    planEl.innerHTML = prescriptionText;
    
    // Close the modal
    closePrescriptionModal();
}