import re

def extract_soap(text: str):
    """
    Extracts problems and medicines from the transcript text.
    """
    problems = []
    medicines = []

    lines = text.lower().split(".")
    for line in lines:
        # Simple symptom detection
        if any(word in line for word in ["headache", "pain", "fever", "cold", "cough", "sore throat"]):
            problems.append(line.strip())

        # Simple medicine detection
        if "take" in line or "tablet" in line or "medicine" in line:
            medicines.append(line.strip())

    # Format as SOAP
    return {
        "subjective": problems,
        "objective": [],
        "assessment": [],
        "plan": medicines
    }
