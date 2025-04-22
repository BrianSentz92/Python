import pyttsx3
import PyPDF2
import re
from tkinter import Tk, filedialog
from datetime import datetime

# --- File Picker ---
def select_pdf_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )
    return file_path

# --- PDF Reader ---
def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# --- Formatter for Natural Speech ---
def format_text_for_speech(text):
    # Format phone numbers
    text = re.sub(r'\b(\d{3})[\s\-]?(\d{3})[\s\-]?(\d{4})\b', r'\1 \2 \3', text)

    # Format acronyms (e.g., CEO → C E O)
    text = re.sub(r'\b([A-Z]{2,})\b', lambda m: ' '.join(m.group(1)), text)

    # Format dates like MM/DD/YYYY → April 22, 2025
    def convert_date(match):
        try:
            date_obj = datetime.strptime(match.group(), '%m/%d/%Y')
            return date_obj.strftime('%B %-d, %Y')  # Mac/Linux
        except ValueError:
            try:
                date_obj = datetime.strptime(match.group(), '%m/%d/%Y')
                return date_obj.strftime('%B %#d, %Y')  # Windows fallback
            except:
                return match.group()
    text = re.sub(r'\b\d{2}/\d{2}/\d{4}\b', convert_date, text)

    # Format emails
    text = re.sub(r'([\w\.-]+)@([\w\.-]+)', lambda m: f"{m.group(1).replace('.', ' dot ')} at {m.group(2).replace('.', ' dot ')}", text)

    # Format URLs
    text = re.sub(r'(https?://[\w\./\-]+)', lambda m: m.group().replace('.', ' dot ').replace('/', ' slash '), text)

    # Format large numbers as digit-by-digit (except years)
    def digit_by_digit(match):
        number = match.group()
        if 1800 < int(number) < 2100:  # likely a year
            return number
        return ' '.join(number)
    text = re.sub(r'\b\d{4,}\b', digit_by_digit, text)

    return text

# --- Text-to-Speech Engine ---
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 130)  # Adjust for natural speed
    engine.setProperty('voice', 'com.apple.speech.synthesis.voice.evan')  # macOS voice
    engine.say(text)
    engine.runAndWait()

# --- Main App Flow ---
if __name__ == "__main__":
    file_path = select_pdf_file()
    if file_path:
        text = extract_text_from_pdf(file_path)
        cleaned_text = format_text_for_speech(text)
        speak_text(cleaned_text)
    else:
        print("No file selected.")
