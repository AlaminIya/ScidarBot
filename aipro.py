import streamlit as st
from pydub import AudioSegment
import google.generativeai as genai
import speech_recognition as sr
from PyPDF2 import PdfReader
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import io

st.title("Scidar AI Assistant")
genai.configure(api_key="")

def get_pdf_summary(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    # Initialize parser and tokenizer
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # Initialize LSA Summarizer
    summarizer = LsaSummarizer()
    
    # Summarize the text
    summary = summarizer(parser.document, 10)  # Adjust the number of sentences in the summary as needed
    
    # Convert summary to string
    summary_text = ""
    for sentence in summary:
        summary_text += str(sentence) + " "
    
    return summary_text

# Function to process speech input
#def process_speech_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something...")
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio)
        return query
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))



# Function to transcribe audio files
def transcribe_audio(audio_file):
    try:
        r = sr.Recognizer()

        # Convert the audio file to PCM WAV format
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)

        # Transcribe the audio data
        transcription = r.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand the audio.")
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))
    except Exception as e:
        st.write("An error occurred: {0}".format(e))




# Function to process query and response
def process_query_response(query):
    response = model.generate_content(query)

    # Displaying the Assistant Message
    with st.chat_message("assistant"):
        st.markdown(response.text)

    # Storing the User Message
    st.session_state.messages.append({"role": "user", "content": query})

    # Storing the Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": response.text})

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Initialize Generative AI Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    },
system_instruction = "You are a friendly AI Assitant for Solina. Specialized In researching and analyzing different project in a management consultancy firm Solina. Your responses only focus on health projects. You may be asked to give response based on pre uploaded files and research papers. You are also an analyst and a consultancy specialist. You help the HR team in solina to Analize CVs and evaluate employees. You also have speech-to-text abilities and can transliterate audio files to text.",
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
)

# Accept user input
query = st.text_input("Hello Friend, What can I do for you Today?")




# Calling the Function when Input is Provided
if query:
    process_query_response(query)
# Accept user input via microphone
audio_file = st.sidebar.file_uploader("Upload an audio file" ) 
if audio_file is not None:
    transcription = transcribe_audio(audio_file)
    if transcription:
        st.text_area("You said:", value=transcription)

# File Upload
uploaded_files = st.sidebar.file_uploader("Upload one or more PDF files here:", type="pdf", accept_multiple_files=True)

if uploaded_files:
    pdf_summaries = [get_pdf_summary([file]) for file in uploaded_files]
    summaries = "\n\n".join(pdf_summaries)
    
    # Process and display summary
    st.write("Summary of Uploaded PDF(s):")
    st.write(summaries)

    # Call function to process query and response with the summary
    process_query_response(summaries)
