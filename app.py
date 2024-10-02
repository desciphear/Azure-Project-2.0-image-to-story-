import streamlit as st
from dotenv import load_dotenv
import os
import winsound
from PIL import Image
from azure.core.exceptions import HttpResponseError

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speech_rec

def load_image(image_file):
     img = Image.open(image_file)
     return img


def Image_Analyze(file_name, genre):

    global cv_client
   
    try:
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')
        
        
        

        image_file = 'images/'+file_name
        
        
        
        with open(image_file, "rb") as f:
            image_data = f.read()

        cv_client = ImageAnalysisClient(
            endpoint = ai_endpoint,
            credential = AzureKeyCredential(ai_key)
        )

        st.write('\nAnalyzing image...')
        try:
            result = cv_client.analyze(
                image_data=image_data,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.DENSE_CAPTIONS
                ]
            )
        except HttpResponseError as e:
            print(f"status code: {e.status_code}")
            print(f"Reason: {e.reason}")
            print(f"Message: {e.error.message}")
        
        if result.caption is not None:
            captions = result.caption.text
    except Exception as ex:
        print(ex)
    storywrite(captions, genre)



def Image_Analyze_voice(file_name, genre):

    global cv_client
   
    try:
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')
        
        
        

        image_file = 'images/'+file_name
        
        
        
        with open(image_file, "rb") as f:
            image_data = f.read()

        cv_client = ImageAnalysisClient(
            endpoint = ai_endpoint,
            credential = AzureKeyCredential(ai_key)
        )

        st.write('\nAnalyzing image...')
        try:
            result = cv_client.analyze(
                image_data=image_data,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.DENSE_CAPTIONS
                ]
            )
        except HttpResponseError as e:
            print(f"status code: {e.status_code}")
            print(f"Reason: {e.reason}")
            print(f"Message: {e.error.message}")
        
        if result.caption is not None:
            captions = result.caption.text
    except Exception as ex:
        print(ex)
    storywrite_voice(captions, genre)



def storywrite_voice(topic, genre):
    load_dotenv()
    azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
    azure_oai_key = os.getenv("AZURE_OAI_KEY")
    azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
    
    
       

       
       
    client = AzureOpenAI(
        azure_endpoint = azure_oai_endpoint, 
        api_key=azure_oai_key,  
        api_version="2024-02-15-preview"
        )
    
    try:

        while True:
                    # Get input text
                    input_text = 'Write a complete story of 200 words on the topic : ' +topic +'The genre of story should be ' + genre
                    if input_text.lower() == "quit":
                        break
                    if len(input_text) == 0:
                        print("Please enter a prompt.")
                        continue

                    print("\nWriting Story Based On Image...\n\n")
                    
                    # Add code to send request...
                    # Send request to Azure OpenAI model
                    response = client.chat.completions.create(
                        model=azure_oai_deployment,
                        temperature=0.7,
                        max_tokens=400,
                        messages=[
                            
                            {"role": "user", "content": input_text}
                        ]
                    )
                    generated_text = response.choices[0].message.content

                    # Print the response
                    if generated_text is not None:
                      
                        
                        st.write( generated_text +"."+ "\n")
                        
                      
                        narrate(generated_text)
                        
                        st.download_button('Download Story as txt file', generated_text)  # Defaults to 'text/plain'

                        with open('myfile.csv') as f:
                            st.download_button('Download CSV', f)
                
                
                        break
                            

                        
                    else:
                         break    
    except Exception as ex:
            print(ex)




def storywrite(topic, genre):
    load_dotenv()
    azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
    azure_oai_key = os.getenv("AZURE_OAI_KEY")
    azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
    
    
       

       
       
    client = AzureOpenAI(
        azure_endpoint = azure_oai_endpoint, 
        api_key=azure_oai_key,  
        api_version="2024-02-15-preview"
        )
    
    try:

        while True:
                    # Get input text
                    input_text = 'Write a complete story of 200 words on the topic : ' +topic +'The genre of story should be ' + genre
                    if input_text.lower() == "quit":
                        break
                    if len(input_text) == 0:
                        print("Please enter a prompt.")
                        continue

                    print("\nWriting Story Based On Image...\n\n")
                    
                    # Add code to send request...
                    # Send request to Azure OpenAI model
                    response = client.chat.completions.create(
                        model=azure_oai_deployment,
                        temperature=0.7,
                        max_tokens=400,
                        messages=[
                            
                            {"role": "user", "content": input_text}
                        ]
                    )
                    generated_text = response.choices[0].message.content

                    # Print the response
                    if generated_text is not None:

                        
                        st.write( generated_text +"."+ "\n")

                        st.download_button('Download Story as txt file', generated_text)  # Defaults to 'text/plain'

                        with open('myfile.csv') as f:
                            st.download_button('Download CSV', f)

                        break     

                        
                    else:
                         break    
    except Exception as ex:
            print(ex)



def narrate(text):
    
    
   
    ai_key = os.getenv('SPEECH_KEY')
    ai_region = os.getenv('SPEECH_REGION')
    speech_config = speech_rec.SpeechConfig(ai_key, ai_region)

    #Choosing Voice for output and Creating Speech Synthesizer for spoken Output
    speech_config.speech_synthesis_voice_name = "en-GB-RyanNeural"
    audio_config = speech_rec.audio.AudioOutputConfig(filename="Audio/output.wav")
    
    #Calling Function To Generate Output Audio
    speech_synthesizer = speech_rec.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speak = speech_synthesizer.speak_text_async(text).get()
    
    
    

    if speak.reason != speech_rec.ResultReason.SynthesizingAudioCompleted:
        print(speak.reason)
    else:
        
        print(speak)



st.title('Image To Story')
st.subheader("Image")
    
file_name =''

    
image_file = st.file_uploader("Upload Images (png, jpg or jpeg)", type=["png","jpg","jpeg"])
    


view = st.button("Preview Image")
if view == True and image_file != None:
    st.image(load_image(image_file),width = 250)
else:
    st.write("Please Upload a valid Image")

genre = st.text_input("Please Enter Genre of story")
    


story = st.button("Write Story")
voice = st.toggle("Enable Narration")
    


print(file_name)

if image_file != None:  

    if story == True and voice == False:
        file_name = image_file.name
        with open(os.path.join("images", image_file.name),"wb")as f:
                f.write(image_file.getbuffer())
        if file_name !='':
            Image_Analyze(file_name, genre)
            
        

    elif story == True and voice == True:
            file_name = image_file.name
            with open(os.path.join("images", image_file.name),"wb")as f:
                    f.write(image_file.getbuffer())
            if file_name !='':
                Image_Analyze_voice(file_name, genre)
                st.header("Narration")
                audio_file = open('Audio/output.wav','rb')
                audio = audio_file.read()
                st.audio(audio, format='audio/ogg')
                
                
                
else:
    st.write("Please Upload a valid file")         
     

