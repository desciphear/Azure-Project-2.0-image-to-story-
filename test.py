from dotenv import load_dotenv
import os
import sys
from azure.core.exceptions import HttpResponseError

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speech_rec


def Image_Analyze():

    global cv_client
   
    try:
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')
        
        
        name = input("Enter Image Name: ")

        image_file = 'images/'+name+'.jpg'
        if len(sys.argv)>1:
            image_file = sys.argv[1]
        
        print(image_file)

        with open(image_file, "rb") as f:
            image_data = f.read()

        cv_client = ImageAnalysisClient(
            endpoint = ai_endpoint,
            credential = AzureKeyCredential(ai_key)
        )

        print('\nAnalyzing image...')
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
    storywrite(captions)


def storywrite(topic):
    load_dotenv()
    azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
    azure_oai_key = os.getenv("AZURE_OAI_KEY")
    azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")

    genre = input("\nWhat should be the genre of the story ?\n")    

    voice_enable = input("\nDo you want to narrate the story(y/n) ?\n")    
       
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
                        print( generated_text +"."+ "\n")
                        if voice_enable.lower() == "y":
                            narrate(generated_text)
                            break
                        elif voice_enable.lower() == "n":
                             print("\n")
                             break
                        else:
                             print("Invalid Response Please type y or n...")
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
    speech_synthesizer = speech_rec.SpeechSynthesizer(speech_config)
    
    #Calling Function To Generate Output Audio
    speak = speech_synthesizer.speak_text_async(text).get()
    if speak.reason != speech_rec.ResultReason.SynthesizingAudioCompleted:
        print(speak.reason)
    else:
        print(speak)

def main():
    Image_Analyze()

if __name__ == "__main__":
    main()


