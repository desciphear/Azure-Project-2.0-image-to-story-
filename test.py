from dotenv import load_dotenv
import os
import sys
from azure.core.exceptions import HttpResponseError

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI


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

    genre = input("What should be the genre of the story ?\n")        
        
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
                        break
                    else:
                         break    
    except Exception as ex:
            print(ex)



def main():
    Image_Analyze()

if __name__ == "__main__":
    main()


