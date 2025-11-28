import pyttsx3
import time
import speech_recognition as sr
from openai import OpenAI
import pygame
import webbrowser
import datetime
import requests
r = sr.Recognizer()
   

def speak(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

if __name__ == "__main__":
    print("initializing")
    speak("initializing")

    # store the name safely
    name = ""

    # openai client (keep OUTSIDE the loop)
    client=OpenAI(api_key="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")
    with sr.Microphone() as source:
      r.adjust_for_ambient_noise(source)
      print("speak jarvis to activate it")
      audio=r.listen(source,timeout=5,phrase_time_limit=4)
      com=r.recognize_google(audio)
    try:
     if "jarvis" in com.lower():
        print(com)
        speak("yes boss i am active now")
     else:
        exit()
        
    except TimeoutError:
     print('sorry')
     speak("sorry try again")
    while True: 

     

     try:
         with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("listening...")
            audio = r.listen(source, timeout=4, phrase_time_limit=7)
            
            word = r.recognize_google(audio)
            print("You said:", word)

            text = word.lower()

            # exit
            if text in ["exit", "quit", "bye"]:
                speak("See you again {name} with full energy ")
                break
                # break
            

            # your name?
            elif "your name" in text:
                speak("My name is Jarvis, and what's your name?")

            # designer?
            elif "who designed you" or "designed you" or "design you" in text:
                speak("I was designed by the group of class 11 students")
            elif "open google" in text.lower():
                webbrowser.open("www.google.com")
                # age?

            elif "your age" in text:
                speak("I don't have a specific age. Just wish me happy birthday anytime!")

            # user saying name
            elif "my name is" in text:
                name = text.split()[-1]    # extract last word
                speak(f"Nice to meet you, {name}")
            elif "time" and "now"  in text:
                now = datetime.datetime.now().strftime("%I:%M %p")
                speak(f"The time is {now}")

            elif "date" and "today" in text:
                 today = datetime.date.today().strftime("%B %d, %Y")
                 speak(f"Today's date is {today}")
            # elif "temperature" in text or "weather" in text:
            # elif "temperature" in text or "weather" in text:
            #     try:
            #         api_key = "5ca60709900a47a3bf15d6fa8262dc98"

            #         city = "Itahari"
        
            #         url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

            #         response = requests.get(url)
            #         data = response.json()
            #         print(data)

            #         temp = data["main"]["temp"]
            #         speak(f"The current temperature in {city} is {temp} degree celsius")

            #     except Exception as e:
            #          speak("Sorry, I cannot fetch the temperature right now.")
            #          print("Error:", e)
            elif "temperature" in text:
                 city = "Itahari"
                 try:
                     url = f"http://wttr.in/{city}?format=j1"
                     response = requests.get(url).json()
                     temp = response["current_condition"][0]["temp_C"]
                     speak(f"The current temperature in {city} is {temp} degree celsius")
                 except Exception as e:
                     speak("Sorry, I cannot fetch the temperature right now.")
                     print(e)


            # assistant remembering user name
            elif "do you know my name" in text:
                if name:
                    speak(f"Yes, your name is {name}")
                else:
                    speak("You have not told me your name yet")
            else:
                try:
                    response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are aa personal assistant named jarvis like a alexa and gemini you are also like a humanoid robot so anser the question in short 1-2 sentences or lines only , store the data you recently said when the user asked the questions based on the recently anser you can reply smoothly sometimes like a mcq wise questions give the very versy short ansers only no need of explaination , and  sometimes according to situation at last for every respond say if there any specific topic"
                        " then i am always here to help you."},
                        {"role": "user", "content": word}
                          ]
                          )

                    answer = response.choices[0].message.content
                    speak(answer)
                    print(answer)
                except UnboundLocalError as e:
                    print(e)
                    speak("sorry try again")
 
     except sr.UnknownValueError:
            speak("Sorry, I could not understand")

     except sr.RequestError:
            speak("Internet error")
            speak('hello')
            

            




    
    

     
