import speech_recognition as sr
import time
import pyttsx3
r=sr.Recognizer()
from openai import OpenAI

import speech_recognition as sr
import time
import pyttsx3
from openai import OpenAI

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
    client = OpenAI(api_key="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")
    with sr.Microphone() as source:
      r.adjust_for_ambient_noise(source)
      print("speak jarvis to activate it")
      audio=r.listen(source,timeout=5,phrase_time_limit=3)
      com=r.recognize_google(audio)
    try:
     if "jarvis" in com.lower():
        print(com)
        speak("yes boss i am active now")
     else:
        exit()
        
    except TimeoutError:
     print('sorry')
    while True: 

     

     try:
         with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("listening...")
            audio = r.listen(source, timeout=3, phrase_time_limit=5)
            
            word = r.recognize_google(audio)
            print("You said:", word)

            text = word.lower()

            # exit
            if text in ["exit", "quit", "bye"]:
                speak("goodbye")
                break
                # break
            

            # your name?
            elif "your name" in text:
                speak("My name is Jarvis, and what's your name?")

            # designer?
            elif "who designed you" in text:
                speak("I was designed by the group of class 11 students")

            # age?
            elif "your age" in text:
                speak("I don't have a specific age. Just wish me happy birthday anytime!")

            # user saying name
            elif "my name is" in text:
                name = text.split()[-1]    # extract last word
                speak(f"Nice to meet you, {name}")

            # assistant remembering user name
            elif "do you know my name" in text:
                if name:
                    speak(f"Yes, your name is {name}")
                else:
                    speak("You have not told me your name yet")

            # ask GPT
            else:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are aa personal assistant named jarvis like a alexa and gemini you are also like a humanoid robot so anser the question in short 1-2 sentences or lines only , store the data you recently said when the user asked the questions based on the recently anser you can reply smoothly sometimes like a mcq wise questions give the very versy short ansers only no need of explaination , and  sometimes according to situation at last for every respond say if there any specific topic"
                        " then i am always here to help you."},
                        {"role": "user", "content": word}
                    ]
                )

                answer = response.choices[0].message.content
                # print("AI:", answer)
                speak(answer)

     except sr.UnknownValueError:
            speak("Sorry, I could not understand")

     except sr.RequestError:
            speak("Internet error")
            speak('hello')




    
    

     
