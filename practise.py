import speech_recognition as sr
import time
import pyttsx3
r=sr.Recognizer()
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

     with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("listening...")
            audio = r.listen(source, timeout=3, phrase_time_limit=5)

     try:
            word = r.recognize_google(audio)
            print("You said:", word)

            text = word.lower()

            # exit
            if text in ["exit", "quit", "bye"]:
                speak("goodbye")
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
                        {"role": "system", "content": "You are aa personal assistant named jarvis like a alexa and gemini you are also like a humanoid robot so anser the question in short 1-2 sentences or lines only , and at last for every respond say if there any specific topic"
                        " then i am always here to help you {name}."},
                        {"role": "user", "content": word}
                    ]
                )

                answer = response.choices[0].message.content
                print("AI:", answer)
                speak(answer)

     except sr.UnknownValueError:
            speak("Sorry, I could not understand")

     except sr.RequestError:
            speak("Internet error")




    
    

     
