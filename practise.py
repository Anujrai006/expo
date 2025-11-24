import pyttsx3
import time
import speech_recognition as sr
from openai import OpenAI
import pygame
pygame.mixer.init()


r = sr.Recognizer()
def play_song(path):
    try:
        # ensure the mixer is initialized
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"mixer init error: {e}")

        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        return True
    except Exception as e:
        # print the error so the user can inspect logs in the terminal
        print(f"play_song error: {e}")
        # best-effort voice feedback; avoid raising if TTS also fails
        try:
            speak("Sorry, the song could not be played")
        except Exception:
            pass
        return False

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
    client = OpenAI(api_key="")
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
            elif "emotional" in text:
              speak("Playing an emotional song")
              play_song("C:/Users/Anuj Rai/python/expowork/expo/emotional.mp3")









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
                speak(answer)

     except sr.UnknownValueError:
            speak("Sorry, I could not understand")

     except sr.RequestError:
            speak("Internet error")
            speak('hello')
            




    
    

     
