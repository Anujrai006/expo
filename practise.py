import pyttsx3
import time
import speech_recognition as sr
from openai import OpenAI
import webbrowser
import datetime
import requests
r = sr.Recognizer()
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
driver=None
# engine=pyttsx3.init()
# voices=engine.getProperty('voices')
# engine.setProperty('voice',voices[1].id)
# engine.setProperty('rate',175)

def speak(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()
    

if __name__ == "__main__":
    print("initializing")
    speak("initializing")
    print("speak jarvis to wake up")
def save_name(name):
    with open("username.txt", "w") as f:
        f.write(name)

def load_name():
    try:
        with open("username.txt", "r") as f:
            return f.read().strip()
    except:
        return ""
name = load_name()
def save_city(city):
    with open("cityname.txt", "w") as c:
                       c.write(city)
def load_city():
        try:
            with open("cityname.txt", "r") as c:
                return c.read().strip()
        except:
            return ""
city = load_city()
# store the name safely

    # openai client (keep OUTSIDE the loop)
    client = OpenAI(api_key="")
    with sr.Microphone() as source:
      r.adjust_for_ambient_noise(source)
      print("speak jarvis to activate it")
      audio=r.listen(source,timeout=5,phrase_time_limit=3)
      com=r.recognize_google(audio)
    try:
        audio = r.listen(source, timeout=5, phrase_time_limit=4)
        com = r.recognize_google(audio)
        if "jarvis" in com.lower():
            speak("yes boss I am active now")
        else:
            exit()
    except:
        speak("sorry try again")
    
while True: 
    try:
         with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("listening...")
            try:
                audio = r.listen(source, timeout=4, phrase_time_limit=7)
                word = r.recognize_google(audio)
                print("You said:", word)
            except sr.WaitTimeoutError:
                 print("No speech detected ,trying again...")
                 continue
            except sr.UnknownValueError:
                 print("Couldn't understand, try again....")
                 continue
            except sr.RequestError:
                 print("Internet problem, try again...")
                 continue
            
        

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
            elif "who designed you" in text or "designed you" in text or "design you" in text:
                speak("I was designed by the group of 7 members of class 11 students")
            elif "open google" in text.lower():
                webbrowser.open("www.google.com")
                speak("opening google")
            elif "open youtube" in text.lower():
                webbrowser.open("www.youtube.com")
                speak("opening youtube")
                # age?
            elif "emotional  song" in text or "emotional" in text:
                speak("enjoy emotional music...")
                
    # Setup Chrome options to disable notifications and autoplay block
                chrome_options = Options()
                chrome_options.add_argument("--disable-notifications")
                chrome_options.add_argument("--start-maximized")
    
                driver = webdriver.Chrome(options=chrome_options)
                driver.get("https://youtu.be/Sc1OI1i-Kgs")  # your song link
    
                time.sleep(5)  # wait for page to load
    
                try:
        # Find the play button and click it
                    play_button = driver.find_element(By.CLASS_NAME, "ytp-large-play-button")
                    play_button.click()
                    speak("Music is now playing")
        
        # Optional: sleep for a few seconds so the assistant doesn't listen while loading
                    time.sleep(180)
        
                except Exception as e:
                    print(e)
                    speak("Could not auto-play the video, please click play manually")

            
 
            elif "stop music" in text or "stop song" in text:
                if driver:
                     driver.quit()
                     driver = None
                     speak("Music stopped")
                else:
                     speak("No music is playing")
            elif "your age" in text:
                speak("I don't have a specific age. Just wish me happy birthday anytime!")
            elif "i am from" in text:
                 city=text.split()[-1]
                 save_city(city)
                 speak(f"woow {city} is wonderful place ")
            elif "time" in text and "now"  in text:
                now = datetime.datetime.now().strftime("%I:%M %p")
                speak(f"The time is {now}")

            elif "date" in text and "today" in text:
                 today = datetime.date.today().strftime("%B %d, %Y")
                 speak(f"Today's date is {today}")
            elif "temperature" in text:
                 
                 try:
                     url = f"http://wttr.in/{city}?format=j1"
                     response = requests.get(url).json()
                     temp = response["current_condition"][0]["temp_C"]
                     speak(f"The current temperature in {city} is {temp} degree celsius")
                 except Exception as e:
                     speak("Sorry, I cannot fetch the temperature right now.")
                     print(e)
            elif "my name is" in text:
              name = text.split()[-1]
              save_name(name)   # <-- Save it permanently
              speak(f"Nice to meet you, {name}")
            elif "do you know my name" in text:
                if name:
                  speak(f"Yes, your name is {name} do you need some extra help")
                else:
                   speak("You have not told me your name yet")


            else:
                try:
                    response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are aa personal assistant named jarvis like a alexa and gemini you are also like a humanoid robot so anser the question clearly  , store the data you recently said when the user asked the questions based on the recently anser you can reply smoothly sometimes like a mcq wise questions give the very versy short ansers only no need of explaination , and  sometimes according to situation at last for every respond say if there any specific topic"
                        " then i am always here to help you."},
                        {"role": "user", "content": word}
                          ]
                          )

                    answer = response.choices[0].message.content
                    speak(answer)
                    print(answer)
                except Exception as e:
                     print(e)
                     speak("try again")
 
    except sr.UnknownValueError:
            speak("Sorry, I could not understand")

    except sr.RequestError:
            speak("Internet error")
        


            




    
    

     
