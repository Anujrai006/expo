import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import pyttsx3
import time
import speech_recognition as sr
from openai import OpenAI
import webbrowser
import datetime
import requests
import threading
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import os

class JarvisUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS - AI Assistant")
        self.root.geometry("1000x700")
        self.root.configure(bg="#0a0e27")
        self.root.resizable(True, True)
        
        # Initialize components
        self.r = sr.Recognizer()
        self.driver = None
        self.name = self.load_name()
        self.city = self.load_city()
        self.is_listening = False
        self.client = OpenAI(api_key="")  # Add your API key here
        
        # Set up the UI
        self.setup_ui()
        self.update_status("System Ready")
        
    def setup_ui(self):
        """Create the main UI layout"""
        # Title bar
        title_frame = tk.Frame(self.root, bg="#1a1f3a", height=60)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        
        title_label = tk.Label(
            title_frame,
            text="🤖 JARVIS - Intelligent Voice Assistant",
            font=("Segoe UI", 20, "bold"),
            fg="#00d4ff",
            bg="#1a1f3a"
        )
        title_label.pack(pady=10)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg="#0a0e27")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left sidebar - Info Panel
        left_panel = tk.Frame(main_frame, bg="#1a1f3a", width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # User Info Section
        info_title = tk.Label(
            left_panel,
            text="👤 User Information",
            font=("Segoe UI", 12, "bold"),
            fg="#00d4ff",
            bg="#1a1f3a"
        )
        info_title.pack(pady=10, padx=10)
        
        # Name
        name_label = tk.Label(
            left_panel,
            text="Name:",
            font=("Segoe UI", 10),
            fg="#ffffff",
            bg="#1a1f3a"
        )
        name_label.pack(anchor=tk.W, padx=15, pady=5)
        
        self.name_display = tk.Label(
            left_panel,
            text=self.name if self.name else "Not set",
            font=("Segoe UI", 9, "bold"),
            fg="#00ff88",
            bg="#1a1f3a"
        )
        self.name_display.pack(anchor=tk.W, padx=15, pady=2)
        
        # City
        city_label = tk.Label(
            left_panel,
            text="City:",
            font=("Segoe UI", 10),
            fg="#ffffff",
            bg="#1a1f3a"
        )
        city_label.pack(anchor=tk.W, padx=15, pady=5)
        
        self.city_display = tk.Label(
            left_panel,
            text=self.city if self.city else "Not set",
            font=("Segoe UI", 9, "bold"),
            fg="#00ff88",
            bg="#1a1f3a"
        )
        self.city_display.pack(anchor=tk.W, padx=15, pady=2)
        
        # Status Section
        status_title = tk.Label(
            left_panel,
            text="📊 Status",
            font=("Segoe UI", 12, "bold"),
            fg="#00d4ff",
            bg="#1a1f3a"
        )
        status_title.pack(pady=(20, 10), padx=10)
        
        self.status_indicator = tk.Canvas(
            left_panel,
            width=40,
            height=40,
            bg="#1a1f3a",
            highlightthickness=0
        )
        self.status_indicator.pack(pady=10)
        
        self.status_label = tk.Label(
            left_panel,
            text="Ready",
            font=("Segoe UI", 10),
            fg="#00ff88",
            bg="#1a1f3a"
        )
        self.status_label.pack()
        
        # Time and Weather
        time_title = tk.Label(
            left_panel,
            text="🕐 Quick Info",
            font=("Segoe UI", 12, "bold"),
            fg="#00d4ff",
            bg="#1a1f3a"
        )
        time_title.pack(pady=(20, 10), padx=10)
        
        self.time_label = tk.Label(
            left_panel,
            text="",
            font=("Segoe UI", 9),
            fg="#ffffff",
            bg="#1a1f3a",
            wraplength=200
        )
        self.time_label.pack(pady=5, padx=10)
        
        self.weather_label = tk.Label(
            left_panel,
            text="",
            font=("Segoe UI", 9),
            fg="#ffffff",
            bg="#1a1f3a",
            wraplength=200
        )
        self.weather_label.pack(pady=5, padx=10)
        
        # Update time and weather
        self.update_info()
        
        # Center - Chat Area
        center_panel = tk.Frame(main_frame, bg="#0a0e27")
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Conversation title
        conv_title = tk.Label(
            center_panel,
            text="💬 Conversation",
            font=("Segoe UI", 12, "bold"),
            fg="#00d4ff",
            bg="#0a0e27"
        )
        conv_title.pack(pady=5)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            center_panel,
            width=60,
            height=25,
            bg="#1a1f3a",
            fg="#ffffff",
            font=("Consolas", 9),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Configure colors for chat
        self.chat_display.tag_config("user", foreground="#00ff88")
        self.chat_display.tag_config("jarvis", foreground="#00d4ff")
        self.chat_display.tag_config("system", foreground="#ffaa00")
        
        # Control Panel
        control_panel = tk.Frame(self.root, bg="#1a1f3a")
        control_panel.pack(fill=tk.X, padx=10, pady=10)
        
        button_frame = tk.Frame(control_panel, bg="#1a1f3a")
        button_frame.pack(fill=tk.X)
        
        # Start Listening Button
        start_btn = tk.Button(
            button_frame,
            text="🎤 Start Listening",
            command=self.start_listening,
            bg="#00d4ff",
            fg="#0a0e27",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        start_btn.pack(side=tk.LEFT, padx=5)
        
        # Stop Button
        stop_btn = tk.Button(
            button_frame,
            text="⏹ Stop",
            command=self.stop_jarvis,
            bg="#ff3333",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear Chat Button
        clear_btn = tk.Button(
            button_frame,
            text="🗑 Clear Chat",
            command=self.clear_chat,
            bg="#9933ff",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings Button
        settings_btn = tk.Button(
            button_frame,
            text="⚙ Settings",
            command=self.open_settings,
            bg="#33cc33",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        settings_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar at bottom
        status_frame = tk.Frame(self.root, bg="#1a1f3a", height=30)
        status_frame.pack(fill=tk.X, padx=0, pady=0)
        
        self.bottom_status = tk.Label(
            status_frame,
            text="Ready to assist...",
            font=("Segoe UI", 9),
            fg="#00d4ff",
            bg="#1a1f3a"
        )
        self.bottom_status.pack(pady=5)
    
    def log_message(self, sender, message):
        """Add message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        if sender == "user":
            self.chat_display.insert(tk.END, f"You: ", "user")
            self.chat_display.insert(tk.END, f"{message}\n\n")
        elif sender == "jarvis":
            self.chat_display.insert(tk.END, f"Jarvis: ", "jarvis")
            self.chat_display.insert(tk.END, f"{message}\n\n")
        else:
            self.chat_display.insert(tk.END, f"[{sender}] ", "system")
            self.chat_display.insert(tk.END, f"{message}\n\n")
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def update_status(self, status, indicator_color="green"):
        """Update status indicators"""
        self.status_label.config(text=status)
        self.bottom_status.config(text=status)
        
        # Update indicator circle
        self.status_indicator.delete("all")
        color_map = {
            "listening": "#ff6600",
            "processing": "#ffaa00",
            "speaking": "#00ff88",
            "ready": "#00d4ff",
            "error": "#ff3333"
        }
        color = color_map.get(indicator_color, indicator_color)
        self.status_indicator.create_oval(5, 5, 35, 35, fill=color, outline=color)
    
    def save_name(self, name):
        with open("username.txt", "w") as f:
            f.write(name)
    
    def load_name(self):
        try:
            with open("username.txt", "r") as f:
                return f.read().strip()
        except:
            return ""
    
    def save_city(self, city):
        with open("cityname.txt", "w") as c:
            c.write(city)
    
    def load_city(self):
        try:
            with open("cityname.txt", "r") as c:
                return c.read().strip()
        except:
            return ""
    
    def speak(self, command):
        """Convert text to speech"""
        try:
            engine = pyttsx3.init()
            engine.say(command)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
    
    def update_info(self):
        """Update time and weather information"""
        # Update time
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.time_label.config(text=f"Time: {current_time}")
        
        # Update weather if city is set
        if self.city:
            try:
                url = f"http://wttr.in/{self.city}?format=j1"
                response = requests.get(url, timeout=2).json()
                temp = response["current_condition"][0]["temp_C"]
                self.weather_label.config(text=f"Temp in {self.city}: {temp}°C")
            except:
                self.weather_label.config(text="Weather: Unable to fetch")
        
        # Schedule next update
        self.root.after(60000, self.update_info)
    
    def activate_jarvis(self):
        """Wait for 'jarvis' activation word"""
        self.update_status("Waiting for activation...", "listening")
        self.log_message("system", "Say 'Jarvis' to activate...")
        
        while True:
            try:
                with sr.Microphone() as source:
                    self.r.adjust_for_ambient_noise(source)
                    audio = self.r.listen(source, timeout=5, phrase_time_limit=3)
                    com = self.r.recognize_google(audio)
                    
                    if "jarvis" in com.lower():
                        self.speak("jarvis activated")
                        self.log_message("system", "✓ Activated!")
                        self.update_status("Activated", "speaking")
                        if self.name:
                            greeting = f"welcome back {self.name}. How can I assist you today?"
                            self.speak(greeting)
                            self.log_message("jarvis", greeting)
                        else:
                            greeting = "welcome back. How can I assist you today?"
                            self.speak(greeting)
                            self.log_message("jarvis", greeting)
                        break
                    else:
                        self.speak("please say jarvis to activate")
            except Exception as e:
                self.speak("sorry try again")
    
    def process_command(self, text):
        """Process voice commands"""
        text_lower = text.lower()
        
        # Exit
        if text_lower in ["exit", "quit", "bye"]:
            farewell = f"See you again {self.name} with full energy" if self.name else "See you again with full energy"
            self.speak(farewell)
            self.log_message("jarvis", farewell)
            return "exit"
        
        # Your name?
        elif "your name" in text_lower:
            response = "My name is Jarvis, and what's your name?"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Designer?
        elif "who designed you" in text_lower or "designed you" in text_lower:
            response = "I was designed by the group of 7 members of class 11 students"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Open Google
        elif "open google" in text_lower:
            webbrowser.open("www.google.com")
            response = "opening google"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Open YouTube
        elif "open youtube" in text_lower:
            webbrowser.open("www.youtube.com")
            response = "opening youtube"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Emotional song
        elif "emotional song" in text_lower or "emotional" in text_lower:
            response = "enjoy emotional music..."
            self.speak(response)
            self.log_message("jarvis", response)
            self.play_music()
        
        # Stop music
        elif "stop music" in text_lower or "stop song" in text_lower:
            if self.driver:
                self.driver.quit()
                self.driver = None
                response = "Music stopped"
            else:
                response = "No music is playing"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Age
        elif "your age" in text_lower:
            response = "I don't have a specific age. Just wish me happy birthday anytime!"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # City
        elif "i am from" in text_lower:
            city = text_lower.split()[-1]
            self.save_city(city)
            self.city = city
            self.city_display.config(text=city)
            response = f"woow {city} is wonderful place"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Time
        elif "time" in text_lower and "now" in text_lower:
            now = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The time is {now}"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Date
        elif "date" in text_lower and "today" in text_lower:
            today = datetime.date.today().strftime("%B %d, %Y")
            response = f"Today's date is {today}"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Temperature
        elif "temperature" in text_lower:
            try:
                if self.city:
                    url = f"http://wttr.in/{self.city}?format=j1"
                    response_data = requests.get(url).json()
                    temp = response_data["current_condition"][0]["temp_C"]
                    response = f"The current temperature in {self.city} is {temp} degrees Celsius"
                    self.speak(response)
                    self.log_message("jarvis", response)
                else:
                    response = "Please tell me your city first"
                    self.speak(response)
                    self.log_message("jarvis", response)
            except Exception as e:
                response = "Sorry, I cannot fetch the temperature right now."
                self.speak(response)
                self.log_message("jarvis", response)
        
        # My name
        elif "my name is" in text_lower:
            name = text_lower.split()[-1]
            self.save_name(name)
            self.name = name
            self.name_display.config(text=name)
            response = f"Nice to meet you, {name}"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Know my name
        elif "do you know my name" in text_lower:
            if self.name:
                response = f"Yes, your name is {self.name} do you need some extra help"
            else:
                response = "You have not told me your name yet"
            self.speak(response)
            self.log_message("jarvis", response)
        
        # Default - Use OpenAI
        else:
            try:
                self.update_status("Processing...", "processing")
                response_obj = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a personal assistant named Jarvis like Alexa and Gemini. You are also like a humanoid robot, so answer the question clearly. Store the data you recently said when the user asked the questions based on the recently answered you can reply smoothly. Sometimes like MCQ-wise questions give very short answers only, no need of explanation, and sometimes according to situation. At last for every response say if there is any specific topic, then I am always here to help you."},
                        {"role": "user", "content": text}
                    ]
                )
                answer = response_obj.choices[0].message.content
                self.speak(answer)
                self.log_message("jarvis", answer)
                self.update_status("Ready", "ready")
            except Exception as e:
                print(f"Error: {e}")
                response = "try again"
                self.speak(response)
                self.log_message("jarvis", response)
                self.update_status("Error - Try again", "error")
    
    def play_music(self):
        """Play music from YouTube"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--start-maximized")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.get("https://youtu.be/Sc1OI1i-Kgs")
            
            time.sleep(5)
            
            try:
                play_button = self.driver.find_element(By.CLASS_NAME, "ytp-large-play-button")
                play_button.click()
                self.log_message("system", "Music is now playing")
            except:
                self.log_message("system", "Please click play manually")
        except Exception as e:
            self.log_message("system", f"Error playing music: {str(e)}")
    
    def start_listening(self):
        """Start listening in a separate thread"""
        thread = threading.Thread(target=self.listen_loop, daemon=True)
        thread.start()
    
    def listen_loop(self):
        """Main listening loop"""
        self.is_listening = True
        self.update_status("Activating...", "listening")
        self.activate_jarvis()
        
        if not self.is_listening:
            return
        
        self.log_message("system", "=== Listening started ===")
        
        while self.is_listening:
            try:
                with sr.Microphone() as source:
                    self.r.adjust_for_ambient_noise(source)
                    self.update_status("Listening...", "listening")
                    self.root.update()
                    
                    audio = self.r.listen(source, timeout=4, phrase_time_limit=7)
                    word = self.r.recognize_google(audio)
                    
                    self.log_message("user", word)
                    self.update_status("Processing...", "processing")
                    
                    result = self.process_command(word)
                    if result == "exit":
                        self.is_listening = False
                        break
                    
                    self.update_status("Ready", "ready")
                    
            except sr.WaitTimeoutError:
                self.speak("No speech detected, trying again")
                self.update_status("Timeout - Retrying", "listening")
            except sr.UnknownValueError:
                self.speak("Couldn't understand, try again")
                self.update_status("Can't understand - Retrying", "listening")
            except sr.RequestError:
                self.speak("Internet problem, try again")
                self.update_status("Internet Error - Retrying", "error")
            except Exception as e:
                print(f"Error: {e}")
                self.log_message("system", f"Error: {str(e)}")
    
    def stop_jarvis(self):
        """Stop listening"""
        self.is_listening = False
        self.update_status("Stopped", "ready")
        self.log_message("system", "=== Listening stopped ===")
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def open_settings(self):
        """Open settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x250")
        settings_window.configure(bg="#1a1f3a")
        
        # Name setting
        tk.Label(settings_window, text="Your Name:", font=("Segoe UI", 10), fg="#ffffff", bg="#1a1f3a").pack(pady=5)
        name_entry = tk.Entry(settings_window, font=("Segoe UI", 10), width=30)
        name_entry.insert(0, self.name)
        name_entry.pack(pady=5)
        
        # City setting
        tk.Label(settings_window, text="Your City:", font=("Segoe UI", 10), fg="#ffffff", bg="#1a1f3a").pack(pady=5)
        city_entry = tk.Entry(settings_window, font=("Segoe UI", 10), width=30)
        city_entry.insert(0, self.city)
        city_entry.pack(pady=5)
        
        # API Key setting
        tk.Label(settings_window, text="OpenAI API Key:", font=("Segoe UI", 10), fg="#ffffff", bg="#1a1f3a").pack(pady=5)
        api_entry = tk.Entry(settings_window, font=("Segoe UI", 10), width=30, show="*")
        api_entry.pack(pady=5)
        
        def save_settings():
            name = name_entry.get()
            city = city_entry.get()
            api_key = api_entry.get()
            
            if name:
                self.save_name(name)
                self.name = name
                self.name_display.config(text=name)
            
            if city:
                self.save_city(city)
                self.city = city
                self.city_display.config(text=city)
            
            if api_key:
                self.client = OpenAI(api_key=api_key)
            
            messagebox.showinfo("Success", "Settings saved!")
            settings_window.destroy()
        
        save_btn = tk.Button(
            settings_window,
            text="Save Settings",
            command=save_settings,
            bg="#00d4ff",
            fg="#0a0e27",
            font=("Segoe UI", 10, "bold"),
            padx=20,
            pady=10
        )
        save_btn.pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = JarvisUI(root)
    root.mainloop()
