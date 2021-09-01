import os
import torch
import playsound
from gtts import gTTS
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer

class Bot():
    def __init__(self):
        self.r = sr.Recognizer()
        self.temp_file = "output.mp3"
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    def _get_audio(self):
        with sr.Microphone() as source:
            audio = self.r.listen(source)
            inp = ""
            try:
                inp = self.r.recognize_google(audio)
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("service is down")
        return inp

    def _speak(self, text):
        tts = gTTS(text=text, lang='en', tld='co.uk')
        tts.save(self.temp_file)
        playsound.playsound(self.temp_file)
        os.remove(self.temp_file)

    def _reply(self, inp):
        new_user_input_ids = self.tokenizer.encode(inp + self.tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = new_user_input_ids
        self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        ret = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return ret

    def chat(self, use_mic, speak):
        greeting = "Hi Human!"
        print(f"bot: {greeting}")
        if speak:
            self._speak(greeting)
        while True:
            if use_mic:
                inp = self._get_audio()
                print(f"you: {inp}")
            else:
                inp = input("you: ")
            if "shutdown" in inp.lower():
                break
            ret = self._reply(inp)
            print(f"bot: {ret}")
            if speak:
                self._speak(ret)

if __name__ == "__main__":
    print("booting up...")
    bot= Bot()
    bot.chat(use_mic=True, speak=True)