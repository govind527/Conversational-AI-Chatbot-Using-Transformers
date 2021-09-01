import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import grpc
import queue
import argparse
import jarvis_api.audio_pb2 as ja
import jarvis_api.jarvis_asr_pb2 as jasr
import jarvis_api.jarvis_asr_pb2_grpc as jasr_srv
import jarvis_api.jarvis_tts_pb2_grpc as jtts_srv
from curtsies.fmtfuncs  import red, bold, green, on_blue, yellow, blue, cyan
import pyaudio
import time
import jarvis_api.jarvis_tts_pb2 as jtts
import wave

RATE=44100
CHUNK= int(RATE /10) #100ms

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large").to("cuda")


def get_args():
    parser=argparse.ArgumentParser(description="Stemming transcription via Jarvis AI services")
    parser.add_argument("--server",default="localhost:50051", type=str,help="URI to GRPC server server endpoint")
    parser.add_argument("--input-device", type=int, default= None,help="output device to use")
    parser.add_argument("--list-devices",action="store_true",help="lsit output devices indices")
    parser.add_argument("--voice",type=str,help="voice name to use",default="ljspeech")
    return parser.parse_args()

class MicrophoneStream(object):
    def __init__(self,rate,chunk,device=None):
        self._rate=rate
        self._chunk=chunk
        self._device=device

        self._buff=queue.Queue()
        self.closed=True
    def __enter__(self):
        self._audio_interface=pyaudio.PyAudio()
        self._audio_stream=self._audio_interface.open(
            format=pyaudio.paInt16,
            input_device_index=self._device,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed=False
        return self
    
    def __exit__(self,type,value,traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed=True

        self._buff.put(None)
        self._audio_interface.terminate()
    
    def _fill_buffer(self,in_data,frame_count,time_info,status_flags):
        self._buff.put(in_data)
        return None,pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk=self._buff.get()
            if chunk is None:
                return 
            data=[chunk]
            while True:
                try:
                    chunk=self._buff.get(block=False)
                    if chunk is None:
                        return 
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)


def listen_print_loop(responses):
    step=0
    num_chars_printed=0
    channel=grpc.insecure_channel("localhost:50051")


    tts_client=jtts_srv.JarvisTTSStub(channel)
    audio_handle=pyaudio.PyAudio()
    req=jtts.SynthesizeSpeechRequest()
    req.text= "Hello"
    req.language_code="en-US"
    req.encoding=ja.AudioEncoding.LINEAR_PCM
    req.sample_rate_hz= 22050
    req.voice_name= 'ljspeech'



    stream=audio_handle.open(format=pyaudio.paFloat32,channels=1,rate=22050,output=True)
    for response in responses:
        if not response.results:
            continue
        result=response.results[0]
        if not result.alternatives:
            continue

        transcript=result.alternatives[0].transcript

        overwrite_chars=' '*(num_chars_printed-len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript+overwrite_chars+'\r')
            sys.stdout.flush()

            num_chars_printed=len(transcript)
        else:

            mic_in=str(transcript + overwrite_chars)
            print(green(mic_in))
            num_chars_printed=0

            new_user_input_ids = tokenizer.encode(mic_in+ tokenizer.eos_token, return_tensors='pt').to("cuda")

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

            # pretty print last output tokens from bot
            output="DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
            print(red(output))
            print()

            tts_out=str(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
            req.text=tts_out

            print(yellow("Generating audio for request..."))
            start=time.time()
            resp=tts_client.Synthesize(req)
            stop=time.time()

            stream.write(resp.audio)
    stream.stop_stram()
    stream.close()
            


        
def main():
    args=get_args()
    if args.list_devices:
        p=pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info=p.get_device_info_by_index(i)
            if info['maxInputChannels']<1:
                continue
            print(f"{info['index']}: {info['name']}")
        sys.exit(0)
    channel=grpc.insecure_channel(args.server)
    client=jasr_srv.JarvisASRStub(channel)

    # tts_client=jtts_srv.JarvisTTSStub(channel)
    # audio_handle=pyaudio.PyAudio()

    # config=jasr.RecongnitionConfig(
    #     encoding=ja.AudioEncoding.LINEAR_PCM,
    #     sample_rate_hertz=RATE,
    #     language_code="en-US",
    #     max_alternatives=1,
    #     enable_automatic_punctuation=True,
    # )
    # streaming_config=jasr.StreamingRecognitionConfig(config=config,interim_results=True)
    
    # with MicrophoneStream(RATE, CHUNK, device=args.input_device) as stream:
    #     audio_generator=stream.generator()
    #     requests=(jasr.StreamRecognizeRequest(audio_content=content) for content in audio_generator) 

    #     def build_generator(cfg, gen):
    #         yield jasr.StreamRecognizeRequest(streaming_config=cfg)
    #         for x in gen:
    #             yield x
    #     responses=client.StreamingRecognize(build_generator(streaming_config,requests))
    #     listen_print_loop(responses)

        

        



