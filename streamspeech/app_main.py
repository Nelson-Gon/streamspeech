import streamlit as st 
from speechbrain.pretrained import HIFIGAN, Tacotron2 
import google.generativeai as genai 
from subprocess import call, Popen, PIPE 
import signal 
import subprocess
from platform import system 
import os 
import numpy as np 
from joblib import Parallel, delayed 
from PIL import Image 


def system_tts():
    use_command = {"darwin": "say", "linux":"espeak"}
    return use_command[system().lower()]

class StreamSpeech:
    """
    StreamSpeech class definition, initializes the necessary models for speech generation 
    
    """
    def __init__(self):
        """
        Initialize key StreamSpeech models 
        """
        self.configure_gemini()        
        self.chosen_model = self.choose_model()
        self.model = genai.GenerativeModel(self.chosen_model)
    
    def choose_model(self):
        model_choice = st.selectbox("Choose a model", ("gemini-pro", "gemini-pro-vision"),
                                    index = 0)
        return model_choice
      

    def get_model(self, hifigan_source = "speechbrain/tts-hifigan-ljspeech", 
                tacotron_source = "speechbrain/tts-hifigan-ljspeech"):
        """Fetches the necessary HIFIGAN and TACOTRON models 

        Args:
            hifigan_source (str, optional): HIFIGAN model for mel spectrogram generation. Defaults to "speechbrain/tts-hifigan-ljspeech".
            tacotron_source (str, optional): TACOTRON model used for decoding generated mel-spectogram. Defaults to "speechbrain/tts-hifigan-ljspeech".

        Returns:
            tuple: Returns a HIFIGAN encoder and a TACOTRON decoder 
        """
        self.hifigan_source = hifigan_source 
        self.tacotron_source = tacotron_source 
        hifi_gan = HIFIGAN.from_hparams(source=self.hifigan_source, savedir="tmpdir_vocoder")
        tacotron2 = Tacotron2.from_hparams(source=self.tacotron_source, savedir="tmpdir_tts")
        return hifi_gan, tacotron2

    
    def configure_gemini(self):
        """Sets the session API key for use with the text generator 
        """
        try:
            genai.configure(api_key = os.environ["google_key"])
        except KeyError:
            st.error("This app requires an API Key named 'google_key' in your environment. Get one at https://developers.generativeai.google/tutorials/setup", icon=":warn:")
        
    def get_text(self):
        """Get user input for user in the text generation model 

        Returns:
            streamlit.text_area: Streamlit text UI 
        """
        return st.text_area(label="Type Here", value="Tell me about yourself in one sentence")
    
    def upload_image(self):
        image_file = st.file_uploader("Choose an image file")
        return image_file
    
    def prompt_gemini(self,user_prompt):
        """Send user input to the Gemini API 

        Args:
            user_prompt (str): User text e.g. a question to ask gemini

        Returns:
            str: Gemini's response 
        """
        response = self.model.generate_content(contents = user_prompt)
        return response.text
    
    def text_to_speech(self,gemini_response):
        """Generate audio from text 
        Args:
            gemini_response (str): Results from gemini prompting 

        Returns:
            np.array: A numpy representation of the generated audio 
        """
        output_, _, _ = self.tacotron2.encode_text(gemini_response)
        waveforms = self.hifi_gan.decode_batch(output_)
        return waveforms 
    

    
    def get_waveform(self,res):
        """Convert text to a mel spectrogram and finally to a waveform 

        Args:
            res (str): Text to generate speech for 

        Returns:
            np.array: NumPy representation of the generated mel-spectogram 
        """
        mel_output, mel_length, alignment = self.tacotron2.encode_text(res)
        return self.hifi_gan.decode_batch(mel_output).squeeze(1)
    
    def merge_audio(self,audio_list):
        """Merge a list of numpy arrays 

        Args:
            audio_list (list): A list of numpy arrays 

        Returns:
            np.array: Numpy array representing the merged audio 
        """
        return np.column_stack(audio_list)
    
    def process_audio(self,res):
        """Generate speech for multiple or single prompts 

        Args:
            res (str): Response from gemini or any form of text 

        Returns:
            np.array: Merged array of audio 
        """
        audios = Parallel(n_jobs = 4, backend="threading")(delayed(self.system_tts)(x) for x in res if x)
        return self.merge_audio(audios)
    
    
    def play_audio(self,tts_response,**kwargs):
        """_summary_

        Args:
            tts_response (np.array): Numpy audio representation 

        Returns:
            streamlit.audio: Audio output for the user 
        """
        audio_ = tts_response  
        return st.audio(audio_, **kwargs)

def main():
    st.set_page_config(page_title="streamspeech", page_icon=f":brain:")
    Processor = StreamSpeech()
    with st.spinner("Processing your input, please wait...."):
        try:
            prompt = Processor.get_text()
            if "vision" in Processor.chosen_model:
                image_input = Processor.upload_image()
                image_file = Image.open(image_input)
                prompt = [prompt, image_file]
 
            gemini_res = Processor.prompt_gemini(prompt)
        except TypeError as err:
            st.error(err)
        except AttributeError as err:
            st.error(err)
        else:
            # TODO: Restore GAN based TTS 
            # sample_rate = st.slider("Sample Rate", min_value = 16000, max_value = 40000, value = 22050, step = 20)
            # Processor.play_audio(audio_res, sample_rate = sample_rate)
            col1, col2 = st.columns(2)
            col1.write(gemini_res)
            if "vision" in Processor.chosen_model:
                col2.image(prompt[1])
            sys_tts = system_tts()
            subprocess.Popen([sys_tts, gemini_res])
            if st.button("Stop audio"):
                os.system(f"pkill {sys_tts}")
            


if __name__=="__main__":
    main()




