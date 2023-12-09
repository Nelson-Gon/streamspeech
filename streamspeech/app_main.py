import streamlit as st 
from speechbrain.pretrained import HIFIGAN, Tacotron2 
import google.generativeai as palm 
import os 
import numpy as np 
from joblib import Parallel, delayed 

class StreamSpeech:
    """
    StreamSpeech class definition, initializes the necessary models for speech generation 
    
    """
    def __init__(self):
        """
        Initialize key StreamSpeech models 
        """
        self.hifi_gan, self.tacotron2 = self.get_model()

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
    
    def configure_palm(self):
        """Sets the session API key for use with the text generator 
        """
        try:
            palm.configure(api_key = os.environ["google_key"])
        except KeyError:
            st.error("This app requires an API Key named 'google_key' in your environment. Get one at https://developers.generativeai.google/tutorials/setup", icon=":warn:")
        
    def get_text(self):
        """Get user input for user in the text generation model 

        Returns:
            streamlit.text_area: Streamlit text UI 
        """
        return st.text_area(label="Type Here", value="Ask me something")
    
    def prompt_palm(self,user_prompt):
        """Send user input to the PaLM API 

        Args:
            user_prompt (str): User text e.g. a question to ask PaLM

        Returns:
            str: PaLM's response 
        """
        response = palm.generate_text(prompt= user_prompt).result
        return response
    
    def text_to_speech(self,palm_response):
        """Generate audio from text 
        Args:
            palm_response (str): Results from PaLM prompting 

        Returns:
            np.array: A numpy representation of the generated audio 
        """
        output_, _, _ = self.tacotron2.encode_text(palm_response)
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
            res (str): Response from PaLM or any form of text 

        Returns:
            np.array: Merged array of audio 
        """
        audios = Parallel(n_jobs = 4, backend="threading")(delayed(self.get_waveform)(x) for x in res if x)
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
    Processor = StreamSpeech()
    st.set_page_config(page_title="streamspeech", page_icon=f":brain:")
    Processor.configure_palm()
    with st.spinner("Processing your input, please wait...."):
        try:
            audio_res = Processor.process_audio(Processor.prompt_palm(Processor.get_text()).split("\n"))
        except AttributeError:
            st.error("No results generated....please try a different search")
        else:
            sample_rate = st.slider("Sample Rate", min_value = 16000, max_value = 40000, value = 22050, step = 20)
            Processor.play_audio(audio_res, sample_rate = sample_rate)

if __name__=="__main__":
    main()




