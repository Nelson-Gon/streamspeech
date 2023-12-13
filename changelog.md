# Changes to streamspeech
* Version: 2023.12.08
- Temporarily disabled GAN based TTS, instead we now use system TTS functions which means we only support Linux and OSX 
- Initial documentation for key functionality 
- Now using classes #4 #6 
- Better error handling e.g. in cases where no results are returned or no api key is present.  
* Version: 2023.12.06
- Fixed issue with installs not installing dependencies as well
- Removed files not needed for the app 
* Version: 2023.12.03
- Now runnable as a module 
- Changed the page title and icon 
- Now using `joblib` to speed up speech and text generation 
- Added a spinner to partially solve the issue of waiting for results 
- Fixed issue with encoders and decoders only outputtting a single response 
- Initial version supporting streamlit based Text to Speech 