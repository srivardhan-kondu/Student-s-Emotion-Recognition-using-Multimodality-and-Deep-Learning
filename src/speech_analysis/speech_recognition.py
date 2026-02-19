"""
Automatic Speech Recognition Module (FR9).
Converts speech to text using Whisper or Google Speech API.
"""

import speech_recognition as sr
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    Automatic Speech Recognition for converting speech to text.
    """
    
    def __init__(self, method: str = 'google'):
        """
        Initialize speech recognizer.
        
        Args:
            method: Recognition method ('google', 'whisper', 'sphinx')
        """
        self.method = method
        self.recognizer = sr.Recognizer()
        logger.info(f"Initialized {method} speech recognizer")
    
    def recognize_from_file(self, audio_path: str, language: str = 'en-US') -> Optional[str]:
        """
        Recognize speech from audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Transcribed text or None if recognition fails
        """
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                
                if self.method == 'google':
                    text = self.recognizer.recognize_google(audio_data, language=language)
                elif self.method == 'sphinx':
                    text = self.recognizer.recognize_sphinx(audio_data)
                else:
                    raise ValueError(f"Unsupported method: {self.method}")
                
                logger.info(f"Transcribed: {text}")
                return text
                
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from service: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during speech recognition: {e}")
            return None
    
    def recognize_from_microphone(self, duration: int = 5, language: str = 'en-US') -> Optional[str]:
        """
        Recognize speech from microphone.
        
        Args:
            duration: Recording duration in seconds
            language: Language code
            
        Returns:
            Transcribed text or None if recognition fails
        """
        try:
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                logger.info(f"Recording for {duration} seconds...")
                audio_data = self.recognizer.listen(source, timeout=duration)
                
                logger.info("Processing audio...")
                if self.method == 'google':
                    text = self.recognizer.recognize_google(audio_data, language=language)
                elif self.method == 'sphinx':
                    text = self.recognizer.recognize_sphinx(audio_data)
                else:
                    raise ValueError(f"Unsupported method: {self.method}")
                
                logger.info(f"Transcribed: {text}")
                return text
                
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out")
            return None
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from service: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during speech recognition: {e}")
            return None


class WhisperRecognizer:
    """
    Whisper-based speech recognition (more accurate).
    """
    
    def __init__(self, model_size: str = 'base'):
        """
        Initialize Whisper recognizer.
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        try:
            import whisper
            self.model = whisper.load_model(model_size)
            self.model_size = model_size
            logger.info(f"Loaded Whisper {model_size} model")
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            raise
    
    def recognize_from_file(self, audio_path: str, language: str = 'en') -> Optional[str]:
        """
        Recognize speech from audio file using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Transcribed text or None if recognition fails
        """
        try:
            result = self.model.transcribe(audio_path, language=language)
            text = result['text'].strip()
            logger.info(f"Transcribed: {text}")
            return text
        except Exception as e:
            logger.error(f"Error during Whisper recognition: {e}")
            return None
