#Speech-to-Text code used in main code for Sign Language Translator

# importing outside library
import speech_recognition as sr


# function that returns a dictionary
# first two items tell us if there were any errors (API/speech recognizing)
# last item tells us what the user has said
def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    #
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response

#Function that returns what a person said as a string
def GetText():
    # creating recognizer and microphone instances
    # used by recognize_speech_from_mic() function
    # done by using outside  library that was imported at the beginning
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Getting information from the microphone as a dictionary
    # first two items tell us if there were any errors (API/recognizing)
    # last item tells us what the user has said
    speech = recognize_speech_from_mic(recognizer, microphone)

    # getting said phrase as a string
    transcribedText = speech['transcription']

    # returning said phrase
    return transcribedText


if __name__ == "__main__":
    print(GetText())
     # create recognizer and mic instances
