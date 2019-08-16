# Voice Control Project
This project uses the Google Voice Kit to log diet information for food allergy and obesity patients. 

# Prerequisites
Java 8 ([Download here](https://www.oracle.com/technetwork/java/javase/downloads/java-archive-javase8-2177648.html)).

- After downloading Java 8, set the 'JAVAHOME' variable in food.py to your own path of jdk/bin/java.

NLTK ([Download here](https://www.nltk.org/install.html)).

Stanford NER ([Download here](https://nlp.stanford.edu/software/CRF-NER.html)).
- After downloading Stanford NER, set the classification model path for StanfordNERTagger to your own path of the classification model. Also, set the Stanford tagged jar file to your own path of the Stanford tagged jar file. 

- If you would like to use the same classiciation model filepath as in food.py ("/home/pi/AIY-voice-kit-python/src/examples/voice/app/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz") and the same Stanford tagged jar filepath as in food.py ("/home/pi/AIY-voice-kit-python/src/examples/voice/app/stanford-ner-2018-10-16/stanford-ner.jar"), then create a directory called "app" in "/home/pi/AIY-voice-kit-python/src/examples/voice" and unzip the Stanford NER zipped file in the "app" directory. 

Gender-guesser ([Download here](https://test.pypi.org/project/gender-guesser/)).

# Run
python3 food.py
