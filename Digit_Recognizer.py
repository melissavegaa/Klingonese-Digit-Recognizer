"""
--------------------------------------------------------------------------
        Class: CS 582: Intro to Speech Processing
      Purpose: In this homework, you will implement the key HMM algorithms in Python
 Instructions:
     1. Write Python code as indicated
     2. Turn in this template with working code for each exercise.
     3. Turn in a sample run to demonstrate your work

        Title: Question 5
          Due: 5.10.2023
         Name: Melissa Vega
 Turn-in Date:
    File Name: Q5_<YOUR_LASTNAME>_<LAST_4_DIGITS_OF_YOUR_STUDENT_ID>.py
--------------------------------------------------------------------------
"""
from speech_recogn.HMMSpeechRecog import HMMSpeechRecog

model = HMMSpeechRecog(add_mfcc_delta_delta=False)
model.train(3, 2)
model.test('/Users/pug17/Question5_Template/Homework3/models/accuracies/klingonese_hmm.txt')

model.pickle('/Users/pug17/Question5_Template/Homework3/models/klingonese_hmm.pkl')
model2 = HMMSpeechRecog.unpickle('/Users/pug17/Question5_Template/Homework3/models/klingonese_hmm.pkl')

predicted_labels0 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/pagh'])
print(f'predicted label {predicted_labels0}')

predicted_labels1 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/.wa'])
print(f'predicted label {predicted_labels1}')

predicted_labels2 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/cha'])
print(f'predicted label {predicted_labels2}')

predicted_labels3 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/wej'])
print(f'predicted label {predicted_labels3}')

predicted_labels4 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/loS'])
print(f'predicted label {predicted_labels4}')

predicted_labels5 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/vagh'])
print(f'predicted label {predicted_labels5}')

predicted_labels6 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/jav'])
print(f'predicted label {predicted_labels6}')

predicted_labels7 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/Soch'])
print(f'predicted label {predicted_labels7}')

predicted_labels8 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/chorgh'])
print(f'predicted label {predicted_labels8}')

predicted_labels9 = model2.predict_files(['/Users/pug17/Question5_Template/Homework3/data/test_data/Hut'])
print(f'predicted label {predicted_labels9}')

model2.calc_mean_entropy()
