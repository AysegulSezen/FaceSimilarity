The human brain can distinguish whether the person in the picture and the person it sees are the same person or not, and identity verification is done this way in airlines, banks, post office etc. We can do the same with machine learning.

After doing the following work, I learned that there is a ready-made package for this process (https://github.com/ageitgey/face_recognition). Using the face_recognition package in the FaceRecognitionPackage.py code; There is a result of guessing whether the person in the video is the same as the person in the picture. The accuracy rate is as good as 99%.

I tried to write a similar process with the formulas of machine learning algorithms. You can find this in FaceSimilarity.py. With approximately 40 photos of the same person, you can guess whether the photo you will show later belongs to the same person. The accuracy rate is low because I don't use filters.

I also tried with Keras. CNN techniques and filters were used, but the accuracy rate is low, too. File is FaceSimilarityByKeras.py
