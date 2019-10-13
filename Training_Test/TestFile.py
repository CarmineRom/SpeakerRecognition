import csv
import h5py
import scipy
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from cv2 import *
from moviepy.editor import *
import rcca
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


def extract_video_features():
    """
    Executes Openface and extracts face features
    :return:
        face features as dictionary
    """

    # Face feature extraction from Openface output file
    file = open("Extracted_Features/"+input_video[:len(input_video)-4]+"_Features/"+input_video[:len(input_video)-4]+".csv")
    reader = csv.DictReader(file)
    features = {}

    for row in reader:

        # Taking only good frames where faces have been detected with a confidence higher than 0.8 (Openface standard)
        if int(row[' success']) == 1 and float(row[' confidence']) > 0.5:
            face_id = int(row[' face_id'])
            frame = int(row['frame']) - 1

            features.setdefault(frame, {})
            face_features = []

            # Mouth LandMarks
            for i in range(0, 68):
                face_features.append(float(row[' x_' + str(i)]))

            for i in range(0, 68):
                face_features.append(float(row[' y_' + str(i)]))

            if f_type == "AU":
                au = ["10", "12", "14", "15", "17", "20", "23", "25", "26"]
                for i in au:
                    face_features.append(float(row[' AU' + i + '_r']))

            features[frame][face_id] = face_features

    return features


def extract_audio_features():

    # Audio extraction with ffmpeg and scipy
    # print("Extracting audio from video...")
    os.system("ffmpeg -i Dataset/" + input_video + " -loglevel panic -ac 1 -vn -y\
                                Extracted_Features/" + input_video[:len(input_video) - 4] + "_Features/audio.wav")

    (rate, sig) = wav.read("Extracted_Features/" + input_video[:len(input_video) - 4] + "_Features/audio.wav")

    # MFCC extraction
    mfcc_feat = mfcc(sig, rate, winstep=1/fps, numcep=12)

    return mfcc_feat


def get_face_features(frame, face):
    """
    Get features of a specified face in a specified frame as distances beetween mouth landmarks couples.
    :param frame: index of the frame
    :param face: index of the face
    :return: face features as array of landmarks distances
    """

    import math

    def distance(p1, p2):
        """
        Calculate euclidean distance between two points
        """
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    if f_type == "LD":
        distances = []
        for couple in [(50, 58), (61, 67), (51, 57), (62, 66), (52, 56), (63, 65), (48, 54),
                       (60, 64), (49, 59), (53, 55)]:
            a_indexes = (couple[0], couple[0] + 68)
            b_indexes = (couple[1], couple[1] + 68)

            a = (video_features[frame][face][a_indexes[0]], video_features[frame][face][a_indexes[1]])

            b = (video_features[frame][face][b_indexes[0]], video_features[frame][face][b_indexes[1]])

            distances.append(distance(a, b))
        return distances
    else:
        return video_features[frame][face][136:]


def get_speakers_labels(video, face_features):
    import math

    def calc_centroid(arr):
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x / length, sum_y / length

    f = scipy.io.loadmat("Dataset/mouthMotion" + str(int(video[:-4])) + ".mat")

    motion = f['mouthMotion'][0]
    speakers_dataset = {}

    for face in range(len(motion)):
        speakers_dataset.setdefault(face, {})
        for frame in range(num_frames):
            speakers_dataset[face].setdefault(frame, {})
            if motion[face][0][frame][0] == 1:
                speakers_dataset[face][frame] = 1
            else:
                speakers_dataset[face][frame] = 0

    h5py_file = True
    try:
        f = h5py.File("Dataset/final" + str(int(video[:-4])) + ".mat", 'r')
    except:
        f = scipy.io.loadmat("Dataset/final" + str(int(video[:-4])) + ".mat")
        h5py_file = False

    final = f['final']

    # For Debug
    corrected_speakers = {}
    corrections = {id: None for id in range(len(speakers_dataset.keys()))}
    # /////////////////

    for id_face in list(speakers_dataset.keys()):
        for frame in list(speakers_dataset[id_face].keys()):
            if face_features.get(frame) is not None:

                if h5py_file == True :
                    landm_mat = f[f[final[0, frame]]['shapes']['shape'][id_face][0]]
                    centroid_landm_mat = calc_centroid(np.array(list(zip(landm_mat[0], landm_mat[1]))))
                else:
                    centroid_landm_mat = calc_centroid(final[frame][0][0][0][0][0][id_face][0])

                mindist = 50  # Da valutare
                mindist_id = None

                for face in face_features[frame].keys():

                    landm_x_face = face_features[frame][face][:68]
                    landm_y_face = face_features[frame][face][68:136]
                    centroid_landm_face = calc_centroid(np.array(list(zip(landm_x_face, landm_y_face))))

                    dist = math.hypot(centroid_landm_face[0] - centroid_landm_mat[0], centroid_landm_face[1] - centroid_landm_mat[1])
                    if dist < mindist:
                        mindist = dist
                        mindist_id = face

                if mindist_id is not None:
                    if corrections[id_face] is not None and corrections[id_face] != mindist_id:
                        print("Overwrite faceId already detected " + str(id_face) + " with " + str(mindist_id)) #for debug
                    else:
                        corrections[id_face] = mindist_id

    for wrong_face, correct_face in corrections.items():
        corrected_speakers[correct_face] = speakers_dataset[wrong_face]

    return corrected_speakers


def detect_speakers(frames_range):
    """
    Executes Cross Modal Association between each frame video and each frame audio
    :return:
        Array of faces indexes representing the speaking face of each frame
    """
    audio_feat = preprocessing.normalize(audio_features)
    correlations = {}
    for frame in list(video_features.keys()):

        if len(video_features[frame].keys()) > 1:  # More than one face has been detected in the frame

            start_frame = 0 if frame - frames_range < 0 else frame - frames_range
            end_frame = list(video_features.keys())[-1] if frame + frames_range >= list(video_features.keys())[-1] \
                else frame + frames_range

            correlations.setdefault(frame, {})
            for face in list(video_features[frame].keys()):
                face_corrs = []
                for f in range(start_frame, end_frame+1):
                    if not f >= len(audio_feat) and video_features.get(f) is not None and video_features[f].get(face) is not None:
                        v_set = np.array(get_face_features(f, face))
                        a_set = np.array(audio_feat[f])
                        v_set = preprocessing.normalize([v_set])[0]
                        v_proj = np.dot(v_set, cca.ws[0])
                        a_proj = np.dot(a_set, cca.ws[1])
                        face_corrs.append(abs(np.corrcoef(v_proj, a_proj)[0, 1]))

                correlations[frame][face] = np.mean(face_corrs)

    speakers = [None]*num_frames

    for frame in correlations.keys():
        max_corr = 0
        max_corr_face = None
        for face, corr in correlations[frame].items():
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                max_corr_face = face

        speakers[frame] = max_corr_face

    return speakers, correlations

# # Calculate the best model (Not executable anymore)
# for f_type in ["LD", "AU"]:
#
#     if f_type is "LD":
#         models_dir = "LD_Models"
#     else:
#         models_dir = "AU_Models"
#
#     bestmodel = None
#     bestaccuracy = 0
#
#     for model in os.listdir(models_dir):
#         cca = rcca.CCA()
#         cca.load(models_dir+"/"+model)
#         videoAccuracies = []
#
#         for input_video in ["002.mp4", "006.mp4", "008.mp4", "014.mp4", "016.mp4", "018.mp4", "025.mp4",
#                                  "035.mp4", "044.mp4", "053.mp4", "056.mp4", "057.mp4", "061.mp4", "068.mp4",
#                                  "073.mp4", "075.mp4"]:
#
#             # print("Processing: ", input_video)
#             videoObj = cv2.VideoCapture("Dataset/" + input_video)
#             fps = videoObj.get(cv2.CAP_PROP_FPS)
#             num_frames = int(videoObj.get(cv2.CAP_PROP_FRAME_COUNT))
#
#             video_features = extract_video_features()
#             audio_features = extract_audio_features()
#
#             # Calculate potential black frames at the beginning of videos and remove related audio features
#             black_frames = abs(num_frames - len(audio_features))
#             audio_features = audio_features[black_frames:]
#
#             speakers_labels = get_speakers_labels(input_video, video_features)
#             speaking_frames = 0
#
#             speakers, correlations = detect_speakers(0)
#             # print(speakers)
#             for frame in list(correlations.keys()):
#                 no_speak = True
#                 for face in list(speakers_labels.keys()):
#                     if speakers_labels[face][frame] == 1:
#                         no_speak = False
#                 if not no_speak:
#                     speaking_frames += 1
#
#
#             correct_classifications = 0
#             for frame in range(len(speakers)):
#                 if speakers[frame] is not None and speakers_labels.get(speakers[frame]) is not None:
#                     if speakers_labels[speakers[frame]][frame] == 1:
#                         correct_classifications += 1
#             # print(speakers)
#             accuracy = correct_classifications / speaking_frames
#             videoAccuracies.append(accuracy)
#
#         if np.mean(videoAccuracies) > bestaccuracy:
#             bestaccuracy = np.mean(videoAccuracies)
#             bestmodel = model
#
#     print("Feature:", f_type)
#     print("Best model:", bestmodel)
#     print("Accuracy", bestaccuracy)
#     print("-----------------------")
#
############################################################################################
############################################################################################
############################################################################################
############################################################################################
# Calculating best frame neighborhood to consider for correlation mean (Only with LD Features)


# f_type = "AU"
# cca = rcca.CCA()
# cca.load("Models/AU_Model.hdf5")
#
# range_accuracies = []
# best_accuracy = 0
# best_range = 0
#
# for frames_range in range(31):
#
#     print(frames_range)
#     videoAccuracies = []
#
#     for input_video in ["002.mp4", "006.mp4", "008.mp4", "014.mp4", "016.mp4", "018.mp4", "025.mp4", "035.mp4",
#                         "044.mp4", "053.mp4", "056.mp4", "057.mp4", "061.mp4", "068.mp4", "073.mp4", "075.mp4"]:
#
#         # print("Processing: ", input_video)
#         videoObj = cv2.VideoCapture("Dataset/" + input_video)
#         fps = videoObj.get(cv2.CAP_PROP_FPS)
#         num_frames = int(videoObj.get(cv2.CAP_PROP_FRAME_COUNT))
#
#         video_features = extract_video_features()
#         audio_features = extract_audio_features()
#
#         # Calculate potential black frames at the beginning of videos and remove related audio features
#         black_frames = abs(num_frames - len(audio_features))
#         audio_features = audio_features[black_frames:]
#
#         speakers_labels = get_speakers_labels(input_video, video_features)
#         speaking_frames = 0
#
#         speakers, correlations = detect_speakers(frames_range)
#         # print(speakers)
#         for frame in list(correlations.keys()):
#             no_speak = True
#             for face in list(speakers_labels.keys()):
#                 if speakers_labels[face][frame] == 1:
#                     no_speak = False
#             if not no_speak:
#                 speaking_frames += 1
#
#
#         correct_classifications = 0
#         for frame in range(len(speakers)):
#             if speakers[frame] is not None and speakers_labels.get(speakers[frame]) is not None:
#                 if speakers_labels[speakers[frame]][frame] == 1:
#                     correct_classifications += 1
#         # print(speakers)
#         accuracy = correct_classifications / speaking_frames
#         videoAccuracies.append(accuracy)
#
#     range_accuracies.append(np.mean(videoAccuracies))
#     if np.mean(videoAccuracies) > best_accuracy :
#         best_accuracy = np.mean(videoAccuracies)
#         best_range = frames_range
#
#
# print("Best frame range accuracy: ", best_range)
# print("Best accuracy: ",best_accuracy)
# plt.plot(list(range(31)), range_accuracies)
# plt.vlines(best_range, ymin=np.min(range_accuracies), ymax=np.max(range_accuracies), color="red")
# plt.show()
############################################################################################
############################################################################################
############################################################################################
############################################################################################
# Single Test
#
#
frames_range = 10 # Defined by previous test
f_type = "LD"
cca = rcca.CCA()
if f_type == "LD":
    model = "LD_Model.hdf5"
else:
    model = "AU_Model.hdf5"

cca.load("Models/"+model)

videos = ["018.mp4"] #["002.mp4", "006.mp4", "008.mp4", "014.mp4", "016.mp4", "018.mp4", "025.mp4", "035.mp4",
                    #"044.mp4", "053.mp4", "056.mp4", "057.mp4", "061.mp4", "068.mp4", "073.mp4", "075.mp4"]

for input_video in videos:

    # print("Processing: ", input_video)
    videoObj = cv2.VideoCapture("Dataset/" + input_video)
    fps = videoObj.get(cv2.CAP_PROP_FPS)
    num_frames = int(videoObj.get(cv2.CAP_PROP_FRAME_COUNT))

    video_features = extract_video_features()
    audio_features = extract_audio_features()

    # Calculate potential black frames at the beginning of videos and remove related audio features
    black_frames = abs(num_frames - len(audio_features))
    audio_features = audio_features[black_frames:]

    speakers_labels = get_speakers_labels(input_video, video_features)
    speaking_frames = 0

    speakers, correlations = detect_speakers(frames_range)
    # print(speakers)
    for frame in list(correlations.keys()):
        no_speak = True
        for face in list(speakers_labels.keys()):
            if speakers_labels[face][frame] == 1:
                no_speak = False
        if not no_speak:
            speaking_frames += 1


    correct_classifications = 0
    for frame in range(len(speakers)):
        if speakers[frame] is not None and speakers_labels.get(speakers[frame]) is not None:
            if speakers_labels[speakers[frame]][frame] == 1:
                correct_classifications += 1
    # print(speakers)
    accuracy = correct_classifications / speaking_frames
    print("Input video: ",input_video)
    print("Accuracy: ",accuracy)





