import csv
import numpy as np
import rcca
import math
from sklearn import preprocessing
from python_speech_features import mfcc
from cv2 import *
import scipy.io.wavfile as wav
import os

def distance(a, b):
    """
    Calculate euclidean distance between two points a and b
    """
    return math.hypot(a[0]-b[0], a[1]-b[1])


def extract_features(video):
    """
    Extract all video features form .csv files produced by Openface
    :param video: input video
    :return: Dictionary of video features
    """

    # Extract video features
    input_file = "Extracted_Features/" + video[:len(video) - 4] + "_Features/" + video[:len(video) - 4] + ".csv"

    file = open(input_file)
    reader = csv.DictReader(file)
    video_feat = {}

    for row in reader:
        # Taking only good frames where faces have been detected with a confidence higher than 0.8 (Openface standard)
        if int(row[' success']) == 1 and float(row[' confidence']) > 0.8:
            face_id = int(row[' face_id'])
            frame = int(row['frame']) - 1

            video_feat.setdefault(frame, {})
            face_features = []

            # mouth action units
            au = ["10", "12", "14", "15", "17", "20", "23", "25", "26"]
            for i in au:
                face_features.append(float(row[' AU' + i + '_r']))
            for i in au:
                face_features.append(float(row[' AU' + i + '_c']))

            #  LandMarks
            for i in range(0, 68):
                face_features.append(float(row[' x_' + str(i)]))

            for i in range(0, 68):
                face_features.append(float(row[' y_' + str(i)]))

            video_feat[frame][face_id] = face_features

    # Extract audio features
    output_audio = video[:len(video) - 4] + ".wav"

    os.system("ffmpeg -i dataset/" + video + " -loglevel panic -ac 1 -vn -y\
                                Extracted_Features/" + video[:len(video) - 4] + "_Features/audio.wav")

    input = "Extracted_Features/" + video[:len(video) - 4] + "_Features/audio.wav"

    (rate, sig) = wav.read(input)

    audio_feat = mfcc(sig, rate, winstep=1 / fps, numcep=12)

    return video_feat, audio_feat


def select_features(feat_type, frame, face, v_feat):
    """
    Select specified features type of a specified face in a specified frame
    :param feat_type: Type of features: LD or AU
    :param v_feat: All video features extracted
    :return: Array of video features of a specified face in a specified frame
    """
    if feat_type is "LD":
        distances = []
        for couple in [(50, 58), (61, 67), (51, 57), (62, 66), (52, 56), (63, 65), (48, 54),
                       (60, 64), (49, 59), (53, 55)]:
            a_indexes = (couple[0] + 18, couple[0] + 86)
            b_indexes = (couple[1] + 18, couple[1] + 86)

            a = (v_feat[frame][face][a_indexes[0]], v_feat[frame][face][a_indexes[1]])

            b = (v_feat[frame][face][b_indexes[0]], v_feat[frame][face][b_indexes[1]])

            distances.append(distance(a, b))
        return distances
    else:
        return v_feat[frame][face][:9]


for feat_type in ["AU", "LD"]: #LD = Landmarks Distances, AU_ Action Units
    v_matrix = []
    a_matrix = []

    print("Creating training datasets...")
    for video in ["mentana.mp4", "girl.mp4", "girl2.mp4", "blogger.mp4", "indian.mp4"]:

        videoObj = cv2.VideoCapture("Dataset/" + video)
        fps = videoObj.get(cv2.CAP_PROP_FPS)
        num_frames = int(videoObj.get(cv2.CAP_PROP_FRAME_COUNT))

        v_features, a_features = extract_features(video)

        temp_matrix = []
        for frame in range(num_frames):
            if v_features.get(frame) is not None:

                temp_matrix.append(select_features(feat_type, frame, 0, v_features))
                a_matrix.append(a_features[frame])

        temp_matrix = preprocessing.normalize(temp_matrix)

        v_matrix.extend(temp_matrix)

    a_matrix = preprocessing.normalize(a_matrix)

    train_v = np.array(v_matrix)
    train_a = np.array(a_matrix)

    print("Number of examples: ", len(train_v))

    max_numCCs = np.shape(train_v)[1] if np.shape(train_v)[1] < np.shape(train_a)[1] else np.shape(train_a)[1]
    numCCs = list(range(3, max_numCCs+1))
    reg_coeffs = [100, 10, 1, 0, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    print("START TRAINING")
    for kernel in [True, False]:
        ktypes = ["poly", "gaussian"] if kernel else [None]
        for ktype in ktypes:
            for coeff in reg_coeffs:
                for numCC in numCCs:

                    cca = rcca.CCA(kernelcca=kernel, numCC=numCC, reg=coeff, ktype=ktype)
                    cca.train([train_v, train_a])

                    cca.save("TrainingModels/" + feat_type + "_" + str(kernel) + "_" + str(ktype) + "_" + str(numCC) + "_" + str(coeff) + ".hdf5")
                    # print("Model: "+f_type+"_"+str(kernel)+"_"+str(ktype)+"_"+str(numCC)+"_"+str(coeff))
                    # print("Cancorrs:", cca.cancorrs)
                    # print("----------------------------------")