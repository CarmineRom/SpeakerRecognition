import shutil
import csv
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from cv2 import *
from moviepy.editor import *
import rcca
import numpy as np
from sklearn import preprocessing


def extract_video_features():
    """
    Executes Openface and extracts face features
    :return: face features as dictionary
    """

    # Openface execution for face detection
    print("Extracting faces features...")
    os.system("openface\FaceLandMarkVidMulti.exe -f input/" + input_video + " -out_dir input")

    # Face feature extraction from Openface output file
    file = open("input/" + input_video[:len(input_video)-4] + ".csv")
    reader = csv.DictReader(file)
    features = {}

    for row in reader:

        # Taking only good frames where faces have been detected with a confidence higher than 0.5 (Openface standard)
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

            features[frame][face_id] = face_features

    return features


def extract_audio_features():

    # Audio extraction with ffmpeg and scipy
    print("Extracting audio from video...")
    os.system("ffmpeg -i input/" + input_video + " -loglevel panic -ac 1 -vn -y input/audio.wav")
    (rate, sig) = wav.read("input/audio.wav")

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

    distances = []
    for couple in [(50, 58), (61, 67), (51, 57), (62, 66), (52, 56), (63, 65), (48, 54),
                   (60, 64), (49, 59), (53, 55)]:
        a_indexes = (couple[0], couple[0] + 68)
        b_indexes = (couple[1], couple[1] + 68)

        a = (video_features[frame][face][a_indexes[0]], video_features[frame][face][a_indexes[1]])

        b = (video_features[frame][face][b_indexes[0]], video_features[frame][face][b_indexes[1]])

        distances.append(distance(a, b))
    return distances


def detect_speakers():
    """
    Executes Cross Modal Association between each frame video and each frame audio
    :return:
        Array of faces indexes representing the speaking face of each frame
    """
    frames_range = 10
    audio_feat = preprocessing.normalize(audio_features)
    correlations = {}
    for frame in list(video_features.keys()):

        if len(video_features[frame].keys()) > 1:  # More than one face has been detected in the frame

            #  Defining the neighborhood in the interval [start_frame, end_frame]
            start_frame = 0 if frame - frames_range < 0 else frame - frames_range
            end_frame = list(video_features.keys())[-1] if frame + frames_range >= list(video_features.keys())[-1] \
                else frame + frames_range

            correlations.setdefault(frame, {})
            for face in list(video_features[frame].keys()):
                face_corrs = []

                # Calculating correlations of each frame in [start_frame, end_frame]
                for f in range(start_frame, end_frame+1):
                    if not f >= len(audio_feat) and video_features.get(f) is not None and video_features[f].get(face) is not None:
                        v_set = np.array(get_face_features(f, face))
                        a_set = np.array(audio_feat[f])
                        v_set = preprocessing.normalize([v_set])[0]
                        v_proj = np.dot(v_set, cca.ws[0])
                        a_proj = np.dot(a_set, cca.ws[1])
                        face_corrs.append(abs(np.corrcoef(v_proj, a_proj)[0, 1]))

                # Save the mean as correlation
                correlations[frame][face] = np.mean(face_corrs)

    # Classification of frames based on higher face's correlation
    speakers = [None]*num_frames
    for frame in correlations.keys():
        max_corr = 0
        max_corr_face = None
        for face, corr in correlations[frame].items():
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                max_corr_face = face

        speakers[frame] = max_corr_face

    return speakers


def build_output():
    """
    Build the output video where talking faces are marked by green rectangle
    """

    # Frames extraction
    count = 0
    success = 1
    while success:
        success, image = videoObj.read()
        cv2.imwrite("input/frame%d.jpg" % count, image)
        count += 1

    # Calculates dimensions of frames and sets the format of the final video for the Writer
    width = int(videoObj.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    format = "mp4v"
    fourcc = VideoWriter_fourcc(*format)
    out_video = cv2.VideoWriter("input/temp_video.mp4", fourcc, int(round(fps)), (width, height))

    # Adds black frames (if exist) at the beginning of video to preserve audio/video synchronization
    for i in range(black_frames):
        img = np.zeros((height, width, 3), dtype="uint8")
        out_video.write(img)
    last_faces_coord = {}

    # Dictionary needed to save last coordinates of a face if it can not be detected
    for face in set(speakers):
        last_faces_coord.setdefault(face, ((0, 0), (0, 0)))
    for i in range(num_frames):

        img = cv2.imread("input/frame" + str(i) + ".jpg")
        # Gets 2 points coordinates of the talking face to draw rectangle
        if speakers[i] is not None:
            if video_features.get(i) is not None and video_features[i].get(speakers[i]) is not None: # Check if face has been detected
                x_min = int(np.min(video_features[i][speakers[i]][:68]))
                x_max = int(np.max(video_features[i][speakers[i]][:68]))
                y_min = int(np.min(video_features[i][speakers[i]][68:]))
                y_max = int(np.max(video_features[i][speakers[i]][68:]))

                pt1 = (x_min - 20, y_min - 20)
                pt2 = (x_max + 20, y_max + 20)

                last_faces_coord[speakers[i]] = (pt1, pt2)
            else: # Face has not been detected so uses last available coordinates
                pt1, pt2 = last_faces_coord[speakers[i]][0], last_faces_coord[speakers[i]][1]

            # Draws rectangle
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 5)

        # Writes edited frame in the final output video
        out_video.write(img)

    out_video.release()
    cv2.destroyAllWindows()

    # Inserts audio in the new builded video
    final_output = VideoFileClip("input/temp_video.mp4").set_audio(AudioFileClip("input/audio.wav"))
    final_output.write_videofile("output/out_" + input_video)


if len(sys.argv) < 2:
    print("Usage: place your .mp4 input video in the 'input' folder and run with its name as argument")
else:

    input_video = str(sys.argv[1])
    if not os.path.exists("input/" + input_video):
        print("%s not found in the input folder" % input_video)
    else:

        videoObj = cv2.VideoCapture("input/" + input_video)
        fps = videoObj.get(cv2.CAP_PROP_FPS)
        num_frames = int(videoObj.get(cv2.CAP_PROP_FRAME_COUNT))

        video_features = extract_video_features()
        audio_features = extract_audio_features()

        # Calculates potential black frames at the beginning of videos and removes related audio features
        if len(audio_features) >= num_frames:
            black_frames = abs(num_frames - len(audio_features))
            audio_features = audio_features[black_frames:]
        else:
            black_frames = 0

        # Loads the CCA Model
        cca = rcca.CCA()
        cca.load("Model.hdf5")

        print("Detecting speakers...")
        speakers = detect_speakers()

        print("Building output...")
        build_output()

        #Temporary and input files cleaning
        for file in os.listdir("input"):
            if os.path.isdir("input/" + file):
                shutil.rmtree("input/" + file)
            elif not file == input_video:
                os.remove("input/" + file)