from sklearn.preprocessing import StandardScaler
from skimage.restoration import estimate_sigma
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import preprocessing
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from scipy import stats
from tqdm import tqdm
from PIL import Image
from dom import DOM

import matplotlib.colors as mcolors
import imquality.brisque as brisque
import numpy as np
import warnings
import pickle
import cv2
import os


def ignore_warning():
    warnings.filterwarnings("ignore")


def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


def estimate_noise(img):
    return estimate_sigma(img, multichannel=True, average_sigmas=True)


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()


def estimate_entropy(im):
    # Compute normalized histogram -> p(g)
    p = np.array([(im == v).sum() for v in range(256)])
    p = p / p.sum()
    # Compute e = -sum(p(g)*log2(p(g)))
    e = -(p[p > 0] * np.log2(p[p > 0])).sum()
    return e


def dinamic_range(img):
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    pixel_brightness = []
    for x in range(1, 480):
        for y in range(1, 640):
            try:
                pixel = image[x, y]
                R, G, B = pixel
                brightness = sum([R, G, B]) / 3
                pixel_brightness.append(brightness)
            except IndexError:
                pass
    if min((pixel_brightness)) != 0:
        max_pic = np.log2(max(pixel_brightness))
        min_pic = np.log2(min(pixel_brightness))
        din_range = round(max_pic - min_pic, 2)
    else:
        din_range = round(np.log2(max(pixel_brightness)), 2)
    return din_range


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


def read_data(path):
    file = open(path, 'r')
    Lines = file.readlines()
    scores = []
    for line in Lines:
        per_score = line.strip().split()
        s = 0
        for score in per_score:
            s += int(score)
        scores.append(s)
    return scores


def initialize_features(length):
    features = {
        'width': np.zeros(length),
        'height': np.zeros(length),
        'size': np.zeros(length),
        'sharpness': np.zeros(length),
        'brightness': np.zeros(length),
        'noise': np.zeros(length),
        'entropy': np.zeros(length),
        'dynamic range': np.zeros(length),
        'brisque': np.zeros(length),
        'blur': np.zeros(length),
        'avg color': np.zeros((length, 3))
    }
    return features


def count_features(features):
    names = ['width(normalized)', 'height(normalized)', 'size(normalized)', 'sharpness(normalized)',
             'brightness(normalized)', 'noise(normalized)', 'entropy(normalized)', 'dynamic range(normalized)',
             'brisque(normalized)', 'blur(normalized)', 'R(normalized)', 'G(normalized)', 'B(normalized)']
    features_list = dict_to_list(features)
    normalized_list = preprocessing.normalize(features_list)
    list_n = {}
    count = 0
    for f in normalized_list:
        f = np.around(f, decimals=2)
        for i in range(len(f)):
            if f[i] in list_n:
                list_n[f[i]] += 1
            else:
                list_n[f[i]] = 1
        # normalized_plot(list_n, names[count])
        count += 1


def normalized_plot(list_n, name):
    list_n = dict(sorted(list_n.items()))
    keys = list(list_n.keys())
    values = list(list_n.values())
    plt.plot(keys, values)
    plt.title(name, color='g', fontsize=25)
    plt.xlabel(name)
    plt.ylabel("count")
    plt.show()


def extract_features(scores):
    features = initialize_features(len(scores))
    count = 1
    for i in tqdm(range(len(scores))):
        try:
            filepath = f'IQA_test/{count:03d}.jpg'
            img = Image.open(filepath)
            img2 = cv2.imread(filepath)
            iqa = DOM()
            img_resize = img.resize((600, 600))
            img2_resize = cv2.resize(img2, (600, 600))

            average_color_row = np.average(img2, axis=0)
            average_color = np.average(average_color_row, axis=0)

            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)

            features['width'][i] = img.width
            features['height'][i] = img.height
            features['size'][i] = round(os.stat(filepath).st_size/1000, 1)
            features['sharpness'][i] = round(iqa.get_sharpness(img2_resize), 3)
            features['brightness'][i] = round(calculate_brightness(img_resize), 3)
            features['noise'][i] = round(estimate_noise(img2_resize), 3)
            features['entropy'][i] = round(estimate_entropy(img2_resize), 3)
            features['dynamic range'][i] = round(dinamic_range(filepath), 3) \
                if abs(round(dinamic_range(filepath), 3)) != np.inf else 0
            features['brisque'][i] = round(brisque.score(img2_resize), 2)
            features['blur'][i] = round(fm, 0)
            features['avg color'][i] = np.around(average_color.tolist(), decimals=1)
            count += 1
        except:
            break
    return features


def print_features(feature, scores):
    RGB_name = ['Red', 'Green', 'Blue']
    for f in feature:
        if f == 'avg color':
            RGB = feature['avg color'].reshape(3, len(feature['avg color']))
            # RGB = feature['avg color'].trasnspose()
            for i in range(len(RGB_name)):
                pear = pearsonr(RGB[i], scores)[0]
                print(f'{RGB_name[i]}: {np.abs(pear)}')
        else:
            print(f'{f}: {np.abs(pearsonr(feature[f], scores)[0])}')


def save_data():
    scores = read_data('IQA_test/train.txt')
    features = extract_features(scores)
    with open('features1.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data():
    with open('features1.pickle', 'rb') as handle:
        features = pickle.load(handle)
    count_features(features)
    scores_train = read_data('IQA_test/train.txt')
    scores_test = read_data('IQA_test/test/test.txt')
    print_features(features, scores_train)
    return features, scores_train, scores_test


def dict_to_list(features):
    features_list = np.zeros((len(features) + 2, len(features['width'])))
    count = 0
    for i in features:
        if count < len(features) - 1:
            features_list[count] = features[i]
            count += 1
        else:
            colors = features[i]
            # colors = colors.reshape(3, len(colors))
            colors = colors.transpose()
            for j in range(len(colors)):
                features_list[count] = colors[j]
                count += 1
    return features_list


def save_tsv(features):
    features_list = dict_to_list(features)
    features_list = features_list.transpose()

    with open("features_list.tsv", "w") as record_file:
        for i in features_list:
            for j in i:
                record_file.write(str(j))
                record_file.write('\t')
            record_file.write('\n')

    return features_list


def plot_3d(features_list, scores):
    features_TSNE = TSNE(n_components=3, learning_rate='auto', init = 'pca').fit_transform(features_list)
    features_TSNE = features_TSNE.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    d = np.random.rand(10, 4)
    d[:, 3] = np.random.randint(1, 300, 10)
    X = features_TSNE[0]
    Y = features_TSNE[1]
    Z = features_TSNE[2]
    dims = scores
    bounds = [3, 4, 5, 6, 7, 8 ,9]
    colors = ['pink', 'purple', 'blue', 'green', 'yellow', 'orange', 'red']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, len(colors))
    fig = plt.figure()
    ax = Axes3D(fig)
    sc = ax.scatter(X, Y, Z, c=dims, marker="o", cmap=cmap, norm=norm)

    ax.set_xlabel('Center X (mm)')
    ax.set_ylabel('Center Y (mm)')
    ax.set_zlabel('Center Z (mm)')
    fig.colorbar(sc)
    # plt.show()



def extract_features_test(filepath):
    features = []
    img = Image.open(filepath)
    img2 = cv2.imread(filepath)
    iqa = DOM()
    img_resize = img.resize((600, 600))
    img2_resize = cv2.resize(img2, (600, 600))

    average_color_row = np.average(img2, axis=0)
    average_color = np.average(average_color_row, axis=0)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    features.append(img.width)
    features.append(img.height)
    features.append(round(os.stat(filepath).st_size / 1000, 1))
    features.append(round(iqa.get_sharpness(img2_resize), 3))
    features.append(round(calculate_brightness(img_resize), 3))
    features.append(round(estimate_noise(img2_resize), 3))
    features.append(round(estimate_entropy(img2_resize), 3))
    features.append(round(dinamic_range(filepath), 3)) \
        if abs(round(dinamic_range(filepath), 3)) != np.inf else 0
    features.append(round(brisque.score(img2_resize), 2))
    features.append(round(fm, 0))
    x = np.around(average_color.tolist(), decimals=1)
    for i in x:
        features.append(i)

    return features


def MVR(features, score, score_test):
    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.3, kernel='sigmoid'))
    regr.fit(features, score)

    with open('test1.pickle', 'rb') as handle:
        features_test = pickle.load(handle)

    acc = stats.pearsonr(score_test, [int(i) for i in regr.predict(features_test)])
    print('accuracy:', acc[0])


if __name__ == '__main__':
    ignore_warning()
    # save_data()m
    features, scores, score_test = load_data()
    features_list = save_tsv(features)
    plot_3d(features_list, scores)
    MVR(features_list, scores, score_test)

