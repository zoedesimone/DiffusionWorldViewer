# Worldviewer Utility functions 

#imports
from matplotlib import pyplot as plt
import torch
from PIL import Image

# SEGA
from semdiffusers import SemanticEditPipeline

# Imports for FairFace
import math
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

import random
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch.nn as nn
from torchvision import models, transforms
import dlib
import os
from tqdm import tqdm
from PIL import Image
import glob
import tqdm

import shutil
import os

import gradio as gr
from gradio.inputs import Slider
import time

import altair as alt

from PIL import Image


#Stable Diffusion Model info
model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe_edit =  SemanticEditPipeline.from_pretrained(model_id,safety_checker=None,)
pipe_edit = pipe_edit.to(device)


def remove_whitespaces(prompt):
    """
    Removes whitespace from a string.

    Args:
        prompt (str): The input string containing whitespace.

    Returns:
        str: The input string with whitespace removed.

    """
    string_without_spaces = prompt.replace(" ", "")
    return string_without_spaces


def save_image(image, seed, prompt, folder, df_file_path, edited):
    """
    Saves an image from stable diffusion to a folder and updates a dataframe with seed, prompt, and file path information.

    Args:
        image (PIL.Image.Image): The image to be saved.
        seed (int): The seed used for generating the image.
        prompt (str): The prompt associated with the image.
        folder (str): The folder path where the image will be saved.
        df_file_path (str): The file path of the dataframe to be updated.
        edited (bool): Indicates if the image is edited (True) or a baseline (False).

    Returns:
        None

    """
    # Create the output folder if it doesn't exist
    #os.makedirs(folder, exist_ok=True)

    # Convert the image to PIL.Image format if it is a torch.Tensor
    if isinstance(image, torch.Tensor):
        # Assuming the image tensor is in RGB format
        image = transforms.ToPILImage()(image.cpu())

    ext = ".jpg"
    filepath = folder + remove_whitespaces(prompt) + "_" + str(seed) + ext
    image.save(filepath)
    print(f"Saved image {seed} successfully!")

    df = open_df(df_file_path)
    new_row = {'prompt': prompt, 'filepath': filepath, 'seed': seed, 'baseline_edited': edited}
    df = df.append(new_row, ignore_index=True)
    print("Successfully added new row to dataframe.")
    df.to_csv(df_file_path, index=False)



def open_df(df_path):
    """
    Opens a DataFrame from a CSV file. If the file doesn't exist, creates an empty DataFrame and saves it to the file path.

    Args:
        df_path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame if the file exists, otherwise an empty DataFrame.

    Raises:
        FileNotFoundError: If the specified file path does not exist.

    """
    # Check if file exists
    if os.path.isfile(df_path):
        # Read data from file and create DataFrame
        df = pd.read_csv(df_path)
        print('DataFrame loaded successfully.')
    else:
        # Create an empty DataFrame
        df = pd.DataFrame()
        # Save the empty DataFrame to the specified path
        df.to_csv(df_path, index=False)
        print('File created successfully.')

    return df


def fairface_sd_df(csv_filepath):
    """
    Creates an empty DataFrame to store the stable diffusion image generation parameters, such as `prompt`,
    `filepath`, `seed`, `baseline_edited`, and scores predicted using FairFace.

    Args:
        csv_filepath (str): The file path to save the DataFrame as a CSV file.

    Returns:
        None
    """
    df = pd.DataFrame(columns=['prompt', 'filepath', 'seed', 'baseline_edited', '0-2', '3-9', '10-19', '20-29',
                               '30-39', '40-49', '50-59', '60-69', '70+', 'White', 'Black', 'Latino_Hispanic',
                               'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern', 'Male', 'Female'])

    df.to_csv(csv_filepath, index=False)


## FairFace Classification Utils


def get_random(length):
    """
    Generate a random list of binary values.

    Args:
        length (int): The desired length of the list.

    Returns:
        list: A list of binary values (0s and 1s) with the specified length.
    """
    random_list = []
    while sum(random_list) != length/2:
        random_list = [random.randint(0, 1) for i in range(length)]
    return random_list


def face_existing(img, cnn_face_detector, default_max_size=800, size=300, padding=0.25):
    """
    Detects faces in an image using a pre-trained CNN face detector.

    Args:
        img (numpy.ndarray): The input image in the form of a NumPy array.
        cnn_face_detector: The pre-trained CNN face detector model.
        default_max_size (int): The maximum size for image resizing. Defaults to 800.
        size (int): The desired size for image resizing. Defaults to 300.
        padding (float): The padding ratio to be applied during image resizing. Defaults to 0.25.

    Returns:
        int: The number of faces detected in the input image.
    """
    old_height, old_width, _ = img.shape

    # Resize the image while preserving the aspect ratio
    if old_width > old_height:
        new_width = default_max_size
        new_height = int(default_max_size * old_height / old_width)
    else:
        new_width = int(default_max_size * old_width / old_height)
        new_height = default_max_size
    img = dlib.resize_image(img, rows=new_height, cols=new_width)

    # Detect faces using the CNN face detector
    dets = cnn_face_detector(img, 1)
    num_faces = len(dets)

    return num_faces

def detect_face(image_paths, SAVE_DETECTED_AT, cnn_face_detector, default_max_size=800, size=300, padding=0.25):
    """
    Detects and saves aligned faces from a list of image paths using a pre-trained CNN face detector.

    Args:
        image_paths (list): A list of image paths to process and detect faces from.
        SAVE_DETECTED_AT (str): The directory path to save the detected faces.
        cnn_face_detector: The pre-trained CNN face detector model.
        default_max_size (int): The maximum size for image resizing. Defaults to 800.
        size (int): The desired size for the aligned faces. Defaults to 300.
        padding (float): The padding ratio to be applied during image resizing. Defaults to 0.25.

    Returns:
        None
    """
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height

    for index, image_path in tqdm(enumerate(image_paths)):
        if index % 1000 == 0:
            print('---%d/%d---' % (index, len(image_paths)))

        img = dlib.load_rgb_image(image_path)

        old_height, old_width, _ = img.shape

        # Resize the image while preserving the aspect ratio
        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height = int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        # Detect faces using the CNN face detector
        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)

        if num_faces != 1:
            print(f"no face found {index}")
            continue

        # Find the 5 face landmarks needed for alignment
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))

        # Obtain aligned faces from the image
        images = dlib.get_face_chips(img, faces, size=size, padding=padding)

        # Save the aligned faces
        for idx, image in enumerate(images):
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
            face_name = os.path.join(SAVE_DETECTED_AT, path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
            dlib.save_image(image, face_name)


def predict_age_gender_race(save_prediction_at, imgs_path = 'baseline_image/'):
    """
    Predicts age, gender, and race for images located in a specified directory and saves the predictions to a CSV file.

    Args:
        save_prediction_at (str): The file path to save the prediction results (CSV format).
        imgs_path (str): The directory path containing the images to predict. Defaults to 'baseline_image/'.

    Returns:
        pandas.DataFrame: A DataFrame containing the prediction results for each image, including face name, predicted race,
                          predicted gender, predicted age, race scores, gender scores, age scores, confidence in race prediction,
                          confidence in gender prediction, and confidence in age prediction.
    """
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path) if 'ipynb' not in x]

    model_fair_7 = models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('dlib_models/res34_fair_align_multi_7_20190809.pt'))
    model_fair_7 = model_fair_7.to('cuda')
    model_fair_7.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    #Add the prediction score for the highest predicted classe by FairFace
    #Used to report the confidence of the model
    confidence_race = []
    confidence_gender = []
    confidence_age = []

    for index, img_name in enumerate(img_names):
        if index % 1000 == 0:
            print("Predicting... {}/{}".format(index, len(img_names)))

        face_names.append(img_name)
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to('cuda')

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # softmax probability of the highest scored class for age, race and gender
        confidence_race.append(race_score[race_pred])
        confidence_gender.append(gender_score[gender_pred])
        confidence_age.append(age_score[age_pred])


    result = pd.DataFrame([face_names,
                           race_preds_fair,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair,
                           gender_scores_fair,
                           age_scores_fair,
                           confidence_race,
                           confidence_gender,
                           confidence_age]).T
    result.columns = ['face_name_align',
                      'race_preds_fair',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair',
                      'gender_scores_fair',
                      'age_scores_fair',
                      'confidence_race',
                      'confidence_gender',
                      'confidence_age']
    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

    result[['face_name_align',
            'race',
            'gender', 'age',
            'race_scores_fair',
            'gender_scores_fair',
            'age_scores_fair',
            'confidence_race',
            'confidence_gender',
            'confidence_age']].to_csv(save_prediction_at, index=False)

    return result


def ensure_dir(directory):
    """
    Ensures that a directory exists. If the directory does not exist, it creates it.

    Args:
        directory (str): The directory path to ensure existence.

    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

#Visualize the underlying distributions of the baseline dataset using FairFace.

def sd_fairface_distribution(df):
    """
    Calculates and saves the distribution of age, race, and gender from a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the age, race, and gender columns.

    Returns:
        tuple: A tuple containing three DataFrames representing the distributions of age, race, and gender, respectively.

    """

    # Calculate age distribution
    age_ranges = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    age_distribution = []
    for a in age_ranges:
        sum = (df.age.values == a).sum()
        age_distribution.append(sum)

    # Calculate race distribution
    race_ranges = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
    race_distribution = []
    for r in race_ranges:
        sum = (df.race.values == r).sum()
        race_distribution.append(sum)

    # Calculate gender distribution
    gender_ranges = ['Male', 'Female']
    gender_distribution = []
    for g in gender_ranges:
        sum = (df.gender.values == g).sum()
        gender_distribution.append(sum)

    # Create DataFrames for age, race, and gender distributions
    df_age = pd.DataFrame(age_distribution, index=age_ranges, columns=['count'])
    df_age['%'] = df_age['count'].div(len(df))
    df_age['label'] = df_age.index

    df_race = pd.DataFrame(race_distribution, index=race_ranges, columns=['count'])
    df_race['%'] = df_race['count'].div(len(df))
    df_race['label'] = df_race.index

    df_gender = pd.DataFrame(gender_distribution, index=gender_ranges, columns=['count'])
    df_gender['%'] = df_gender['count'].div(len(df))
    df_gender['label'] = df_gender.index

    # Save distributions to CSV files
    df_age.to_csv("age.csv", index=False)
    df_gender.to_csv("gender.csv", index=False)
    df_race.to_csv("race.csv", index=False)

    return df_age, df_race, df_gender


##SD Editing functions

def flip_image_gender(prompt, edit, seed):
    """
    Returns an edited SD image by performing a gender flip edit based on the given 'edit',
    using the original prompt and seed.

    Args:
        prompt (str): The original prompt for generating the image.
        edit (list(str)): The gender flip edit to apply to the image.
        seed (int): The seed used for random number generation.

    Returns:
        torch.Tensor: The edited SD image.

    """

    target = {
        'editing_prompt': edit,
        'reverse_editing_direction': [True, False],
        'edit_warmup_steps': 5,
        'edit_guidance_scale': 5,
        'edit_threshold': 0.9,
        'edit_momentum_scale': 0.5,
        'edit_mom_beta': 0.6
    }

    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe_edit(prompt=prompt, generator=gen, num_images_per_prompt=1, guidance_scale=7, **target)

    return out.images[0]

def get_images_in_folder(output_folder):
    """
    Retrieves all the images in the specified output folder and returns them as a list of images.

    Args:
        output_folder (str): The path to the folder containing the images.

    Returns:
        list: A list of PIL Image objects representing the images in the folder.
    """
    image_list = []

    # Iterate over the files in the folder
    for filename in os.listdir(output_folder):
        # Check if the file is an image
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Load the image using PIL
            image = Image.open(os.path.join(output_folder, filename))
            # Append the image to the list
            image_list.append(image)

    return image_list

def generate_sd_images(prompt, seed, n_images):
    """
    Generates and saves StableDiffusion-generated images in batches of 1 to a specified `image_folder` and saves information to `csv_filepath`.

    The function runs StableDiffusion in batches to avoid running out of memory (CUDA out of MEMORY error).

    Args:
        prompt (str): The prompt used for generating the images.
        seed (int): The starting seed value for generating the images.
        n_images (int): The number of images to generate.

    Returns:
        None
    """
    image_folder = "baseline_image/"
    csv_filepath = "image_dataframe.csv"

    seeds = np.arange(1, n_images, 1).tolist()  # save the seeds for later

    images = []

    for seed in range(n_images):
        print(f"Seed: {seed}")
        gen = torch.Generator(device=device).manual_seed(seed)
        org = pipe_edit(prompt=prompt, generator=gen, num_images_per_prompt=1, guidance_scale=7)
        save_image(org.images[0], seed, prompt, folder=image_folder, df_file_path=csv_filepath, edited=False)
        images.append(org.images[0])
    return images



def absolute_gender_edit(prompt, seed, n_images, percent_f,percent_m, output_folder, df_file_path):
    """
    Creates SD images to achieve absolute gender distributions based on the
    specified male [percent_m] and female [percent_f] probabilities (0-100).
    Saves information regarding the edited images in a dataframe specified by 'df_file_path'.

    Args:
        prompt (str): The prompt for generating the image.
        seed (int): The seed used for random number generation.
        n_images (int): The number of images to generate.
        percent_f (float): The percentage of images to modify to female.
        percent_m (float): The percentage of images to modify to male.
        output_folder (str): The path to the folder where the edited images will be saved.
        df_file_path (str): The path to the dataframe file to save image information. Use "existing" to append to an existing file.

    Returns:
        output_images: output images
    """
    #Editing logic from SEGA
    edit1 = ['male person', 'female person'] #male to female
    edit2 = ['female person','male person'] #female to male

    # calculate the number of images to modify
    num_to_modify_f = int(n_images * percent_f)
    print(f"num_to_modify_f:{num_to_modify_f}")

    # randomly select the indices of the images to modify
    images_list = [i for i in range(n_images)]
    print(images_list)
    indices_to_modify_f = random.sample(images_list, num_to_modify_f)

    print(indices_to_modify_f)
    #indices_to_modify_m = list(range(len(image_filenames))) - (indices_to_modify_f)
    indices_to_modify_m = list(set(images_list) - set(indices_to_modify_f)) #get the other indices
    print(indices_to_modify_m)

    print(f"indices_to_modify_f: {indices_to_modify_f}")
    print(f"indices_to_modify_m: {indices_to_modify_m}")

    images = []

    # loop over the selected indices and modify the corresponding images
    for indexf in indices_to_modify_f:

        # perform the desired modifications
        edited_image = flip_image_gender(prompt, edit1, indexf) #male to female
        save_image(edited_image, indexf, prompt, output_folder,df_file_path, edited = True)
        images.append(edited_image)
        print(f"saved edited m --> f image")

    for indexm in indices_to_modify_m:

        # perform the desired modifications
        edited_image = flip_image_gender(prompt, edit2, indexm) #male to female
        save_image(edited_image, indexm, prompt, output_folder,df_file_path, edited = True)
        images.append(edited_image)
        print(f"saved edited f --> m image")

    return images

def relative_gender_edit(prompt, seed, path_to_images, percent_to_modify_f,percent_to_modify_m, output_folder, df_file_path):
    """
    Edits the SD images in the 'path_to_images' directory to achieve relative gender distributions
    based on the specified male [percent_to_modify_m] and female [percent_to_modify_f] probabilities (0-100).
    Saves information regarding the edited images in a dataframe specified by 'df_file_path'.

    Args:
        path_to_images (str): The path to the original image directory.
        percent_to_modify_f (float): The percentage of images to modify to female.
        percent_to_modify_m (float): The percentage of images to modify to male.
        output_folder (str): The path to the folder where the edited images will be saved.
        df_file_path (str): The path to the dataframe file to save image information. Use "existing" to append to an existing file.

    Returns:
        output_images: output images
    """
    #Editing logic from SEGA
    edit1 = ['male person', 'female person'] #male to female
    edit2 = ['female person','male person'] #female to male

    # get the list of image filenames in the directory
    image_filenames = os.listdir(path_to_images)

    # calculate the number of images to modify
    num_to_modify_f = int(len(image_filenames) * (percent_to_modify_f / 100.0))


    # randomly select the indices of the images to modify

    indices_to_modify_f = random.sample(range(len(image_filenames)), num_to_modify_f)
    print(indices_to_modify_f)
    #indices_to_modify_m = list(range(len(image_filenames))) - (indices_to_modify_f)
    indices_to_modify_m = list(set(list(range(len(image_filenames)))) - set(indices_to_modify_f)) #get the other indices
    print(indices_to_modify_m)

    print(f"indices_to_modify_f: {indices_to_modify_f}")
    print(f"indices_to_modify_m: {indices_to_modify_m}")

    # loop over the selected indices and modify the corresponding images
    for indexf in indices_to_modify_f:
        # open the image using PIL
        image = Image.open(os.path.join(path_to_images, image_filenames[indexf]))
        print(f"changing image:{image_filenames[indexf]}")

        # Split the file path into a base name and an extension
        base_name, extension = os.path.splitext(image_filenames[indexf])
        # Get the character before the '.' in the base name
        character = base_name[-1]
        seed = int(character)
        print(f"seed: {seed}")

        # perform the desired modifications
        #TODO: insert the stable diffusion edit here
        edited_image = flip_image_gender(prompt, edit1, seed) #male to female

        # save the modified image
        file_path = image_filenames[indexf]
        save_image(edited_image, seed, prompt, output_folder,df_file_path, edited = True)

        print(f"saved edited m --> f image")

    for indexm in indices_to_modify_m:
        # open the image using PIL
        image = Image.open(os.path.join(path_to_images, image_filenames[indexm]))
        print(f"changing image:{image_filenames[indexm]}")


        # Split the file path into a base name and an extension
        base_name, extension = os.path.splitext(image_filenames[indexm])
        # Get the character before the '.' in the base name
        character = base_name[-1]
        seed = int(character)
        print(f"seed: {seed}")

        # perform the desired modifications
        edited_image = flip_image_gender(prompt, edit2, seed) #male to female

        # save the modified image
        file_path = image_filenames[indexm]
        save_image(edited_image, seed, prompt, output_folder,df_file_path, edited = True)

        print(f"saved edited f --> m image")

    output_images = get_images_in_folder(output_folder)
    return output_images


#@title ##Simultaneous edits gender, race and age edit
def generate_edit_list(n, gender_counts, race_counts, age_counts):
    """
    Generate a list of n images with attributes based on given gender, race, and age distributions.

    Parameters:
    - n (int): The number of images to generate.
    - gender_counts (list of int): A list of counts representing the distribution of genders. Should match the length of gender_options.
    - race_counts (list of int): A list of counts representing the distribution of races. Should match the length of race_options.
    - age_counts (list of int): A list of counts representing the distribution of ages. Should match the length of age_options.

    Returns:
    - list of lists: A list of images where each image is represented as a list with attributes [gender, age, race].

    Note:
    - The function uses weighted random choice to determine the attributes of each image based on the provided distributions.

    Example:
    >>> generate_edit_list(2, [3, 2], [2, 2, 1, 1, 1, 1, 1], [5, 3, 1, 2, 2, 3, 3, 2, 1])
    [['Female person', 'Infant', 'White person'], ['Male person', 'Adult', 'Black person']]
    """

    gender_options = ['Male person', 'Female person']
    race_options = ['White person', 'Black person', 'Latino_Hispanic person', 'East Asian person', 'Southeast Asian person', 'Indian person', 'Middle Eastern person']
    age_options = ['Infant', 'Child', 'Teenager', 'Young adult', 'Adult', 'Middle-aged person', 'Middle-aged person', 'Senior citizen', 'Elderly person']

    total_gender = sum(gender_counts)
    total_race = sum(race_counts)
    total_age = sum(age_counts)

    gender_percentages = [count / total_gender * 100 for count in gender_counts]
    race_percentages = [count / total_race * 100 for count in race_counts]
    age_percentages = [count / total_age * 100 for count in age_counts]

    images = []

    for _ in range(n):
        gender = random.choices(gender_options, weights=gender_percentages)[0]
        race = random.choices(race_options, weights=race_percentages)[0]
        age = random.choices(age_options, weights=age_percentages)[0]

        image = [gender, age, race]
        images.append(image)

    return images

def edit_image(prompt, edits, seed):
    """
    Returns an edited SD image by performing a gender, race and age edit based on the given 'edits',
    using the original prompt and seed.

    Args:
        prompt (str): The original prompt for generating the image.
        edits (list(str)): The edits to apply to the image.
        seed (int): The seed used for random number generation.

    Returns:
        torch.Tensor: The edited SD image.

    """
    single_string = ", ".join(edits )
    edits = [single_string]

    target = {
        'editing_prompt': edits,
        'reverse_editing_direction': False,
        'edit_warmup_steps': 10, #instead of 5
        'edit_guidance_scale': 5,
        'edit_threshold': 0.9,
        'edit_momentum_scale': 0.5,
        'edit_mom_beta': 0.6
    }

    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe_edit(prompt=prompt, generator=gen, num_images_per_prompt=1, guidance_scale=7, **target)

    return out.images[0]

def absolute_gender_race_age_edit(prompt, seed, n_images, edit_strength, p_female, p_male,
                                  p_white, p_black, p_latino, p_eastasian, p_southeast, p_indian, p_middleeast,
                                  p_infant, p_child, p_teenager, p_youngadult,p_adult, p_middleaged, p_middleaged2, p_seniorcitizen, p_elderly):
    """
    Creates SD images to achieve absolute gender, race and age distributions based on the
    specified male [percent_m] and female [percent_f] probabilities (0-100).
    Saves information regarding the edited images in a dataframe specified by 'df_file_path'.

    Args:
        prompt (str): The prompt for generating the image.
        seed (int): The seed used for random number generation.
        n_images (int): The number of images to generate.
        percent_f (float): The percentage of images to modify to female.
        percent_m (float): The percentage of images to modify to male.
        output_folder (str): The path to the folder where the edited images will be saved.
        df_file_path (str): The path to the dataframe file to save image information. Use "existing" to append to an existing file.

    Returns:
        output_images: output images
    """
    output_folder= "edited_images/"
    df_file_path="edited_images.csv"

    gender_counts = [p_female, p_male]
    race_counts = [p_white, p_black, p_latino, p_eastasian, p_southeast, p_indian, p_middleeast]
    age_counts = [p_infant, p_child, p_teenager, p_youngadult,p_adult, p_middleaged, p_middleaged2, p_seniorcitizen, p_elderly]

    edit_list = generate_edit_list(n_images, gender_counts, race_counts, age_counts)

    images = []
    print(edit_list)

    # Print the generated image list
    for edit in edit_list:

        # perform the desired modifications
        edited_image = edit_image(prompt, edit, seed)
        save_image(edited_image, seed, prompt, output_folder, df_file_path, edited = True)
        images.append(edited_image)
        print(f"saved edited image with edit: {str(edit)}.")
        seed += 1

    return images


def get_image_paths(folder_path, extensions=[".jpg", ".jpeg", ".png"]):
    """
    Retrieve a list of image file paths within a specified folder and its subfolders.

    Args:
        folder_path (str): The path to the folder to search for images.
        extensions (list, optional): A list of file extensions to consider as image files.
            Defaults to [".jpg", ".jpeg", ".png"].

    Returns:
        list: A list of file paths to image files found within the folder and its subfolders.
    """
    
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    print(f"image_paths: {image_paths}")
    return image_paths

def clear_folder(folder):
    """
    Clears all files with the '.jpg' extension from the specified folder.

    Args:
        folder (str): The path to the folder to be cleared.
    """
    jpg_files = [file for file in os.listdir(folder) if file.endswith('.jpg')]

    for file in jpg_files:
        file_path = os.path.join(folder, file)
        os.remove(file_path)
    print("Cleared all files")

#@title #Helper functions

def process_data(data):
    """
    Process data by setting 'tag' values based on 'value' and 'variable' columns.

    Args:
        data (dict): A dictionary containing 'value' and 'variable' lists.

    Returns:
        dict: A dictionary with the 'tag' list updated based on 'value' and 'variable'.
    """
    if 'tag' not in data:
        data['tag'] = [''] * len(data['value'])

    for i in range(len(data['value'])):
        if data['value'][i] == 0:
            data['tag'][i] = ''
        else:
            data['tag'][i] = data['variable'][i]

    return data

def checkbox_selection_to_unary_list(selected_list, original_list):
  """
    Convert a selected list of items to a unary list based on an original list.

    Args:
        selected_list (list): List of selected items.
        original_list (list): Original list of items.

    Returns:
        list: A unary list where selected items are represented as 1 and others as 0.
    """
  selected_length = len(selected_list)
  lst = []

  for i in original_list:
        if i in selected_list:
            lst.append(1 / selected_length)
        else:
            lst.append(0)

  return lst

def make_stacked_plot(df, x_value_type, color_scheme):
  """
    Create a stacked bar chart based on DataFrame values.

    Args:
        df (pd.DataFrame): DataFrame containing plot data.
        x_value_type (str): Type of x-values, either "quantitative" or "nominal".
        color_scheme (str): Color scheme for the chart, choose from available schemes: https://vega.github.io/vega/docs/schemes/#reference

    Returns:
        alt.Chart: Stacked bar chart created using Altair.
    """
  bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("value", type="quantitative", title="%", stack='zero',scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("index", type="nominal", title=""),
            color=alt.Color("variable", scale= alt.Scale(scheme= color_scheme), type= x_value_type, title="", legend=None),

        )
  )

  text=alt.Chart().mark_text(align='center',baseline='line-top', dx=-15, dy=3).encode(
      x=alt.X("value", stack = "zero",scale=alt.Scale(domain=[0, 100])),
      y=alt.Y("index"),
      color=alt.Color('variable', legend=None, scale=alt.Scale(range=['white'])),
      text=alt.Text('tag')
  )
  chart = alt.layer(bars, text, data=df).resolve_scale(color='independent').properties(
    width= 500,
    height=30
)
  return chart

def update_tags(df):
    """
    Update tags in a DataFrame based on a predefined mapping.

    Args:
        df (pd.DataFrame): DataFrame with 'tag' column to be updated.

    Returns:
        pd.DataFrame: DataFrame with 'tag' values updated based on a predefined mapping.
    """

    tag_mapping = {
        "White": "WH",
        "Black": "BL",
        "Hispanic": "HI",
        "East Asian": "EA",
        "Southeast Asian": "SE",
        "Indian": "I",
        "Middle Eastern": "ME",
        "0-2": "0-2",
        "3-9": "3-9",
        "03-09": "3-9",
        "10-19":"10-19",
        "20-29":"20-29",
        "30-39":"30-39",
        "40-49":"40-49",
        "50-59":"50-59",
        "60-69":"60-69",
        "70+":"70+",
        "Female": "F",
        "Male": "M",
        "":"",
        "Latino_Hispanic": "HI",
        'White': "WH"
    }

    df['tag'] = df['tag'].map(tag_mapping)
    return df

def calculate_midpoints(df):
    """
    Calculate midpoints for a DataFrame based on the 'value' column.

    Args:
        df (pd.DataFrame): DataFrame with 'value' column.

    Returns:
        list: List of calculated midpoints.
    """

    midpoints = []
    cumulative_sum = 0

    for index, row in df.iterrows():
        value = row["value"]
        midpoints.append(value / 2 + cumulative_sum)
        cumulative_sum += value

    return midpoints

def generate_baseline_demographics():
    """
    Generate baseline demographics data and return DataFrames for gender, race, and age.

    Returns:
        pd.DataFrame: DataFrames for gender, race, and age demographics.
    """

    results_pth = "results/baseline_fairface_results.csv"
    imgs_path = "baseline_images/"

    if not os.listdir(imgs_path):
        #print("No files found in the folder. Doing nothing.")
        #Create empty placeholder daatframes for the plot to plot
        data_g = {
        "index": [],
        "variable": [],
        "value": [],
        "tag": []}
        data_r = {
        "index": [],
        "variable": [],
        "value": [],
        "tag": []}

        data_a = {
        "index": [],
        "variable": [],
        "value": [],
        "tag": []}

        df_g = pd.DataFrame(data_g)
        df_r = pd.DataFrame(data_r)
        df_a = pd.DataFrame(data_a)


    else:

      baseline_pred = predict_age_gender_race(save_prediction_at = results_pth, imgs_path = imgs_path)
      print("predicted the fair face distributions")
      df_a, df_r, df_g = sd_fairface_distribution(baseline_pred)

      # Convert values to percentages
      df_g["value"] = (df_g["%"] * 100).round(2)
      df_a["value"] = (df_a["%"] * 100).round(2)
      df_r["value"] = (df_r["%"] * 100).round(2)

      df_g["variable"] = df_g["label"]
      df_a["variable"] = df_a["label"]
      df_r["variable"] = df_r["label"]

      #add tags to all of the data
      df_g = process_data(df_g)
      df_a = process_data(df_a)
      df_r = process_data(df_r)

      #abbreviate tags
      update_tags(df_g)
      update_tags(df_r)
      update_tags(df_a)

    df_g["index"] = "gender"
    df_a["index"] = "age"
    df_r["index"] = "race"
    return df_g, df_r, df_a


def plot_baseline_demographics():
    """
    Generate and plot baseline demographics data for gender, race, and age.

    Returns:
        tuple: Gender, race, and age stacked bar plots.
    """

    df_g, df_r, df_a = generate_baseline_demographics()
    #3 separate plots
    gender_plot =  make_stacked_plot(df_g,"nominal", 'accent' )
    race_plot =  make_stacked_plot(df_r,"nominal",'tableau10' )
    age_plot =  make_stacked_plot(df_a,"nominal" ,'blues')

    return gender_plot, race_plot, age_plot

def interpolate_relative(df, slider):
    """
    Interpolate values in a DataFrame based on a slider factor.

    Args:
        df (pd.DataFrame): DataFrame with 'value' column to be interpolated.
        slider (float): Interpolation factor.

    Returns:
        pd.DataFrame: DataFrame with interpolated 'value' column.
    """

    df['parity'] = 100/len(df['label'])
    df["relative"] = df["value"] * (1 - slider) + df["parity"] * slider
    df["value"] = df["relative"]
    df["value"] = df["value"]/100
    print(df["value"])
    return df

def make_plots(radio_btn, rel_slider, selected_genders, selected_races, selected_ages):
  """
    Create stacked bar plots based on user-selected options.

    Args:
        radio_btn (str): Radio button selection.
        rel_slider (float): Relative slider value.
        selected_genders (list): Selected gender options.
        selected_races (list): Selected race options.
        selected_ages (list): Selected age options.

    Returns:
        tuple: Gender, race, and age stacked bar plots.
    """
  # Create the DataFrame
  if radio_btn == "Parity":
    data_g = {
        "index": ["gender", "gender"],
        "variable": ["Female", "Male"],
        "value": [0.5, 0.5],
        "tag": ["Female", "Male"]
    }
    data_r = {
            "index": ["race", "race","race","race","race","race","race"],
            "variable": ["Black", "East Asian", "Hispanic", "Indian","Middle Eastern", "Southeast Asian", "White"],
            "value": [0.1428,0.1428,0.1428,0.1428,0.1428,0.1428,0.1428 ],
            "tag": ["Black", "East Asian", "Hispanic", "Indian","Middle Eastern", "Southeast Asian", "White"],
        }

    data_a = {
            "index": ["age","age","age","age","age","age","age","age","age"],
            "variable": ["0-2","03-09","10-19", "20-29","30-39","40-49","50-59","60-69","70+"],
            "value": [0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111],
            "tag": ["0-2","03-09","10-19", "20-29","30-39","40-49","50-59","60-69","70+"]
        }

    df_g = pd.DataFrame(data_g)
    df_r = pd.DataFrame(data_r)
    df_a = pd.DataFrame(data_a)

    #abbreviate tags
    update_tags(df_g)
    update_tags(df_r)
    update_tags(df_a)

    # Convert values to percentages
    df_g["value"] = (df_g["value"] * 100).round(2)
    df_a["value"] = (df_a["value"] * 100).round(2)
    df_r["value"] = (df_r["value"] * 100).round(2)

    #add a column called mid_pts where to position the labels
    df_g['mid_pts'] = calculate_midpoints(df_g)
    df_r['mid_pts'] = calculate_midpoints(df_r)
    df_a['mid_pts'] = calculate_midpoints(df_a)

    #3 separate plots
    gender_plot =  make_stacked_plot(df_g,"nominal", 'accent' )
    race_plot =  make_stacked_plot(df_r,"nominal",'tableau10' )
    age_plot =  make_stacked_plot(df_a,"nominal" ,'blues')

    # Concatenate the DataFrames vertically

  elif radio_btn == "US demogr.":
    data_g = {
        "index": ["gender", "gender"],
        "variable": ["Female", "Male"],
        "value": [0.51, 0.49],
        "tag": ["Female", "Male"],
    }
    data_r = {
            "index": ["race", "race", "race","race","race","race","race",],
            "variable": ["Black", "East Asian", "Hispanic", "Indian","Middle Eastern", "Southeast Asian", "White"],
            "value": [0.13,0.03,0.19,0.03,0.03,0.03,0.56],
            "tag": ["Black", "East Asian", "Hispanic", "Indian","Middle Eastern", "Southeast Asian", "White"],
        }#https://www.census.gov/quickfacts/fact/table/US/AGE775222 : mid eastern are considered white in US census

    data_a = {
            "index": ["age","age","age","age","age","age","age","age","age"],
            "variable": ["0-2","03-09","10-19", "20-29","30-39","40-49","50-59","60-69","70+"],
            "value": [0.0339,0.074,0.1356,0.1350,0.1363,0.129,0.1273,0.1096,0.1193],
            "tag": ["0-2","03-09","10-19", "20-29","30-39","40-49","50-59","60-69","70+"],
        }#https://en.wikipedia.org/wiki/Demographics_of_the_United_States

    df_g = pd.DataFrame(data_g)
    df_r = pd.DataFrame(data_r)
    df_a = pd.DataFrame(data_a)

    #abbreviate tags
    update_tags(df_g)
    update_tags(df_r)
    update_tags(df_a)

    # Convert values to percentages
    df_g["value"] = (df_g["value"] * 100).round(2)
    df_a["value"] = (df_a["value"] * 100).round(2)
    df_r["value"] = (df_r["value"] * 100).round(2)

    #add a column called mid_pts where to position the labels
    df_g['mid_pts'] = calculate_midpoints(df_g)
    df_r['mid_pts'] = calculate_midpoints(df_r)
    df_a['mid_pts'] = calculate_midpoints(df_a)

    #3 separate plots
    gender_plot =  make_stacked_plot(df_g,"nominal", 'accent' )
    race_plot =  make_stacked_plot(df_r,"nominal",'tableau10' )
    age_plot =  make_stacked_plot(df_a,"nominal" ,'blues')

  elif (radio_btn == "Absolute cat." ):

    # Create data for gender, race and age and convert names to abbreviations

    genders = ["Male", "Female"]
    races = ["Black", "East Asian", "Hispanic", "Indian","Middle Eastern", "Southeast Asian", "White"]
    ages = ["0-2","03-09","10-19", "20-29","30-39","40-49","50-59","60-69","70+"]

    data_g = {
        "index": ["gender", "gender"],
        "variable": genders,
        "value": checkbox_selection_to_unary_list(selected_genders, genders)
    }

    # Create data for race
    data_r = {
        "index": ["race"] * len(races),
        "variable": races,
        "value": checkbox_selection_to_unary_list(selected_races, races)
    }

    # Create data for age
    data_a = {
        "index": ["age"] * len(ages),
        "variable": ages,
        "value": checkbox_selection_to_unary_list(selected_ages, ages)
    }

    #add tags to all of the data - NEEDS FIXING - TODO
    data_g = process_data(data_g)
    data_a = process_data(data_a)
    data_r = process_data(data_r)

    # Create DataFrames
    df_g = pd.DataFrame(data_g)
    df_r = pd.DataFrame(data_r)
    df_a = pd.DataFrame(data_a)

    #abbreviate tags
    update_tags(df_g)
    update_tags(df_r)
    update_tags(df_a)

    # Convert values to percentages
    df_g["value"] = (df_g["value"] * 100).round(2)
    df_a["value"] = (df_a["value"] * 100).round(2)
    df_r["value"] = (df_r["value"] * 100).round(2)

    #add a column called mid_pts where to position the labels
    df_g['mid_pts'] = calculate_midpoints(df_g)
    df_r['mid_pts'] = calculate_midpoints(df_r)
    df_a['mid_pts'] = calculate_midpoints(df_a)

    #3 separate plots
    gender_plot =  make_stacked_plot(df_g,"nominal", 'accent' )
    race_plot =  make_stacked_plot(df_r,"nominal",'tableau10' )
    age_plot =  make_stacked_plot(df_a,"nominal" ,'blues')

  elif (radio_btn == "Relative to the baseline"):
    #predict the basline categories and modify that based on the relative factor
    df_g, df_r, df_a =  generate_baseline_demographics()

    #if the dataframes are empty (aka the baseline has not run)
    #  if the baseline df are not empty, load those values in
    if (df_g['value'] != 0).any():
        print("There are non-zero values in the 'value' column.")
        df_g = interpolate_relative(df_g, rel_slider)
        df_r = interpolate_relative(df_r, rel_slider)
        df_a = interpolate_relative(df_a, rel_slider)

        df_g["value"] = (df_g["value"] * 100).round(2)
        df_a["value"] = (df_a["value"] * 100).round(2)
        df_r["value"] = (df_r["value"] * 100).round(2)

        df_g = process_data(df_g)
        df_a = process_data(df_a)
        df_r = process_data(df_r)

        update_tags(df_g)
        update_tags(df_r)
        update_tags(df_a)

    #otherwise put in empty plots

    gender_plot =  make_stacked_plot(df_g,"nominal", 'accent' )
    race_plot =  make_stacked_plot(df_r,"nominal",'tableau10' )
    age_plot =  make_stacked_plot(df_a,"nominal" ,'blues')

  return gender_plot, race_plot, age_plot




def load_imgs():
    """
    Load edited images from the 'edited_images' folder.

    Returns:
        list: A list of loaded edited images.
    """
    folder_path ="edited_images/"
    images = os.listdir(folder_path)

    generated_imgs = []
     # Sort the image names
    sorted_images = sorted(images)

    generated_imgs = []
    for image_name in sorted_images:
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path)
            generated_imgs.append(image)

    return generated_imgs