'''
Copyright (C) [2025] [Esteban Velasquez Toro]

    Estimated Cervical Rotation Point as An anchor (ECRP) GUI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

'''

import math
import os
import re

import multiprocessing
import socket
from turtledemo.penrose import start
from venv import create

import cv2
import mediapipe as mp
import time
import sys
import json

from decorator import append
from llama_cpp import Llama
import openai
from openai import OpenAI
import numpy as np
#from numpy.array_api import astype

import pynput
from pynput.keyboard import Key, Controller as keyController
from pynput.mouse import Button, Controller as mouseController
import pyttsx3
from sympy import false, trunc

#bugs: not working AI, Nonetype exception. Numbers dont work


#Virtual Keyboard
keyboard = keyController()# Controller()
mouse = mouseController()

#TTS engine
TTSengine = pyttsx3.init()

#video settings
imageOrVideo = True
imageCache={}
#------------------
#labelsMainMenu=["Quick", "BackSpace","LLM","Space","Mouse", "ABC","Numbers","Special Chars"]
labelsLettersMenu=["Back", "BackSpace"]
#------------------
#------------------
labelsMouseMenu=["Click","Back","D. Click","R. Click"]
labelsMouse=["Down","Left","Up","Right"]
#------------------
labelsQuickOptions=["LLM","BackSpace","Back"]
labelsLLMOptions=["Quick","BackSpace","Back"]
labelsLMM=["","","","","",]
#------------------
configFilePath='./config.txt'

#------------------
#menu selection init hidden variables
createdLabelsList=[]
selectionCurrentTime=0

#menu selection config file init variables

seedWord="Welcome "

#ui variables
selectorFrameColor = (0, 255, 0)  # BGR
ReferenceFrameColor = (255, 0, 0)  # BGR
variableSelectionSlotColor = (0, 0, 255)  # BGR
systemSelectionSlotColor = (0, 0, 255)  # BGR
highlightSlotColor=(100,100,255)#BGR

alpha = 0.3
font=cv2.FONT_HERSHEY_SIMPLEX


def speak(text,thettsRate, thettsVolume, thettsVoiceType,):
    engine = pyttsx3.init()

    engine.setProperty('rate', thettsRate)
    engine.setProperty('volume', thettsVolume)
    voices = engine.getProperty('voices')
    for i, voice in enumerate(voices):
        print(f"Voice {i}: {voice.name} - {voice.id}")
    if "female" in thettsVoiceType:
        desiredVoiceId = 1
    else:
        desiredVoiceId = 0

    engine.setProperty('voice', voices[desiredVoiceId].id)

    engine.say(text)
    engine.runAndWait()

def speak_non_blocking(text,thettsRate, thettsVolume, thettsVoiceType,):
    p = multiprocessing.Process(target=speak, args=(text,thettsRate, thettsVolume, thettsVoiceType,))
    p.start()
    return p  # Return the process in case you want to track or terminate it

def is_connected():
    try:
        # Connect to a well-known host to check internet access
        socket.create_connection(("www.google.com", 80), timeout=5)
        return True
    except OSError:
        return False

def getAllGuffModels(directory):
    # Create a list to store file paths and sizes
    files_with_sizes = []
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gguf'):
                file_path = os.path.join(root, file)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                files_with_sizes.append((file_path, file_size_mb))
    # Sort the list by file size (smallest first)
    sorted_files = sorted(files_with_sizes, key=lambda x: x[1])
    # Extract only the file paths
    sorted_file_paths = [file_path for file_path, _ in sorted_files]
    return sorted_file_paths

def loadLLM(thePath,contextSize=512,batchSize=126):
    sortedPaths=getAllGuffModels("./")
    print(f"sortedPaths: {sortedPaths}")
    theLLM = Llama(model_path=sortedPaths[0], n_ctx=contextSize, n_batch=batchSize,n_gpu_layers=30, use_gpu=True)
    return theLLM

def generate_text(
    llm,
    prompt="What are the five common emotions?",
    max_tokens=24,
    temperature=0.1,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output["choices"][0]["text"].strip()
    return output_text

def changeLlmWaitTime(theTime):
    # Read all lines from the file
    timeLine=f"llmWaitTime={theTime}\t\t\t\t\t\t\t\t\t###Average time per call to LLM service"

    with open(configFilePath, 'r') as file:
        lines = file.readlines()

    # Replace the last line if the file is not empty
    if lines:
        lines[-1] = timeLine + '\n'  # Add a newline at the end

    # Write the modified lines back to the file
    with open(configFilePath, 'w') as file:
        file.writelines(lines)

def generate_prompt_from_template(input):
    chat_prompt_template = f"""<|im_start|>system
You are a helpful chatbot.<|im_end|>
<|im_start|>user
{input}<|im_end|>"""
    return chat_prompt_template

def GetConfigSettings():
    selectionType='BackToCenter'
    timeOnLocation=3
    ttsRate = 150  # modify as needed, I prefer a default value of 150 wpm
    ttsVolume = 0.9  # modify as needed, I prefer a default value of 0.9
    ttsVoiceType = 'Female'  # Possible options are: Male, Female
    ignoreGuiAngles = 270  # list of angles not to be used
    ignoreAngleArc = 30
    centerSizePercentageX=20
    centerSizePercentageY=20
    offsetPercentageX = 0
    offsetPercentageY = -20
    totalOptionsN = 3
    mouseSpeed = 5
    selectionWaitTime = 0.4
    labelsMainMenu = ["Quick", "BackSpace", "LLM", "Space", "Mouse", "ABC", "Numbers", "Special Chars"]
    labelsABC = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelsNumbers="0123456789"
    labelsSpecial="`~!@#$%^&*()-_\"+[]\;',./<>?:{}="
    labelsQuick = ["Yes", "No", "Not Sure", "Food", "Bathroom", "Hot", "Cold", "Hurts"]
    fontScale = 0.4
    fontThickness = 1
    camSizeX = 640
    camSizeY = 640
    showFPS =False
    showWritten=False
    llmContextSize=512
    llmBatchSize=126
    llmWaitTime=0
    maxWhatsWrittenSize=20
    showWrittenMode="Single"
    seedWord="Emotions"
    LlmService="local"
    LlmKey=""
    global selectorFrameColor,ReferenceFrameColor,variableSelectionSlotColor,systemSelectionSlotColor,highlightSlotColor
    selectorFrameColor = (0, 255, 0)  # BGR
    ReferenceFrameColor = (255, 0, 0)  # BGR
    variableSelectionSlotColor = (0, 0, 255)  # BGR
    systemSelectionSlotColor = (0, 0, 255)  # BGR
    highlightSlotColor = (100, 100, 255)  # BGR


    with open(configFilePath, 'r') as file:
        for line in file:
            # Strip whitespace and check if the line starts with #
            line = line.strip()
            if line.startswith('#') or not line:
                continue  # Skip comments and empty lines

            key, value = line.strip().split('=',1)
            value = value.split('###', 1)[0].strip()

            if key == "selectorFrameColor":#BGR
                RGB =value.split(',')
                selectorFrameColor = (int(RGB[2]),int(RGB[1]), int(RGB[0]))  # BGR
            elif key == "ReferenceFrameColor":
                RGB = value.split(',')
                ReferenceFrameColor = (int(RGB[2]),int(RGB[1]), int(RGB[0]))  # BGR
            elif key == "variableSelectionSlotColor":
                RGB = value.split(',')
                variableSelectionSlotColor = (int(RGB[2]),int(RGB[1]), int(RGB[0]))  # BGR
            elif key == "systemSelectionSlotColor":
                RGB = value.split(',')
                systemSelectionSlotColor = (int(RGB[2]),int(RGB[1]), int(RGB[0]))  # BGR
            elif key == "highlightSlotColor":
                RGB = value.split(',')
                highlightSlotColor = (int(RGB[2]),int(RGB[1]), int(RGB[0]))  # BGR
            elif key == "selectionType":
                selectionType = value
            elif key == "ignoreGuiAngles":
                ignoreGuiAngles = value.split(',')
            elif key == "ignoreAngleArc":
                ignoreAngleArc = float(value)
            elif key == "ttsRate":
                ttsRate = float(value)
            elif key == "ttsVolume":
                ttsVolume = float(value)
            elif key == "ttsVoiceType":
                ttsVoiceType = str(value).lower()
            elif key == "timeOnLocation":
                timeOnLocation = float(value)
            elif key == "centerSizePercentageX":
                centerSizePercentageX = int(value)
            elif key == "centerSizePercentageY":
                centerSizePercentageY = int(value)
            elif key == "offsetPercentageX":
                offsetPercentageX = int(value)
            elif key == "offsetPercentageY":
                offsetPercentageY = int(value)
            elif key == "totalOptionsN":
                totalOptionsN = int(value)
            elif key == "mouseSpeed":
                mouseSpeed = int(value)
            elif key == "selectionWaitTime":
                selectionWaitTime = float(value)
            elif key == "labelsMainMenu":
                labelsMainMenu = value.split(',')
            elif key == "labelsABC":
                labelsABC = value
            elif key == "labelsNumbers":
                labelsNumbers = value
            elif key == "labelsSpecial":
                labelsSpecial = value
            elif key == "labelsQuick":
               labelsQuick = value.split(',')
            elif key == "fontScale":
                fontScale = float(value)
            elif key == "fontThickness":
                fontThickness = int(value)
            elif key == "camSizeX":
                camSizeX = int(value)
            elif key == "camSizeY":
                camSizeY = int(value)
            elif key == "showFPS":
                if value == "False" or value == "False" or value == "0":
                    value=""
                showFPS = bool(value)
            elif key == "showWritten":
                if value == "False" or value == "false" or value == "0"or value == "f" or value == "F":
                    value=""
                showWritten = bool(value)
            elif key == "llmContextSize":
                llmContextSize = int(value)
            elif key == "llmBatchSize":
                llmBatchSize = int(value)
            elif key == "llmWaitTime":
                llmWaitTime = int(value)
            elif key == "maxWhatsWrittenSize":
                maxWhatsWrittenSize = int(value)
            elif key == "showWrittenMode":
                showWrittenMode = value
            elif key == "SeedWord":
                SeedWord = value
            elif key == "LlmService":
                LlmService = value
            elif key == "LlmKey":
                LlmKey = value



    # Print the variables
    print(f"selectionType: {selectionType} \n"
          f", timeOnLocation: {timeOnLocation} \n"
          f", Colors: ({selectorFrameColor}), ({ReferenceFrameColor}),({variableSelectionSlotColor}),({systemSelectionSlotColor}),({highlightSlotColor}) \n"
          f", ignoreGuiAngles,ignoreAngleArc: ({ignoreGuiAngles}),{ignoreAngleArc}"
          f", ttsRate,ttsVolume,ttsVoiceType: ({ttsRate},{ttsVolume},{ttsVoiceType})"
          f", offsetPercentage: ({offsetPercentageX},{offsetPercentageY}),\n"
          f", totalOptionsN: {totalOptionsN}, mouseSpeed: {mouseSpeed}, selectionWaitTime: {selectionWaitTime}\n"
          f", labelsMainMenu: {labelsMainMenu}\n"
          f", labelsABC: {labelsABC}, labelsQuick: {labelsQuick},\n"
          f", fontScale: {fontScale}, fontThickness: {fontThickness}, camSizeX: {camSizeX}, camSizeY: {camSizeY}\n"
          f", showFPS: {showFPS}, showWritten: {showWritten}\n"
          f", llmContextSize: {llmContextSize}, llmBatchSize: {llmBatchSize}, llmWaitTime: {llmWaitTime}\n"
          f", maxWhatsWrittenSize: {maxWhatsWrittenSize}, showWrittenMode: {showWrittenMode}\n"
          f", seedWord: {seedWord}, LlmService: {LlmService}, LlmKey: {LlmKey}\n")
    return (selectionType,timeOnLocation,ignoreGuiAngles,ignoreAngleArc,ttsRate,ttsVolume,ttsVoiceType,
            centerSizePercentageX,centerSizePercentageY,offsetPercentageX,offsetPercentageY,
            totalOptionsN,mouseSpeed,selectionWaitTime,labelsMainMenu,labelsABC,labelsNumbers,labelsSpecial,labelsQuick,
            fontScale,fontThickness,camSizeX,camSizeY, showFPS, showWritten,llmContextSize,llmBatchSize,llmWaitTime,
            maxWhatsWrittenSize,showWrittenMode,seedWord,LlmService,LlmKey,
            selectorFrameColor,ReferenceFrameColor,variableSelectionSlotColor,systemSelectionSlotColor,highlightSlotColor)

def GetAreaPoints(totalN, centerOfFaceX, centerOfFaceY, areaSize, theignoreGuiAngles, theignoreAngleArc, offsetAngleDeg=0):
    # Ensure input is list
    if not isinstance(theignoreGuiAngles, list):
        theignoreGuiAngles = [theignoreGuiAngles]

    adjusted_ignore_angles = [(float(angle) - offsetAngleDeg) % 360 for angle in theignoreGuiAngles]

    # Step 1: Build ignore arcs and normalize with adjusted angles
    ignore_arcs = []
    for angle in adjusted_ignore_angles:
        center = angle % 360
        half_arc = theignoreAngleArc / 2
        start = (center - half_arc) % 360
        end = (center + half_arc) % 360
        ignore_arcs.append((start, end))

    # Normalize to 0–360 with wraparound handling
    normalized_arcs = []
    for start, end in ignore_arcs:
        start %= 360
        end %= 360
        if start < end:
            normalized_arcs.append((start, end))
        else:
            # handle wrap-around
            normalized_arcs.append((start, 360))
            normalized_arcs.append((0, end))

    # Sort and merge overlapping arcs
    normalized_arcs.sort()
    merged_arcs = []
    for arc in normalized_arcs:
        if not merged_arcs:
            merged_arcs.append(arc)
        else:
            last_start, last_end = merged_arcs[-1]
            current_start, current_end = arc
            if current_start <= last_end:
                merged_arcs[-1] = (last_start, max(last_end, current_end))
            else:
                merged_arcs.append(arc)

    # Step 2: Get available arcs
    available_arcs = []
    last_end = 0
    for arc_start, arc_end in merged_arcs:
        if arc_start > last_end:
            available_arcs.append((last_end, arc_start))
        last_end = arc_end
    if last_end < 360:
        available_arcs.append((last_end, 360))

    # Step 3: Place segments along available arcs (remaining code remains the same)
    #total_available_degrees = sum(end - start for start, end in available_arcs)
    total_available_degrees=0
    for arc in available_arcs:
        total_available_degrees=total_available_degrees + (arc[1]-arc[0])
    #print(f"total available Degrees: {total_available_degrees}")
    degrees_per_segment = total_available_degrees / totalN

    contours = []
    centerOfContours = []
    rotationAngle = total_available_degrees / (totalN * 2)
    segments_drawn = 0

    def rotateSelectionSlot(x, y, angle_deg):
        angle_rad = math.radians(angle_deg)
        dx = x - centerOfFaceX
        dy = y - centerOfFaceY
        rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + centerOfFaceX
        ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + centerOfFaceY
        return rx, ry

    #print(f"available_arcs: {available_arcs}")
    for arc_start, arc_end in available_arcs:
        arc_length = arc_end - arc_start
        num_segments = int(round(arc_length / degrees_per_segment))

        for j in range(num_segments):
            if segments_drawn >= totalN:
                break

            angle1_deg = (arc_start + j * degrees_per_segment)
            angle2_deg = (arc_start + (j + 1) * degrees_per_segment)

            angle1_rad = math.radians(angle1_deg)
            angle2_rad = math.radians(angle2_deg)

            x1 = centerOfFaceX + math.cos(angle1_rad) * areaSize
            y1 = centerOfFaceY + math.sin(angle1_rad) * areaSize
            x2 = centerOfFaceX + math.cos(angle2_rad) * areaSize
            y2 = centerOfFaceY + math.sin(angle2_rad) * areaSize

            rotatedX1, rotatedY1 = rotateSelectionSlot(x1, y1, offsetAngleDeg)
            rotatedX2, rotatedY2 = rotateSelectionSlot(x2, y2, offsetAngleDeg)

            points = [[centerOfFaceX, centerOfFaceY], [rotatedX1, rotatedY1], [rotatedX2, rotatedY2]]
            ctr = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
            contours.append(ctr)

            centerAngleDeg = (angle1_deg + angle2_deg) / 2
            centerAngleRad = math.radians(centerAngleDeg)
            cx = centerOfFaceX + math.cos(centerAngleRad) * areaSize
            cy = centerOfFaceY + math.sin(centerAngleRad) * areaSize
            rcx, rcy = rotateSelectionSlot(cx, cy, offsetAngleDeg)

            centerOfContours.append((int(rcx), int(rcy)))

            segments_drawn += 1

    return contours, centerOfContours

def GetFaceSizeAndECRP(shape, landMarks):
    faceXmin = shape[1]
    faceXmax = 0
    faceYmin = shape[0]
    faceYmax = 0
    for counter, theLandMark in enumerate(landMarks):
        # -------------------------
        # Landmark Recognition-----
        # -------------------------
        x = theLandMark.x
        y = theLandMark.y
        if x < faceXmin:
            faceXmin = x
        if x > faceXmax:
            faceXmax = x
        if y < faceYmin:
            faceYmin = y
        if y > faceYmax:
            faceYmax = y
    centerOfFaceX = int(shape[1] * (faceXmin + ((faceXmax - faceXmin) / 2)))
    centerOfFaceY = int(shape[0] * (faceYmin + ((faceYmax - faceYmin) / 2)))
    return faceXmin,faceXmax,faceYmin,faceYmax, centerOfFaceX,centerOfFaceY

def read_NVC_structure(root_dir):
    folder_structure = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get relative path from root
        rel_path = os.path.relpath(dirpath, root_dir)
        rel_path = '.' if rel_path == '.' else rel_path.replace('\\', '/')

        folder_structure[rel_path] = {
            'folders': sorted(dirnames),
            'files': sorted(filenames)
        }

    return folder_structure

def is_point_inside_ellipse(point, center, radius_x, radius_y):
    x, y = point
    cx, cy = center
    dx = (x - cx) / radius_x
    dy = (y - cy) / radius_y
    return dx**2 + dy**2 < 1

def GetGUI(theUIFrame,radiusAsPercentX,radiusAsPercentY,totalN,centerFaceX,centerFaceY,nosePosition,theignoreGuiAngles,theignoreAngleArc):
    theContours, centerContours = GetAreaPoints(totalN, centerFaceX, centerFaceY,100,theignoreGuiAngles,theignoreAngleArc)  # area number, total areas
    # set center of face

    for i in range(totalN):
        #for color purposes
        ECRP_radius= abs(nosePosition[0]-centerFaceX)
        if (cv2.pointPolygonTest(theContours[i], nosePosition, False)>=0) \
                and not is_point_inside_ellipse(nosePosition,[centerFaceX,centerFaceY],radiusAsPercentX,radiusAsPercentY):
            cv2.fillPoly(theUIFrame, [theContours[i]], highlightSlotColor)
        elif i %2 ==0:
            cv2.fillPoly(theUIFrame, [theContours[i]], [250, 250, 250])
        else:
            cv2.fillPoly(theUIFrame, [theContours[i]], [150, 150, 150])
    #cv2.circle(theUIFrame, (centerFaceX, centerFaceY), radiusAsPercent, blueFrameColor, -1)
    cv2.ellipse(theUIFrame, (centerFaceX, centerFaceY), (radiusAsPercentX, radiusAsPercentY), 0, 0, 360,
                selectorFrameColor, -1)
    polyEllipse = cv2.ellipse2Poly((centerFaceX, centerFaceY), (radiusAsPercentX, radiusAsPercentY),
                                   0, 0, 360, 1)
    # set center of nose as a controller
    cv2.circle(theUIFrame, nosePosition, int(radiusAsPercentX * 0.25), variableSelectionSlotColor, -1)

    return polyEllipse,theContours,centerContours

def getLLM(queue,theLlmService,LlmKey,lastWord,theSeedword,thetotalOptionsN,thellmContextSize, thellmBatchSize):
    if lastWord is None or lastWord == "":
        prompt = f"Give me a list of only {thetotalOptionsN} words with no explanation that go after: \"{theSeedword}\""
    else:
        prompt = f"Give me a list of only {thetotalOptionsN} words with no explanation that go after: \"{lastWord}\""
    # print(f"prompt: {prompt}")

    if theLlmService=="ChatGPT" and is_connected():
        #print("Calling ChatGPT: ")
        try:
            client = OpenAI(api_key=LlmKey)
            session = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            response = session.choices[0].message.content
            print (f"response: \n {response}")
            reply = re.sub(r'\d+\.?\s*', '', response) #r means raw string, \d matches digits, \.? matches literal dot, ? means optional, \s matches whitespace
            result=reply.splitlines()
        except Exception as e:
            print(f"An error occurred: {e}")
            try:
                # Try fetching available models to verify the API key
                response = client.models.list()
                print("API key is valid! ✅")
            except openai.AuthenticationError:
                print("Invalid API key. ❌")
            except Exception as e:
                print(f"An error occurred: {e}")

            print(f"Changing mode to local LLM")
            theLlmService="Local"
            prompt = generate_prompt_from_template(prompt)
            print("calling local LLM: ")
            llm = loadLLM("zephyr-7b-beta.Q4_K_M.gguf", thellmContextSize, thellmBatchSize)
            generatedText = generate_text(llm, prompt)
            generatedText = re.sub(r'\d+\.?\s*', '', generatedText)
            # print(f"generated Text: {generatedText}")
            result = generatedText.splitlines()
            # print(f"generated Text: {thelabelsLMM}")
    else:
        prompt = generate_prompt_from_template(prompt)
        print("calling local LLM: ")
        llm = loadLLM("zephyr-7b-beta.Q4_K_M.gguf", thellmContextSize, thellmBatchSize)
        generatedText = generate_text(llm, prompt)
        generatedText = re.sub(r'\d+\.?\s*', '', generatedText)
        #print(f"generated Text: {generatedText}")
        result = generatedText.splitlines()
        #print(f"generated Text: {thelabelsLMM}")
    queue.put(result)

def determineMaxWrittenSize(thescreenWidth, thewhatsWritten,thefont,thefontScale,thefontThickness,themaxWhatsWrittenSize):

    #do not count words with jpg, png or jpeg in them
    valid_extensions = ('.jpg', '.png', '.jpeg')
    words = thewhatsWritten.split()
    totalCharsToIgnore = sum(len(word) for word in words if word.lower().endswith(valid_extensions))
    themaxWhatsWrittenSize = themaxWhatsWrittenSize + totalCharsToIgnore
    (text_width, text_height), baseline = cv2.getTextSize(thewhatsWritten, thefont, thefontScale, thefontThickness)
    oneCharTextWidthPxls = math.ceil(text_width / len(thewhatsWritten))  # average char size in pixels
    max_chars = math.floor(thescreenWidth / oneCharTextWidthPxls)
    # show up until max numbers of characters
    if max_chars < themaxWhatsWrittenSize:
        themaxWhatsWrittenSize = int(max_chars)
    #if max_chars<len(thewhatsWritten):
    thewhatsWritten=thewhatsWritten[-themaxWhatsWrittenSize:]
    (text_width, text_height), baseline = cv2.getTextSize(thewhatsWritten, thefont, thefontScale, thefontThickness)
    #print(f"max chars: {max_chars}, len(whatsWritten): {len(thewhatsWritten)}, whats written: \"{thewhatsWritten}\"")
    return (text_width, text_height), baseline,themaxWhatsWrittenSize,thewhatsWritten,oneCharTextWidthPxls

from typing import List, Tuple
def replace_and_track_indices(original: str,pattern: str,replacement: str,flags=0
) -> Tuple[str, List[Tuple[int, int, str]]]:#type hint
    """
    Replaces all matches of `pattern` in `original` with `replacement`,
    and returns the new string along with the indices and matched strings
    of the replacements in the new string.

    Args:
        original (str): The input string.
        pattern (str): Regex pattern to find.
        replacement (str): Replacement string.
        flags (int): Regex flags (e.g., re.IGNORECASE).

    Returns:
        Tuple[str, List[Tuple[int, int, str]]]:
            - Modified string
            - List of (start_index, end_index, original_match) in the new string
    """
    matches = list(re.finditer(pattern, original, flags=flags))
    result: List[Tuple[int, int, str]] = []
    new_string = ''
    last_end = 0

    for match in matches:
        start, end = match.span()
        matched_text = match.group()
        new_string += original[last_end:start] + replacement
        replace_start = len(new_string) - len(replacement)
        replace_end = len(new_string)
        result.append((replace_start, replace_end, matched_text))
        last_end = end

    new_string += original[last_end:]
    return new_string, result


def showWhatsWritten(theTopFrame,dimensionsTop,thewhatsWritten, thefont,thefontScale,thefontThickness,theshowWrittenMode,themaxWhatsWrittenSize):
    #start options
    percentageTakenByWritten = .1 #10% of screen taken by secondary screen
    top_height = int(dimensionsTop[0] * percentageTakenByWritten)
    # splitIntoHalves
    top_half = theTopFrame[:top_height, :]
    bottom_half = theTopFrame[top_height:, :]
    topHalfSize = top_half.shape
    (text_width, text_height), baseline,themaxWhatsWrittenSize,thewhatsWritten,oneCharTextWidthPxls = determineMaxWrittenSize(topHalfSize[1],thewhatsWritten, thefont,thefontScale, thefontThickness,themaxWhatsWrittenSize)

    # put black background
    top_half = np.zeros((topHalfSize[0], topHalfSize[1], 3), dtype=np.uint8)

    if "mirror" in theshowWrittenMode.lower():
        left_half = top_half[:, :topHalfSize[1] // 2]  # Left half
        right_half = top_half[:, topHalfSize[1] // 2:]  # Right half
        sideHalfSize = left_half.shape
        #(text_width, text_height), baseline,themaxWhatsWrittenSize,thewhatsWritten,oneCharTextWidthPxls = determineMaxWrittenSize(sideHalfSize[1],thewhatsWritten, thefont,thefontScale, thefontThickness,themaxWhatsWrittenSize)
        # put white text on black background
        if theshowWrittenMode.lower()=="CloneAndMirror".lower():
            cv2.putText(right_half, thewhatsWritten,
                        (int((sideHalfSize[1] / 2) - text_width / 2), text_height + 20), thefont,
                        thefontScale, (255, 255, 255), thefontThickness, cv2.LINE_AA)
            cv2.putText(left_half, thewhatsWritten,
                        (int((sideHalfSize[1] / 2) - text_width / 2), text_height + 20), thefont,
                        thefontScale, (255, 255, 255), thefontThickness, cv2.LINE_AA)
            # Create a mirrored version of the image
            mirrored_right_half = cv2.flip(right_half, 1)
            combined_TopHalf = cv2.hconcat([left_half, mirrored_right_half])
            theTopFrame = cv2.vconcat([combined_TopHalf, bottom_half])
        elif theshowWrittenMode.lower()=="CloneAndFlipMirror".lower():
            cv2.putText(right_half, thewhatsWritten,
                        (int((sideHalfSize[1] / 2) - text_width / 2), text_height + 20), thefont,
                        thefontScale, (255, 255, 255), thefontThickness, cv2.LINE_AA)
            cv2.putText(left_half, thewhatsWritten,
                        (int((sideHalfSize[1] / 2) - text_width / 2), text_height + 20), thefont,
                        thefontScale, (255, 255, 255), thefontThickness, cv2.LINE_AA)
            # Create a mirrored version of the image
            mirrored_right_half = cv2.flip(right_half, 1)
            mirrored_right_half = cv2.flip(mirrored_right_half, 0)
            combined_TopHalf = cv2.hconcat([left_half, mirrored_right_half])
            theTopFrame = cv2.vconcat([combined_TopHalf, bottom_half])
        elif theshowWrittenMode.lower()=="Mirror".lower():
            cv2.putText(top_half, thewhatsWritten,
                        (int((dimensionsTop[1] / 2) - text_width / 2), text_height + 20), thefont,
                    thefontScale, (255, 255, 255), thefontThickness, cv2.LINE_AA)
            # Create a mirrored version of the image
            mirrored_half = cv2.flip(top_half, 1)

            theTopFrame = cv2.vconcat([mirrored_half, bottom_half])
        else:
            #put white text on black background
            mirrored_top_half = cv2.flip(top_half, 1)
            theTopFrame = cv2.vconcat([mirrored_top_half, bottom_half])
    elif theshowWrittenMode.lower()=="Single".lower():
        # put white text on black background


        pixelsPerImage=math.ceil(100/oneCharTextWidthPxls)+20#get image size. 100 pixels per image for now
        charsPerImage=math.ceil(pixelsPerImage/oneCharTextWidthPxls)

        spaceOfImage=" "*(charsPerImage+2)
        #print( f"oneCharTextWidthPxls, pixPerImage, charsPerImage, len(spaceOfImage): ({oneCharTextWidthPxls}, {pixelsPerImage}, {charsPerImage}, {len(spaceOfImage)})")
        (text_width, text_height), baseline, themaxWhatsWrittenSize, thewhatsWritten, oneCharTextWidthPxls = determineMaxWrittenSize(
            topHalfSize[1], thewhatsWritten, thefont, thefontScale, thefontThickness, themaxWhatsWrittenSize)


        modifiedToWrite = thewhatsWritten

        #matches = list(re.finditer(r'\S*\.(jpg|jpeg|png)', modifiedToWrite, flags=re.IGNORECASE))
        modifiedToWrite, replacement_info = replace_and_track_indices(modifiedToWrite, r'\S*\.(jpg|jpeg|png)', spaceOfImage, flags=re.IGNORECASE)
        (text_width,text_height), baseline, themaxWhatsWrittenSize, modifiedToWrite, oneCharTextWidthPxls = determineMaxWrittenSize(
            topHalfSize[1], modifiedToWrite, thefont, thefontScale, thefontThickness, themaxWhatsWrittenSize)
        textX = int((topHalfSize[1] / 2) - (text_width / 2))
        textY = int(text_height + 20)
        theOrg = (textX, textY)
        # Put text
        cv2.putText(top_half, modifiedToWrite,
                    theOrg, thefont,
                    thefontScale, (255, 255, 255), thefontThickness, cv2.LINE_AA)

        # put image
        sumPrevSizes=0
        for matchesCounted, match in enumerate(replacement_info):
            matched_text = match[2]
            startIndex = match[0]+1
            endIndex = match[1]
            image_key = matched_text  # example:'Expressive/Surprised.png'
            image_key = image_key.lstrip("./")
            if image_key in imageCache:
                overlay_img = imageCache[image_key]
                if overlay_img is not None:
                    substring = matched_text[:startIndex]
                    (substring_size, _) = cv2.getTextSize(substring, font, thefontScale, thefontThickness)
                    imageX = math.floor((textX + substring_size[0]))
                    imageY = int(topHalfSize[2]/2)
                    top_half = overlay_image(top_half, overlay_img, x=imageX,
                                                y=imageY)
                    sumPrevSizes = endIndex#sumPrevSizes + (endIndex - startIndex)
                    #print(f"Match:{matches_counted}, key: {image_key}, (x,y)=({imageX}, {imageY}), startIndex: {startIndex}, endIndex:{endIndex}, sumPrevSizes:{sumPrevSizes}")
                    #print(f"Match:{matches_counted}, key: {image_key}, (x,y)=({imageX}, {imageY})")
            else:#part of a key, or something else is happening
                theReplacedSpaces=" "*(len(image_key))
                thewhatsWritten.replace(image_key,theReplacedSpaces)#""




        # Create a mirrored version of the image
        # mirrored_top_half = cv2.flip(top_half, 1)
        theTopFrame = cv2.vconcat([top_half, bottom_half])
    #else:#dont Show anything

    return theTopFrame,thewhatsWritten

def definePagination(selectedSelectionSlot,paginationIndex,totalN,labelsMenu):
    if selectedSelectionSlot == "More Options".lower():  # More Options
        theCurrentSelection = [-1, "MainMenu"]
        paginationIndex = paginationIndex + 1
        if (paginationIndex * totalN - (2 * paginationIndex)) > len(labelsMenu):
            paginationIndex = 0
        print(f"paginatingMenu: {selectedSelectionSlot}, pagination: {paginationIndex}")
    elif selectedSelectionSlot == "Previous Options".lower():  # More Options
        theCurrentSelection = [-1, "MainMenu"]
        paginationIndex = paginationIndex - 1
        if paginationIndex < 0:
            paginationIndex = len(labelsMenu) / (totalN - 2)  # math.floor((len(labelsMainMenu)*2)/totalN)
            if paginationIndex.is_integer():
                paginationIndex = int(paginationIndex - 1)
            else:
                paginationIndex = math.floor(paginationIndex)
    return paginationIndex

def GetMenuSystem(queue, theTopFrame, totalN, theCurrentSelection,theprevSelection,
                  theCreatedLabelList,theOriginalCreatedLabelList, theprevCreatedLabelsList, centerOfContours, systemSelectorColor,
                  variableSelectionSlotColor, dimensionsTop, theshowFPS, thestartTime, thelastWord, theprevLastWord, thellmIsWorkingFlag,
                  theLlmService, theLlmKey, thellmWaitTime, thefont, thefontScale, thefontThickness, theredFrameColor,
                  thewhatsWritten, thefps, thelabelsLMM, theSeedWord, thelabelsQuick, thelabelsABC, labelsMainMenu, thelabelsNumbers, thelabelsSpecial,
                  themouseSpeed, theshowWritten, theshowWrittenMode, themaxWhatsWrittenSize, thetotalOptionsN, paginationIndex,
                  thellmContextSize, thellmBatchSize, prettyPrintRects, centerOfFace, nvcStructure,currentNVCPath,flagSameSelection):
    if theshowFPS:
        cv2.putText(theTopFrame, "FPS: " + str(int(thefps)), org=(int(dimensionsTop[0] / 2), 20),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
    endTime = time.time()
    totalTime = math.ceil(endTime - thestartTime)

    if thellmIsWorkingFlag:
        if thellmWaitTime<=0:
            warningText = f"LLM: \"{thelastWord}\",Calculating seconds left for next call"
        else:
            warningText=f"LLM: \"{thelastWord}\",aprox {int(totalTime/thellmWaitTime*100)}% done, {int(thellmWaitTime-totalTime)} seconds left"
        #print(dimensionsTop)
        thelabelsLMM = ["", "", "", "", ""]
        (text_width, text_height), baseline = cv2.getTextSize(warningText, thefont, thefontScale, thefontThickness)
        topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, warningText,(int(dimensionsTop[1] / 2),dimensionsTop[0]+int(text_height*0.4)),
                            theredFrameColor,thefontScale,thefontThickness,prettyPrintRects,centerOfFace,onCenter=True)#x and y are inverted in call

    if theshowWritten:
        theTopFrame,thewhatsWritten = showWhatsWritten(theTopFrame,dimensionsTop,thewhatsWritten,thefont,thefontScale,thefontThickness,theshowWrittenMode,themaxWhatsWrittenSize)

    #--------------------------
    # Menu System--------------
    #--------------------------
    if theCurrentSelection[0] < 0:#no option has been chosen
        if theCurrentSelection[1]=="MainMenu":
            GetMainMenu(totalN, theTopFrame, theCurrentSelection, centerOfContours, systemSelectorColor, variableSelectionSlotColor, thefontScale, thefontThickness, thelabelsNumbers, thelabelsSpecial, thelabelsABC, labelsMainMenu, prettyPrintRects, centerOfFace, paginationIndex)
        elif(theCurrentSelection[1]=="MultipleLetters"):
            GetCharacterDivisionMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection, centerOfContours, systemSelectorColor, variableSelectionSlotColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace)
        elif (theCurrentSelection[1] == "MultipleNumbers"):
            GetCharacterDivisionMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection, centerOfContours, systemSelectorColor, variableSelectionSlotColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace)
        elif (theCurrentSelection[1] == "MultipleSpecialChars"):
            GetCharacterDivisionMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection, centerOfContours, systemSelectorColor, variableSelectionSlotColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace)
        elif(theCurrentSelection[1]=="Quick"):
            DisplayOtherMenus(thelabelsQuick, labelsQuickOptions, totalN, theTopFrame, centerOfContours, systemSelectorColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace,paginationIndex)
        elif (theCurrentSelection[1] == "MouseControl"):
            DisplayMouseMenu(labelsMouse, labelsMouseMenu, totalN, theTopFrame, centerOfContours, systemSelectorColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace,paginationIndex)
            theCurrentSelection[0] = -1
        elif (theCurrentSelection[1] == "LLM"):
            DisplayOtherMenus(thelabelsLMM, labelsLLMOptions, totalN, theTopFrame, centerOfContours, systemSelectorColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace,paginationIndex)
        elif theCurrentSelection[1].lower() == "More Options".lower():  # More Options
            #print(f"Menu: {theCurrentSelection[1]}")
            GetMainMenu(totalN, theTopFrame, theCurrentSelection, centerOfContours, systemSelectorColor, variableSelectionSlotColor, thefontScale,
                        thefontThickness, thelabelsNumbers, thelabelsSpecial, thelabelsABC, labelsMainMenu,
                        prettyPrintRects, centerOfFace, paginationIndex)
        elif theCurrentSelection[1].lower() == "Previous Options".lower():  # More Options
            #print(f"Menu: {theCurrentSelection[1]}")
            GetMainMenu(totalN, theTopFrame, theCurrentSelection, centerOfContours, systemSelectorColor, variableSelectionSlotColor, thefontScale,
                        thefontThickness, thelabelsNumbers, thelabelsSpecial, thelabelsABC, labelsMainMenu,
                        prettyPrintRects, centerOfFace, paginationIndex)
        elif theCurrentSelection[1] == "NVC":  # Non Verbal Communication: Emojis, custom images, etc!
            #print(f"No selection Menu: {theCurrentSelection[1]}")
            GetNVCmenu(theCreatedLabelList,theOriginalCreatedLabelList,totalN, theTopFrame, centerOfContours, systemSelectorColor, variableSelectionSlotColor, thefontScale,
                        thefontThickness, prettyPrintRects, centerOfFace, paginationIndex,nvcStructure,currentNVCPath)
    else:  # if an option has been chosen
        if(theCurrentSelection[1]=="MainMenu"):#chosen from main menu
            theprevSelection=[-1,"MainMenu"]
            theCreatedLabelList,prettyPrintRects,paginationIndex=GetMainMenu(totalN, theTopFrame, theCurrentSelection, centerOfContours, systemSelectorColor, variableSelectionSlotColor, thefontScale, thefontThickness,
                                                                             thelabelsNumbers, thelabelsSpecial, thelabelsABC, labelsMainMenu, prettyPrintRects, centerOfFace, paginationIndex)
            selectedSelectionSlot = theCreatedLabelList[theCurrentSelection[0]].lower()
            print(f"selectedSelectionSlot: {theCreatedLabelList[theCurrentSelection[0]].lower()}")
            if selectedSelectionSlot=="Quick".lower():#Quick
                paginationIndex = 0
                print("Menu: " + str(labelsMainMenu[theCurrentSelection[0]]))
                theCurrentSelection = [-1, "Quick"]
            elif selectedSelectionSlot== "BackSpace".lower():#Backspace
                theCurrentSelection[0] = -1
                print("pressed: Backspace")
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
                if len(thewhatsWritten)>0:
                    thewhatsWritten = thewhatsWritten[:-1]
            elif selectedSelectionSlot=="LLM".lower():#LLM
                paginationIndex = 0
                print(f"Menu: {str(labelsMainMenu[theCurrentSelection[0]])}, selection: {theCurrentSelection[0]}")
                theCurrentSelection = [-1, "LLM"]
            elif selectedSelectionSlot=="Space".lower():#Space
                theCurrentSelection[0] = -1
                print("pressed: Space")
                keyboard.press(Key.space)
                keyboard.release(Key.space)
                thewhatsWritten = thewhatsWritten+" "
                last_word = thewhatsWritten.split()[-1] if thewhatsWritten.strip() else None
                if last_word is not None:
                    thelastWord=last_word
            elif selectedSelectionSlot=="Mouse".lower():#Mouse Control
                paginationIndex = 0
                flagSameSelection=True
                theCurrentSelection = [-1, "MouseControl"]
                print(f"menu:  {theCurrentSelection}")
            elif selectedSelectionSlot == "ABC".lower():#ABC
                paginationIndex = 0
                print(f"Letters Menu: {thelabelsABC}")
                theCurrentSelection = [-1,"MultipleLetters"]
                theCreatedLabelList, theCurrentSelection,prettyPrintRects = GetCharacterDivisionMenu(
                    thelabelsABC, totalN, theTopFrame, theCurrentSelection,centerOfContours,systemSelectorColor,variableSelectionSlotColor,thefontScale,thefontThickness,prettyPrintRects,centerOfFace)
            elif selectedSelectionSlot == "Numbers".lower():  # Numbers
                paginationIndex = 0
                print(f"Letters Menu: {thelabelsNumbers}")
                theCurrentSelection = [-1,"MultipleNumbers"]
                theCreatedLabelList, theCurrentSelection,prettyPrintRects = GetCharacterDivisionMenu(
                    thelabelsNumbers, totalN, theTopFrame, theCurrentSelection,centerOfContours,systemSelectorColor,variableSelectionSlotColor,thefontScale,thefontThickness,prettyPrintRects,centerOfFace)
            elif selectedSelectionSlot == "Special Chars".lower():#Special Characters
                paginationIndex = 0
                print(f"Letters Menu: {thelabelsSpecial}")
                theCurrentSelection = [-1,"MultipleSpecialChars"]
                theCreatedLabelList = thelabelsSpecial
                theCreatedLabelList, theCurrentSelection,prettyPrintRects = GetCharacterDivisionMenu(
                    thelabelsSpecial, totalN, theTopFrame, theCurrentSelection,centerOfContours,systemSelectorColor,variableSelectionSlotColor,thefontScale,thefontThickness,prettyPrintRects,centerOfFace)

            elif selectedSelectionSlot == "More Options".lower():#More Options
                flagSameSelection = True
                theCurrentSelection = [-1,"MainMenu"]
                paginationIndex=paginationIndex+1
                if (paginationIndex * totalN-(2*paginationIndex)) >= len(labelsMainMenu):
                    paginationIndex = 0
                print(f"Menu: {selectedSelectionSlot}, pagination: {paginationIndex}")
            elif selectedSelectionSlot == "Previous Options".lower():  # More Options
                flagSameSelection = True
                theCurrentSelection = [-1, "MainMenu"]
                paginationIndex=paginationIndex-1
                if paginationIndex<0:
                    paginationIndex= len(labelsMainMenu) / (totalN - 2)#math.floor((len(labelsMainMenu)*2)/totalN)
                    if paginationIndex.is_integer():
                        paginationIndex=int(paginationIndex-1)
                    else:
                        paginationIndex=math.floor(paginationIndex)
                print(f"Menu: {selectedSelectionSlot}, pagination: {paginationIndex}")
            elif selectedSelectionSlot == "NVC".lower():#Non Verbal Communication: Emojis, custom images, etc!
                paginationIndex = 0
                print(f"Menu: {selectedSelectionSlot}")
                theCurrentSelection = [-1,"NVC"]
                GetNVCmenu(theCreatedLabelList,theOriginalCreatedLabelList,totalN, theTopFrame, centerOfContours, systemSelectorColor, variableSelectionSlotColor,
                           thefontScale,
                           thefontThickness, prettyPrintRects, centerOfFace, paginationIndex, nvcStructure,currentNVCPath)
            #end main menu selection
        else:#inside a subMenu
            if theCurrentSelection[1] == "More Options":  # More Options
                flagSameSelection=True
                theCurrentSelection = theprevSelection
                paginationIndex = paginationIndex + 1
                if (paginationIndex * totalN - (2 * paginationIndex)) >= len(theOriginalCreatedLabelList):
                    paginationIndex = 0
                print(f"SubMenu: {theprevSelection[1]}, pagination: {paginationIndex}")
            elif theCurrentSelection[1] == "Previous Options":  # More Options
                flagSameSelection=True
                theCurrentSelection = theprevSelection
                paginationIndex = paginationIndex - 1
                if paginationIndex < 0:
                    paginationIndex = len(theOriginalCreatedLabelList) / (totalN - 2)  # math.floor((len(labelsMainMenu)*2)/totalN)
                    if paginationIndex.is_integer():
                        paginationIndex = int(paginationIndex - 1)
                    else:
                        paginationIndex = math.floor(paginationIndex)
                print(f"SubMenu: {theprevSelection[1]}, pagination: {paginationIndex}")
            elif (theCurrentSelection[1] == "NVC"):
                theprevSelection = [-1, "NVC"]#Used for pagination
                theTopFrame, theCreatedLabelList,theOriginalCreatedLabelList, prettyPrintRects, paginationIndex,currentNVCPath,theCurrentSelection=GetNVCmenu(theCreatedLabelList,theOriginalCreatedLabelList,totalN,
                            theTopFrame, centerOfContours, systemSelectorColor, variableSelectionSlotColor,
                           thefontScale,thefontThickness, prettyPrintRects, centerOfFace, paginationIndex, nvcStructure,currentNVCPath,theCurrentSelection)
                selectedSelectionSlot = theCreatedLabelList[theCurrentSelection[0]].lower()
                paginationIndex=definePagination(selectedSelectionSlot,paginationIndex,totalN,theCreatedLabelList)

                if any(ext in theCreatedLabelList[theCurrentSelection[0]] for ext in [".png", ".jpg", ".jpeg"]):
                    #keyboard.type(" " + theCreatedLabelList[theCurrentSelection[0]] + " ")
                    thewhatsWritten = thewhatsWritten + " " + theCreatedLabelList[theCurrentSelection[0]] + " "
                    thelastWord = theCreatedLabelList[theCurrentSelection[0]]

                theCurrentSelection[0] = -1
            elif(theCurrentSelection[1]=="MouseControl"):
                theprevSelection = [-1, theCurrentSelection[1]]#Used for pagination
                theCreatedLabelList,theOriginalCreatedLabelList=DisplayMouseMenu(labelsMouse, labelsMouseMenu, totalN, theTopFrame, centerOfContours, systemSelectorColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace,paginationIndex)
                #"Click","Back","Double Click","RightClick"
                selectedSelectionSlot = theCreatedLabelList[theCurrentSelection[0]].lower()
                paginationIndex = definePagination(selectedSelectionSlot, paginationIndex, totalN,
                                                   theOriginalCreatedLabelList)
                if theCreatedLabelList[theCurrentSelection[0]] == "Back":  #"Back",
                    theCurrentSelection = [-1, "MainMenu"]
                    print("Selected Option: MainMenu")
                elif theCreatedLabelList[theCurrentSelection[0]] == "Down":#down
                    mouse.move(0,themouseSpeed)
                    print("mouse moved down: "+str(themouseSpeed))
                elif theCreatedLabelList[theCurrentSelection[0]] == "Click":  # Click
                    mouse.press(Button.left)
                    mouse.release(Button.left)
                    theCurrentSelection[0] = -1
                    print("mouse Left Clicked")
                elif theCreatedLabelList[theCurrentSelection[0]] == "Left":  # Left
                    mouse.move(-themouseSpeed, 0)
                    print("mouse moved left: " + str(themouseSpeed))
                elif theCreatedLabelList[theCurrentSelection[0]] == "D. Click":  # Double Click
                    mouse.click(Button.left,2)
                    theCurrentSelection[0] = -1
                    print("mouse double clicked")
                elif theCreatedLabelList[theCurrentSelection[0]] == "Up":  # Up
                    mouse.move(0,-themouseSpeed)
                    print("mouse moved up: " + str(themouseSpeed))
                elif theCreatedLabelList[theCurrentSelection[0]] == "R. Click":  # RightClick
                    mouse.press(Button.right)
                    mouse.release(Button.right)
                    theCurrentSelection[0] = -1
                elif theCreatedLabelList[theCurrentSelection[0]] == "Right":  # Right
                    mouse.move(themouseSpeed, 0)
                    print("mouse moved left: " + str(themouseSpeed))
                elif " Options" in theCreatedLabelList[theCurrentSelection[0]]:  # Right
                    mouse.move(themouseSpeed, 0)
                    flagSameSelection=True
                    print("mouse moved left: " + str(themouseSpeed))
                elif theCurrentSelection[0] == -1:
                    mouse.move(0, 0)
                    print("mouse is still. ")

            elif(theCurrentSelection[1]=="MultipleLetters" or theCurrentSelection[1]=="MultipleNumbers" or theCurrentSelection[1]=="MultipleSpecialChars"):
                #select character set
                theprevSelection = [-1, theCurrentSelection[1]]  # Used for pagination
                #print(f"label list 1: {theCreatedLabelList}")
                if theCurrentSelection[1]== -1:
                    return theCurrentSelection,theCreatedLabelList, theTopFrame,thelastWord,thelabelsLMM,thewhatsWritten
                elif (theCreatedLabelList[0]=="" and theCurrentSelection[1]=="MultipleLetters"):
                    theCreatedLabelList=thelabelsABC
                elif (theCreatedLabelList[0]=="" and theCurrentSelection[1]=="MultipleNumbers"):
                    theCreatedLabelList=thelabelsNumbers
                elif (theCreatedLabelList[0]=="" and theCurrentSelection[1]=="MultipleSpecialChars"):
                    theCreatedLabelList=thelabelsSpecial

                theOriginalCreatedLabelList=theCreatedLabelList
                if theCurrentSelection[0]>=len(labelsLettersMenu):#in variable menu
                    sizeOfLabelText = len(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)])
                    if (sizeOfLabelText == 1):
                        # print(f"currentSelectionId,labelsLettersMenu,pressed: {theCurrentSelection[0]},{labelsLettersMenu}, {str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)])}")
                        DisplayCharactersMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection,
                                              centerOfContours, systemSelectorColor, thefontScale, thefontThickness,
                                              prettyPrintRects, centerOfFace)
                        if (str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)])) == '_':
                            keyboard.press(Key.space)
                            keyboard.release(Key.space)
                            thewhatsWritten = thewhatsWritten + " "
                            thelastWord = thewhatsWritten.split()[-1] if thewhatsWritten.strip() else None
                        else:
                            if theCurrentSelection[0] != -1:
                                keyboard.press(
                                    str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))
                                keyboard.release(
                                    str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))
                                thewhatsWritten = thewhatsWritten + str(
                                    theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)])
                    elif (sizeOfLabelText > 1):
                        selectedFromListIndex = theCurrentSelection[0] - len(labelsLettersMenu)
                        if (selectedFromListIndex < 0):
                            selectedFromListIndex = 0
                        theprevCreatedLabelsList.append(theCreatedLabelList)
                        # print(f"label list 2: {theCreatedLabelList}, selected from list: {selectedFromListIndex}")
                        theCreatedLabelList, theCurrentSelection, prettyPrintRects = GetCharacterDivisionMenu(
                            theCreatedLabelList[selectedFromListIndex], totalN, theTopFrame, theCurrentSelection,
                            centerOfContours,
                            systemSelectorColor, variableSelectionSlotColor, thefontScale, thefontThickness,
                            prettyPrintRects, centerOfFace)
                else:#in system menu
                    sizeOfLabelText = len(labelsLettersMenu[theCurrentSelection[0]])
                    #when selecting the letter menu
                    if labelsLettersMenu[theCurrentSelection[0]]=="Back":#["Back", "BackSpace", "Main Menu"]
                        #theCurrentSelection = [-1, "Back"]
                        if theprevCreatedLabelsList:
                            if len(theprevCreatedLabelsList) >= 2:
                                theCreatedLabelList=theprevCreatedLabelsList.pop()
                                theCreatedLabelList = ''.join(filter(None, theCreatedLabelList))
                            if len(theprevCreatedLabelsList[0]>12):#if pressed back in last menu
                                theprevCreatedLabelsList.clear()
                                theCurrentSelection = [-1, "MainMenu"]
                            else:
                                if theCurrentSelection[1]=="MultipleLetters":
                                    theCreatedLabelList=thelabelsABC
                                elif  theCurrentSelection[1] == "MultipleNumbers":
                                    theCreatedLabelList = thelabelsNumbers
                                elif theCurrentSelection[1] == "MultipleSpecialChars":
                                    theCreatedLabelList = thelabelsSpecial
                            theCreatedLabelList, theCurrentSelection,prettyPrintRects = GetCharacterDivisionMenu(
                                theCreatedLabelList, totalN, theTopFrame, theCurrentSelection, centerOfContours, systemSelectorColor,
                                variableSelectionSlotColor, thefontScale, thefontThickness,prettyPrintRects,centerOfFace)
                            theCurrentSelection[0] = -1
                        #print("menu: "+str(labelsMainMenu[theCurrentSelection[0]]))
                    elif labelsLettersMenu[theCurrentSelection[0]] == "BackSpace":#BackSpace
                        theCurrentSelection[0] = -1
                        print("pressed: Backspace")
                        keyboard.press(Key.backspace)
                        keyboard.release(Key.backspace)
                        if len(thewhatsWritten)>0:
                            thewhatsWritten = thewhatsWritten[:-1]
                    elif labelsLettersMenu[theCurrentSelection[0]] == "Main":#Main Menu
                        theprevCreatedLabelsList.clear()
                        theCurrentSelection = [-1,"MainMenu"]

                print(f'the actual and prev created label lists: {theCreatedLabelList}, {theprevCreatedLabelsList}')
                if theCurrentSelection[0]>=len(labelsLettersMenu):
                    selectedSelectionSlot = theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)].lower()
                else:
                    selectedSelectionSlot = labelsLettersMenu[theCurrentSelection[0]].lower()
                #selectedSelectionSlot = theCreatedLabelList[theCurrentSelection[0]-len(labelsLettersMenu)].lower()
                paginationIndex = definePagination(selectedSelectionSlot, paginationIndex, totalN, theCreatedLabelList)
                theCurrentSelection[0] = -1
            elif(theCurrentSelection[1]=="Quick"):#"LLM","BackSpace","Back"
                theprevSelection = [-1, theCurrentSelection[1]]#Used for pagination
                theCreatedLabelList,theOriginalCreatedLabelList=DisplayOtherMenus(thelabelsQuick, labelsQuickOptions, totalN, theTopFrame, centerOfContours, systemSelectorColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace,paginationIndex)
                selectedSelectionSlot = theCreatedLabelList[theCurrentSelection[0]].lower()
                paginationIndex = definePagination(selectedSelectionSlot, paginationIndex, totalN,
                                                   theOriginalCreatedLabelList)
                if theCreatedLabelList[theCurrentSelection[0]]=="LLM":
                    theCurrentSelection = [-1, "LLM"]
                elif theCreatedLabelList[theCurrentSelection[0]]=="BackSpace":
                    print("pressed: Backspace")
                    keyboard.press(Key.backspace)
                    keyboard.release(Key.backspace)
                    if len(thewhatsWritten) > 0:
                        thewhatsWritten = thewhatsWritten[:-1]
                elif theCreatedLabelList[theCurrentSelection[0]]=="Back":
                    theCurrentSelection = [-1, "MainMenu"]
                else:
                    if not " Options" in theCreatedLabelList[theCurrentSelection[0]]:
                        print("Typed Quick: " + theCreatedLabelList[theCurrentSelection[0]])
                        keyboard.type(" " + theCreatedLabelList[theCurrentSelection[0]] + " ")
                        thewhatsWritten = thewhatsWritten + " " + theCreatedLabelList[theCurrentSelection[0]] + " "
                        thelastWord = theCreatedLabelList[theCurrentSelection[0]]
                theCurrentSelection[0] = -1
            elif(theCurrentSelection[1]=="LLM"):
                theprevSelection = [-1, theCurrentSelection[1]]  # Used for pagination
                DisplayOtherMenus(labelsLMM, labelsLLMOptions, totalN, theTopFrame, centerOfContours, systemSelectorColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace,paginationIndex)
                if theCurrentSelection[0] == 0:
                    theCurrentSelection = [-1, "Quick"]
                elif theCurrentSelection[0] == 1:
                    print("pressed: Backspace")
                    keyboard.press(Key.backspace)
                    keyboard.release(Key.backspace)
                    if len(thewhatsWritten)>0:
                        thewhatsWritten = thewhatsWritten[:-1]
                elif theCurrentSelection[0] == 2:
                    theCurrentSelection = [-1, "MainMenu"]
                elif (theCurrentSelection[0] >= len(labelsQuickOptions)):
                    print("Typed Quick LLM: " + thelabelsLMM[theCurrentSelection[0] - len(labelsLLMOptions)])
                    keyboard.type(" " + thelabelsLMM[theCurrentSelection[0] - len(labelsLLMOptions)] + " ")
                    thewhatsWritten = thewhatsWritten + " " + thelabelsLMM[theCurrentSelection[0] - len(labelsLLMOptions)] + " "
                    thelastWord=thelabelsLMM[theCurrentSelection[0] - len(labelsLLMOptions)]
                selectedSelectionSlot = theCreatedLabelList[theCurrentSelection[0]].lower()
                paginationIndex = definePagination(selectedSelectionSlot, paginationIndex, totalN, theCreatedLabelList)
                theCurrentSelection[0] = -1

    # --------------------------
    # LLM System--------------
    # --------------------------
    #print(f'logic 1 thelastWord != theprevLastWord: {thelastWord}, {theprevLastWord}')
    if thelastWord != theprevLastWord:
        if queue.empty() and not thellmIsWorkingFlag:  # Check if we need to call the slow method
            thestartTime = time.time()
            llmCall = multiprocessing.Process(target=getLLM, args=(
            queue, theLlmService, theLlmKey, thelastWord, theSeedWord, thetotalOptionsN, thellmContextSize,
            thellmBatchSize))
            llmCall.start()
            thellmIsWorkingFlag = True
        else:
            if thellmWaitTime <= 0:
                theText = f"LLM has been called with \"{thelastWord}\", Calculating Timeframe for future calls"
            else:
                theText = f"LLM has been called with \"{thelastWord}\", aprox time: {totalTime} seconds out of {thellmWaitTime}"
            thelabelsLMM = ["", "", "", "", ""]
            topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, theText, (int(dimensionsTop[1] / 2), int(dimensionsTop[0])), systemSelectorColor,
                                                           thefontScale, thefontThickness, prettyPrintRects, centerOfFace, onCenter=True)

    if not queue.empty():
        thellmIsWorkingFlag = False
        result = queue.get()
        print(f"Prompt: {thelastWord}")
        print(f"result: {result}")
        if totalTime > thellmWaitTime:
            thellmWaitTime = totalTime
            changeLlmWaitTime(totalTime)
        print(f"LLM Call took: {totalTime} seconds")
        thelabelsLMM = result

    return (theCurrentSelection,theprevSelection,theCreatedLabelList,theOriginalCreatedLabelList,theprevCreatedLabelsList, theTopFrame,thelastWord,thelabelsLMM,
            thewhatsWritten,thestartTime,thellmIsWorkingFlag,prettyPrintRects,paginationIndex,currentNVCPath,flagSameSelection)

def DisplayMouseMenu(theLabelsList,theLabelsOptions,totalN,theTopFrame,centerOfContours,colorSystem,thefontScale,thefontThickness,prettyPrintRects,centerOfFace,paginationIndex):
    orderedList = theLabelsOptions + theLabelsList
    createdLabels = paginationSystem(totalN, orderedList, paginationIndex)
    for i in range(totalN):
        # set option labels on topFrame to make them not transparent
        if i <= len(theLabelsOptions):
            topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, createdLabels[i], centerOfContours[i], colorSystem,thefontScale,thefontThickness,prettyPrintRects,centerOfFace,onCenter=True)
        else:
            topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, createdLabels[i], centerOfContours[i], variableSelectionSlotColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace, onCenter=True)
    return createdLabels, orderedList

def DisplayCharactersMenu(theLabelsList, totalN, theTopFrame, theCurrentSelection,centerOfContours,color,thefontScale,thefontThickness,prettyPrintRects,centerOfFace):
    #lettersPerArea = len(theLabelsList) / (totalN - len(labelsLettersMenu))
    lettersPerArea = math.ceil(len(theLabelsList) / (totalN - len(labelsLettersMenu)))
    for i in range(0,totalN):
        # set option labels on topFrame to make them not transparent
        if i <= len(labelsLettersMenu):
            topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, labelsLettersMenu[i], centerOfContours[i], color,thefontScale,thefontThickness,prettyPrintRects,centerOfFace)
        else:
            topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, theLabelsList[i-len(labelsLettersMenu)], centerOfContours[i], variableSelectionSlotColor, thefontScale, thefontThickness, prettyPrintRects, centerOfFace)

def DisplayOtherMenus(theLabelsList,theLabelsOptions, totalN, theTopFrame,centerOfContours,color,thefontScale,thefontThickness,prettyPrintRects,centerOfFace,paginationIndex):
    orderedList=theLabelsOptions+theLabelsList
    createdLabels = paginationSystem(totalN, orderedList, paginationIndex)
    for i in range(totalN):
        # set option labels on topFrame to make them not transparent
        if createdLabels[i] in theLabelsOptions or " Options" in createdLabels[i]:
            topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, createdLabels[i], centerOfContours[i], color,thefontScale,thefontThickness,prettyPrintRects,centerOfFace)
        else:
            topFrame, text_size, prettyPrintRects = prettyPrintInCamera(theTopFrame, createdLabels[i],
                                                                        centerOfContours[i], variableSelectionSlotColor, thefontScale,
                                                                        thefontThickness, prettyPrintRects,
                                                                        centerOfFace)
    return createdLabels,orderedList

def GetCharacterDivisionMenu(theLabelsList, totalN, theTopFrame, theCurrentSelection,centerOfContours,color,lettersColor,thefontScale,thefontThickness,prettyPrintRects,centerOfFace):
    createdLabel = []
    lettersPerArea = math.ceil(len(theLabelsList) / (totalN - len(labelsLettersMenu)))
    if lettersPerArea>0:
        for i in range(totalN):
            # set option labels on topFrame to make them not transparent
            if i < len(labelsLettersMenu):
                topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, labelsLettersMenu[i], centerOfContours[i],color,thefontScale,thefontThickness,prettyPrintRects,centerOfFace)
            else:
                # divide ABC between remaining areas
                startABCIndex = int((i - len(labelsLettersMenu)) * lettersPerArea)
                endABCIndex = int((i - len(labelsLettersMenu)) * lettersPerArea + lettersPerArea)
                createdLabel.append(theLabelsList[startABCIndex:endABCIndex])
                if len(createdLabel[i-len(labelsLettersMenu)])<0:
                    createdLabel[i-len(labelsLettersMenu)] = theLabelsList[i-len(labelsLettersMenu)]
        #print(f"createdLabel: {createdLabel}")
        for loopText in range(len(labelsLettersMenu),totalN):
            #print(loopText)
            if isinstance(createdLabel[loopText - len(labelsLettersMenu)], str):
                topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, createdLabel[loopText-len(labelsLettersMenu)], centerOfContours[loopText],
                                    lettersColor,thefontScale,thefontThickness,prettyPrintRects,centerOfFace)
            elif isinstance(createdLabel[loopText - len(labelsLettersMenu)][0], str):
                topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, createdLabel[loopText-len(labelsLettersMenu)][0], centerOfContours[loopText],
                                    lettersColor,thefontScale,thefontThickness,prettyPrintRects,centerOfFace)
                #print(f"createdLabel: {createdLabel}")
    return createdLabel,theCurrentSelection,prettyPrintRects

def rectangles_intersect(rect1, rect2):
    x1_a, y1_a, x2_a, y2_a = rect1
    x1_b, y1_b, x2_b, y2_b = rect2

    # If one rectangle is to the left of the other
    if x2_a < x1_b or x2_b < x1_a:
        return False

    # If one rectangle is above the other
    if y2_a < y1_b or y2_b < y1_a:
        return False
    return True

def move_rect(rect, direction, n):
    x1, y1, x2, y2 = rect
    dx, dy = direction

    # Normalize the vector
    length = math.hypot(dx, dy)
    if length == 0:
        return rect  # no movement

    dx_norm = dx / length
    dy_norm = dy / length

    # Move all points by n * normalized direction
    x1_new = x1 + dx_norm * n
    y1_new = y1 + dy_norm * n
    x2_new = x2 + dx_norm * n
    y2_new = y2 + dy_norm * n

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

def put_centered_text(img, text, rect, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(255,255,255), thickness=1,lineType=cv2.LINE_AA):
    x1, y1, x2, y2 = rect

    # Center of the rectangle
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Get size of the text
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the bottom-left starting point
    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2  # Add half the height to vertically center

    # Put the text
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, lineType)

def get_directionVector(rect, target_point):
    x1, y1, x2, y2 = rect
    rect_center_x = (x1 + x2) / 2
    rect_center_y = (y1 + y2) / 2

    target_x, target_y = target_point

    # Direction vector from rect center to target
    dx =  rect_center_x -target_x
    dy = rect_center_y - target_y

    return (dx, dy)


def GetNVCmenu(theLabelsList,theOriginalCreatedLabelList,totalN, theTopFrame, centerOfContours, systemSelectionSlotColor, variableSelectionSlotColor,
               thefontScale, thefontThickness, prettyPrintRects, centerOfFace,
               paginationIndex, nvcStructure, originalCurrentPath,currentSelection=[-1,"NVC"]):
    theLabelsList=[]
    global imageCache
    currentPath=originalCurrentPath.split('/')[-1]

    content = nvcStructure.get(currentPath)
    image_files=[]
    if content:
        theLabelsList.extend(content['folders'])
        for f in content['files']:
            if f.lower().endswith(('.jpg', '.png','.jpeg')):
                image_files.append(str(currentPath+"/"+f))
        theLabelsList.extend(image_files)

    #print(f"currentPath: {currentPath}, createdLabels: {theLabelsList}, paginationIndex: {paginationIndex}")
    theOriginalCreatedLabelList=theLabelsList
    createdLabels = paginationSystem(totalN,theLabelsList,paginationIndex)

    for i in range(totalN):
        if i<totalN-1:
            if not any(ext in createdLabels[i].lower() for ext in ['.jpg', '.png', '.jpeg']):
                topFrame, text_size, prettyPrintRects = prettyPrintInCamera(theTopFrame, createdLabels[i],
                                                                        centerOfContours[i], variableSelectionSlotColor, thefontScale,
                                                                        thefontThickness, prettyPrintRects, centerOfFace,
                                                                        onCenter=True)
            else:
                #show image
                image_key = createdLabels[i]#example:'Expressive/Surprised.png'
                image_key = image_key.lstrip("./")
                overlay_img = imageCache[image_key]
                height, width = overlay_img.shape[:2]
                theTopFrame = overlay_image(theTopFrame, overlay_img, x=int(centerOfContours[i][0]-width/2), y=int(centerOfContours[i][1]-height/2))
        else:
            topFrame, text_size, prettyPrintRects = prettyPrintInCamera(theTopFrame, createdLabels[i],
                                                                        centerOfContours[i], systemSelectionSlotColor,
                                                                        thefontScale,
                                                                        thefontThickness, prettyPrintRects,
                                                                        centerOfFace,
                                                                        onCenter=True)

    if currentSelection[0] !=-1:
        if theLabelsList[currentSelection[0]] == "Back":
            if currentPath == ".":
                currentSelection[1] = "MainMenu"
            else:
                originalCurrentPath = originalCurrentPath.rsplit('/', 1)[0]
        else:
            selection = theLabelsList[currentSelection[0]]
            if selection != "" and not (".png" in selection or ".jpg" in selection):
                originalCurrentPath = originalCurrentPath + "/" + theLabelsList[currentSelection[0]]

    return theTopFrame, createdLabels,theOriginalCreatedLabelList, prettyPrintRects, paginationIndex,originalCurrentPath,currentSelection


def prettyPrintInCamera(topFrame, text, theOrg, theColor, thefontScale,thefontThickness,theprettyPrintRects,centerOfFace,lineType=cv2.LINE_AA, onCenter=True):
    text_size=0
    if text !="":
        text_size = cv2.getTextSize(text, font, thefontScale, thefontThickness)[0]
        paddingPercentage=.1
        proportionalPadding=[math.floor(text_size[0]*paddingPercentage),math.floor(text_size[1]*paddingPercentage)]
        # Create a background rectangle
        background_x1 = theOrg[0] - proportionalPadding[0]
        background_y1 = theOrg[1] - text_size[1] - proportionalPadding[1]
        background_x2 = theOrg[0] + text_size[0] + proportionalPadding[0]
        background_y2 = theOrg[1] + proportionalPadding[1]
        background_Rect=(background_x1,background_y1,background_x2,background_y2)
        pixelsToSubstract = (math.floor(text_size[0] / 2), math.floor(text_size[1] / 2))
        direction=get_directionVector(background_Rect,centerOfFace)
        rectanglesIntersect=True
        numberOfShifts=0

        while rectanglesIntersect is True:
            background_x1,background_y1,background_x2,background_y2 = move_rect(background_Rect,direction,pixelsToSubstract[0]/2*numberOfShifts)
            if onCenter:
                background_x1=background_x1 - pixelsToSubstract[0]
                background_y1=background_y1 - pixelsToSubstract[1]
                background_x2=background_x2 - pixelsToSubstract[0]
                background_y2=background_y2 - pixelsToSubstract[1]
            proposedRect = (background_x1, background_y1, background_x2, background_y2)
            if theprettyPrintRects:
                for currentRect in theprettyPrintRects:
                    rectanglesIntersect = rectangles_intersect(proposedRect, currentRect)
                    if rectanglesIntersect is True:
                        #print(f'Found no crashes!')
                        break
            else:
                rectanglesIntersect=False
            numberOfShifts=numberOfShifts+1

        theprettyPrintRects.append(proposedRect)

        cv2.rectangle(topFrame, (background_x1, background_y1), (background_x2, background_y2), (255, 255, 255), -1)
        put_centered_text(topFrame, text, proposedRect,font, thefontScale, theColor, thefontThickness,lineType)
        #(img, text, rect, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(255,255,255), thickness=1)
        #cv2.putText(topFrame, text, theOrg, font, thefontScale, theColor, thefontThickness,lineType)
    return topFrame,text_size,theprettyPrintRects

def paginationSystem(totalN, originalLabelsMenu, paginationIndex):
    # create pagination if total menu options wont fit
    labelsMenu=[]
    if not originalLabelsMenu:
        labelsMenu.append(" ")
        while len(labelsMenu) < totalN:
            labelsMenu[-1] = " "
            labelsMenu.append("Back")
    elif totalN < len(originalLabelsMenu):
        startIndex = paginationIndex * totalN - (2 * (paginationIndex))
        endIndex = startIndex + totalN
        labelsMenu = originalLabelsMenu[startIndex:endIndex]
        labelsMenu = ["Previous Options"] + labelsMenu
        labelsMenu.append("More Options")

        while len(labelsMenu) > totalN:
            labelsMenu.pop(-2)
            endIndex = endIndex - 1
        while len(labelsMenu) < totalN:
            labelsMenu[-1] = ""
            labelsMenu.append("More Options")
            # print(f"paginationIndex: {paginationIndex}, startIndex: {startIndex}, endIndex: {endIndex},totalN: {totalN}, labelsMainMenu: {labelsMainMenu} ")
    elif totalN > len(originalLabelsMenu):
        labelsMenu = originalLabelsMenu
        while len(labelsMenu) < totalN:
            if labelsMenu[-1]=="Back":
                labelsMenu[-1]=""
            if len(labelsMenu) < totalN-1:
                labelsMenu.append("")
            else:
                labelsMenu.append("Back")
    return labelsMenu

def GetMainMenu(totalN,theTopFrame,theCurrentSelection,centerOfContours,color,lettersColor,thefontScale,thefontThickness,
                thelabelsNumbers,thelabelsSpecial,thelabelsABC,originalLabelsMainMenu,prettyPrintRects,centerOfFace,paginationIndex):

    createdLabel = []

    labelsMainMenu=paginationSystem(totalN,originalLabelsMainMenu,paginationIndex)

    for i in range(totalN):
        # set option labels on topFrame to make them not transparent
        if (totalN <= len(labelsMainMenu) and theCurrentSelection[1] == "MultipleNumbers"):
            createdLabel.append(thelabelsNumbers)
        elif (totalN <= len(labelsMainMenu) and theCurrentSelection[1] == "MultipleSpecialChars"):
            createdLabel.append(thelabelsSpecial)
        elif(totalN<=len(labelsMainMenu) and theCurrentSelection[1]=="MultipleLetters"):
            createdLabel.append(thelabelsABC)
        elif (theCurrentSelection[1] == "MainMenu"):
            createdLabel=labelsMainMenu
        elif(totalN<=len(labelsMainMenu)):
            createdLabel.append(thelabelsABC)


        if i < len(labelsMainMenu):
            topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, labelsMainMenu[i], centerOfContours[i], color,thefontScale,thefontThickness,prettyPrintRects,centerOfFace, cv2.LINE_AA)
        else:
            # divide ABC between remaining areas
            lettersPerArea = math.ceil(len(thelabelsABC) / (totalN - len(labelsMainMenu)))
            startABCIndex = math.floor((i - len(labelsMainMenu)) * lettersPerArea)
            endABCIndex = math.ceil((i - len(labelsMainMenu)) * lettersPerArea + lettersPerArea)
            createdLabel.append(thelabelsABC[startABCIndex:endABCIndex])
            topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, createdLabel[i - len(labelsMainMenu)], centerOfContours[i],
                                                           lettersColor,thefontScale,thefontThickness,prettyPrintRects,centerOfFace)
    while len(createdLabel) < totalN:
        createdLabel.append("")
    return createdLabel,prettyPrintRects,paginationIndex

def GetSelectionLogic(theTopFrame,dimensionsTop,theSelectionCurrentTime,theselectionType,theCurrentSelection
                      ,theSelected,thePrevSelected,ellipsePoly,nosePosition,contours,thetimeOnLocation,theredFrameColor,
                      thefontScale,thefontThickness,thefps,flagSameSelection,thewhatsWritten,theselectionWaitTime,prettyPrintRects,centerOfFace):
    if theCurrentSelection[1]=="MouseControl":
        theSelected = -1
        noseOnNeutral = cv2.pointPolygonTest(ellipsePoly, nosePosition, False)  # 0,1 is on contour,-1 is not on contour

        if noseOnNeutral < 0 and not flagSameSelection:
            # print(f'GetSelectionLogic: timeOnLocation: {thetimeOnLocation}, theSelectionCurrentTime: {theSelectionCurrentTime}')
            for idx, contour in enumerate(contours):
                noseOptionResult = cv2.pointPolygonTest(contour, nosePosition, False)
                if noseOptionResult > 0:
                    theSelected = idx
                    if thePrevSelected != theSelected:
                        thePrevSelected = theSelected
                        theSelectionCurrentTime = 0
                        flagSameSelection = False
                    else:
                        if theSelectionCurrentTime <= .1:
                            theSelectionCurrentTime += selectionCurrentTime + 1 / thefps
                        else:  # time has passed
                            print(f"Timer Reset, selected: {theCurrentSelection[0]}")
                            theCurrentSelection[0] = thePrevSelected
                            theSelectionCurrentTime = 0
                            print("Selected index: " + str(theCurrentSelection[0]))
                    break
        elif noseOnNeutral>=0:#nose is on neutral
            flagSameSelection = False
        else:
            theCurrentSelection[0] = -1
    elif theselectionType=='TimedOnLocation':
        theSelected = -1
        noseOnNeutral = cv2.pointPolygonTest(ellipsePoly, nosePosition, False)  # 0,1 is on contour,-1 is not on contour
        if noseOnNeutral<0:
            #print(f'GetSelectionLogic: timeOnLocation: {thetimeOnLocation}, theSelectionCurrentTime: {theSelectionCurrentTime}')
            for idx, contour in enumerate(contours):
                noseOptionResult = cv2.pointPolygonTest(contour, nosePosition, False)
                if noseOptionResult > 0:
                    theSelected = idx
                    if thePrevSelected!=theSelected:
                        thePrevSelected = theSelected
                        theSelectionCurrentTime=0
                        flagSameSelection = False
                    else:
                        if  flagSameSelection==False:
                            if theSelectionCurrentTime <= thetimeOnLocation:
                                theSelectionCurrentTime += selectionCurrentTime + 1 / thefps
                                desiredText=f'Selection on {(thetimeOnLocation-theSelectionCurrentTime):.1f}'
                                #print(desiredText)#topFrame,dimensionsTop,
                                (text_width, text_height), baseline = cv2.getTextSize(thewhatsWritten, font, thefontScale, thefontThickness)
                                topFrame,text_size,prettyPrintRects=prettyPrintInCamera(theTopFrame, desiredText,
                                                                (int(dimensionsTop[1] / 2), int(dimensionsTop[0]+int(text_height*0.4)-(text_height*1.6*2))), #-3 is padding
                                                                 theredFrameColor,thefontScale,thefontThickness,prettyPrintRects,centerOfFace,
                                                                onCenter=True)  # x and y are inverted in call
                            else:  # time has passed
                                #print(f"Timer Reset, selected: {theCurrentSelection[0]}")
                                theCurrentSelection[0] = thePrevSelected
                                theSelectionCurrentTime = 0
                                flagSameSelection = True
                    break
        else:
            if noseOnNeutral >= 0:
                flagSameSelection=False
    else:#BackToCenter or any other
        noseOnNeutral = cv2.pointPolygonTest(ellipsePoly, nosePosition, False)  # 0,1 is on contour
        if noseOnNeutral >= 0:  # inside of ellipse
            if theSelectionCurrentTime != 0:
                print("Timer Reset Neutral Zone")
                theSelectionCurrentTime = 0
                theSelected=-1
        elif noseOnNeutral == -1: #outside of ellipse
            for idx, contour in enumerate(contours):
                noseOptionResult = cv2.pointPolygonTest(contour, nosePosition, False)
                if noseOptionResult > 0:
                    theSelectionCurrentTime = selectionCurrentTime + 1 / thefps
                    theSelected = idx
                    if thePrevSelected != theSelected:
                        # change timer depending on area change
                        theSelectionCurrentTime = 0
                        thePrevSelected=theSelected
                        # print("Timer Reset Area Changed")
                    elif theSelectionCurrentTime >= theselectionWaitTime:
                        theSelected = idx
                        thePrevSelected = theSelected
                        break
        if (theSelected == -1) and (thePrevSelected != theSelected):
            theCurrentSelection[0] = thePrevSelected
            print("Selected: " + str(theCurrentSelection[0]))
            thePrevSelected = theSelected

    return theSelected,thePrevSelected,theCurrentSelection,theSelectionCurrentTime,flagSameSelection,prettyPrintRects

def resize_image_keep_aspect(image, target_width):
    height, width = image.shape[:2]
    scale = target_width / width
    new_height = int(height * scale)
    resized = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def preload_images(nvc_structure, root_dir,targetWidth=50):
    global imageCache
    valid_extensions = ('.png', '.jpg', '.jpeg')

    for path, content in nvc_structure.items():
        for file in content['files']:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root_dir, path, file)
                full_path = os.path.normpath(full_path)  # Normalize for OS
                image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    resized = resize_image_keep_aspect(image, targetWidth)
                    key = f"{path}/{file}" if path != '.' else file
                    imageCache[key] = resized
    print("Loaded image keys:")
    for key in imageCache:
        print(key)
    return imageCache

import numpy as np

def overlay_image(frame, overlay, x=0, y=0):
    frame_h, frame_w = frame.shape[:2]
    overlay_h, overlay_w = overlay.shape[:2]

    # Calculate the region of the overlay that fits within the frame
    start_x = max(x, 0)
    start_y = max(y, 0)
    end_x = min(x + overlay_w, frame_w)
    end_y = min(y + overlay_h, frame_h)

    # Exit if no overlap exists
    if start_x >= end_x or start_y >= end_y:
        return frame

    # Crop the overlay to the overlapping region
    overlay_start_x = start_x - x
    overlay_start_y = start_y - y
    overlay_end_x = overlay_start_x + (end_x - start_x)
    overlay_end_y = overlay_start_y + (end_y - start_y)
    cropped_overlay = overlay[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]

    # Extract the region of the frame to update
    frame_region = frame[start_y:end_y, start_x:end_x]

    # Apply alpha blending if overlay has an alpha channel
    if cropped_overlay.shape[2] == 4:
        alpha = cropped_overlay[:, :, 3] / 255.0
        for c in range(3):
            frame_region[:, :, c] = (alpha * cropped_overlay[:, :, c] +
                                   (1 - alpha) * frame_region[:, :, c]).astype('uint8')
    else:
        frame_region[:, :] = cropped_overlay[:, :, :3]

    return frame

def mainLoop(queue):
    # local variables initiallized from their global counterparts:
    thewhatsWritten = seedWord
    thelastWord = seedWord
    theprevLastWord = ""
    thecurrentSelection = [-1,"MainMenu"]
    theprevSelection = thecurrentSelection
    theprev_frame_time = 0
    thefps = 0
    thecreatedLabelsList = createdLabelsList
    theOriginalCreatedLabelList=createdLabelsList
    theprevCreatedLabelsList=createdLabelsList
    theselected=-1
    theprevSelectedIndex=theselected

    theselectionCurrentTime=selectionCurrentTime
    thelabelsLMM=labelsLMM
    thelabelsMainMenu=""
    flagSameSelection=False
    thestartTime=time.time()
    thellmIsWorkingFlag=False
    prettyPrintRects=[]

    #variables from config:
    (theselectionType,thetimeOnLocation,
     theignoreGuiAngles, theignoreAngleArc,
     thettsRate, thettsVolume, thettsVoiceType,
     thecenterSizeX,thecenterSizeY,theoffsetX,theoffsetY,
     thetotalOptionsN,themouseSpeed,theselectionWaitTime,
     thelabelsMainMenu,thelabelsABC,thelabelsNumbers,thelabelsSpecial,thelabelsQuick,
     thefontScale,thefontThickness,
     thecamSizeX,thecamSizeY,theshowFPS,theshowWritten,
     thellmContextSize,thellmBatchSize,thellmWaitTime,
     themaxWhatsWrittenSize,theshowWrittenMode,theseedWord,
     theLlmService,theLlmKey,
     selectorFrameColor,ReferenceFrameColor,variableSelectionSlotColor,systemSelectionSlotColor,highlightSlotColor)=GetConfigSettings()

    #thellm=loadLLM("zephyr-7b-beta.Q4_K_M.gguf",llmContextSize,llmBatchSize)
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    #set pagination
    paginationIndex=0

    #read nvc structure and file names:
    nvcStructure = read_NVC_structure('./NVC_Graphics')
    currentNVCPath="."
    print(f"NVC Structure: \n\n {nvcStructure} \n\n")
    imageCache = preload_images(nvcStructure, './NVC_Graphics')
    #Set camera
    if imageOrVideo:
        sourceTop=0
        if len(sys.argv) > 1:
            sourceTop = sys.argv[1]
        #print("SourceTop: "+str(sourceTop)+", sourceSide: "+str(sourceSide))
        cameraTopView = cv2.VideoCapture(sourceTop)
        # cameraTopView.set(cv2.CAP_PROP_FPS, 30.0)
        cameraTopView.set(cv2.CAP_PROP_FRAME_WIDTH, thecamSizeX);#640
        cameraTopView.set(cv2.CAP_PROP_FRAME_HEIGHT, thecamSizeY);#480
    else:
        cameraTopView = cv2.imread("testImages/Sofa2.jpg")

    while cv2.waitKey(1) != 27:  # Escape
        #remove locations of prettyPrintRects
        prettyPrintRects.clear()
        #capture new frames or use a test image
        if imageOrVideo:
            hasTopFrame, topFrame = cameraTopView.read()
            if not hasTopFrame:
                break
        else:
            topFrame = cv2.imread("testImages/Sofa2.jpg")
        dimensionsTop = topFrame.shape

        #set up GUI layer
        topFrame = cv2.flip(topFrame, 1)
        uiFrame = topFrame.copy()

        #Facial Recognition
        imgRGB = cv2.cvtColor(topFrame, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        # font
        selectorPosition = (50, 50)
        color = ReferenceFrameColor#
        lettersColor=(50,50,255)

        if results.multi_face_landmarks:
            for faceLandMarks in results.multi_face_landmarks:
                #mpDraw.draw_landmarks(topFrame,faceLandMarks,mpFaceMesh.FACEMESH_FACE_OVAL, drawSpecs,drawSpecs)
                #get max size of face
                shape = topFrame.shape
                faceXmin,faceXmax,faceYmin,faceYmax,ecrpX,ecrpY=GetFaceSizeAndECRP(shape, faceLandMarks.landmark)

                for idx, landMark in enumerate(faceLandMarks.landmark):
                    if idx>44 and idx<46:#nose points:44 and 45
                        x = landMark.x
                        y = landMark.y
                        relative_x = int(x * shape[1])
                        relative_y = int(y * shape[0])
                        selectorPosition=(relative_x, relative_y)

                        #circle on center of face
                        sizeOfFace=math.sqrt(math.pow(shape[1]*(faceXmax-faceXmin),2)+math.pow(shape[0]*(faceYmax-faceYmin),2))
                        radiusAsPercentageX=int((thecenterSizeX/2)/100*sizeOfFace)
                        radiusAsPercentageY= int((thecenterSizeY/2)/100 * sizeOfFace)
                        offsetAsPixelsX=int((theoffsetX/2)/100*sizeOfFace)
                        offsetAsPixelsY = int((theoffsetY/2) / 100 * sizeOfFace)
                        #print(f"offsetAsPixels: ({offsetAsPixelsX},{offsetAsPixelsY})")
                        ecrpX = ecrpX+offsetAsPixelsX
                        ecrpY = ecrpY+offsetAsPixelsY
                        centerOfFace=[ecrpX,ecrpY]
                        #print(ecrpX,ecrpY, faceXmin,faceXmax,faceYmin,faceYmax)
                        # cv2.putText(topFrame,str(idx), org, font,fontScale, color, thickness, cv2.LINE_AA)

                        # ---------GUI--------------
                        # create the n zones for buttons, geometry must be created by placing points in clockwise order
                        # -------------------------
                        ellipsePoly,contours,centerOfContours=GetGUI(uiFrame,radiusAsPercentageX,radiusAsPercentageY,
                                                                     thetotalOptionsN,ecrpX,ecrpY,selectorPosition,
                                                                     theignoreGuiAngles,theignoreAngleArc)
                        (thecurrentSelection,theprevSelection,thecreatedLabelsList,theOriginalCreatedLabelList,theprevCreatedLabelsList,topFrame,thelastWord,
                         thelabelsLMM,thewhatsWritten,thestartTime,thellmIsWorkingFlag,
                         prettyPrintRects,paginationIndex,currentNVCPath,flagSameSelection) = GetMenuSystem (
                             queue, topFrame, thetotalOptionsN,thecurrentSelection,theprevSelection, thecreatedLabelsList,theOriginalCreatedLabelList, theprevCreatedLabelsList,
                               centerOfContours, color, lettersColor, dimensionsTop, theshowFPS, thestartTime,
                               thelastWord, theprevLastWord, thellmIsWorkingFlag,
                               theLlmService, theLlmKey,
                               thellmWaitTime, font,
                               thefontScale,
                               thefontThickness,
                               variableSelectionSlotColor,
                               thewhatsWritten, thefps, thelabelsLMM, thelastWord,
                               thelabelsQuick,
                               thelabelsABC,
                               thelabelsMainMenu,
                               thelabelsNumbers,
                               thelabelsSpecial,
                               themouseSpeed, theshowWritten, theshowWrittenMode, themaxWhatsWrittenSize,
                               thetotalOptionsN, paginationIndex,
                               thellmContextSize,
                               thellmBatchSize, prettyPrintRects, centerOfFace,nvcStructure,currentNVCPath,
                                flagSameSelection
                               )

                        # -------------------------
                        # Selection Logic----------
                        # -------------------------
                        theselected,theprevSelectedIndex,thecurrentSelection,theselectionCurrentTime,flagSameSelection,prettyPrintRects = GetSelectionLogic(
                            topFrame,dimensionsTop,theselectionCurrentTime,theselectionType,
                            thecurrentSelection,theselected,theprevSelectedIndex,ellipsePoly,selectorPosition,contours,
                            thetimeOnLocation,variableSelectionSlotColor,thefontScale,thefontThickness,thefps,flagSameSelection,
                            thewhatsWritten,theselectionWaitTime,prettyPrintRects,centerOfFace)

                        if thelastWord != theprevLastWord:
                            toBeSpoken=thelastWord
                            #thewhatsWritten = thewhatsWritten + " " + thelastWord
                            if toBeSpoken.lower().endswith(('.jpg', '.jpeg', '.png')):
                                toBeSpoken= toBeSpoken.split('.')[0]
                            toBeSpoken = toBeSpoken.replace('/', ' ')#remove slashes for better readeability


                            print(f'speaking: {toBeSpoken}')
                            speak_non_blocking(toBeSpoken,thettsRate, thettsVolume, thettsVoiceType,)
                            theprevLastWord = thelastWord


        # FPS calculations
        thenew_frame_time = time.time()
        if thenew_frame_time != theprev_frame_time:
            theprevFps = thefps
            thefps = 1 / (thenew_frame_time - theprev_frame_time)
            theprev_frame_time = thenew_frame_time

        #Display
        #cv2.imshow("top frame", topFrame)
        combinedCalibImage = topFrame.copy()
        uiFrame = cv2.addWeighted(uiFrame, alpha, combinedCalibImage, 1 - alpha, 0)
        cv2.imshow("Facial Control HMI", uiFrame)



if __name__ == '__main__':
    multiprocessing.freeze_support()  # Only necessary if using PyInstaller
    thellmIsWorkingFlag=True
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=mainLoop, args=(queue,))
    p.start()
    p.join()