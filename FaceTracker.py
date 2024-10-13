#To Deploy exe:
#on laptop:
#pyinstaller --onefile --windowed --add-data "./config.txt;." --add-data "C:\Users\evelasquez\PycharmProjects\FacialControlHCI\.venv\lib\site-packages\mediapipe;mediapipe/" FaceTracker.py
#on PC:
#pyinstaller --onefile --windowed --add-data "./config.txt;." --add-data "C:\Users\velas\PycharmProjects\ballTracker\venv\lib\site-packages\mediapipe;mediapipe/" FaceTracker.py
import math
import re

import multiprocessing
import cv2
import mediapipe as mp
import time
import sys
import requests
import json
from llama_cpp import Llama

import numpy as np
#from numpy.array_api import astype

import pynput
from pynput.keyboard import Key, Controller as keyController
from pynput.mouse import Button, Controller as mouseController

import FaceTracker

#Virtual Keyboard
keyboard = keyController()# Controller()
mouse = mouseController()

#video settings
imageOrVideo = True

#fps variables
fps=30
speedFps=0
prevFps=0
prev_frame_time=0
new_frame_time=0

#menu selection variables
totalOptionsN = 8
mouseSpeed=5
selectionWaitTime=0.4
#------------------
labelsMainMenu=["Quick", "BackSpace","LLM","Space","Mouse", "ABC","Numbers","Special Chars"]
labelsLettersMenu=["Quick", "BackSpace", "Back"]
#------------------
#------------------
labelsMouseMenu=["Click","Back","Double Click","RightClick"]
labelsMouse=["Down","Left","Up","Right"]
#------------------
labelsQuickOptions=["LLM","BackSpace","Back"]
labelsLLMOptions=["Quick","BackSpace","Back"]
labelsLMM=["test1","test2","test3","test4","test5",]
#------------------
configFilePath='./config.txt'

#------------------
#menu selection init hidden variables
createdLabelsList=[]
selectionCurrentTime=0
currentSelection=[-1,"MainMenu"]#first is option chosen, second is how deep
selected = -1
prevSelected = -1
lastWord=""
prevLastWord=" "
showFPS=False
showWritten=False

#menu selection config file init variables
totalOptionsN = 8
mouseSpeed = 5
selectionWaitTime = 0.4
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
llmWaitTime=10
startTime=0
llmIsWorkingFlag=False
whatsWritten=""
maxWhatsWrittenSize=20
showWrittenMode="Single"

#ui variables
greenFrameColor = (0, 255, 0)  # BGR
redFrameColor = (0, 0, 255)  # BGR
alpha = 0.3
font=cv2.FONT_HERSHEY_SIMPLEX


def loadLLM(thePath,contextSize=512,batchSize=126):

    theLLM = Llama(model_path="zephyr-7b-beta.Q4_K_M.gguf", n_ctx=contextSize, n_batch=batchSize,use_gpu=True)
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
    timeLine=f"llmWaitTime={theTime}"

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
    totalOptionsN = 8
    mouseSpeed = 5
    selectionWaitTime = 0.4
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
    llmWaitTime=10
    maxWhatsWrittenSize=20
    showWrittenMode="Single"

    with open(configFilePath, 'r') as file:
        for line in file:
            # Strip whitespace and check if the line starts with #
            line = line.strip()
            if line.startswith('#') or not line:
                continue  # Skip comments and empty lines

            key, value = line.strip().split('=',1)
            if key == "totalOptionsN":
                totalOptionsN = int(value)
                FaceTracker.totalOptionsN=totalOptionsN
            elif key == "mouseSpeed":
                mouseSpeed = int(value)
                FaceTracker.mouseSpeed=mouseSpeed
            elif key == "selectionWaitTime":
                selectionWaitTime = float(value)
                FaceTracker.selectionWaitTime=selectionWaitTime
            elif key == "labelsABC":
                labelsABC = value
                FaceTracker.labelsABC=labelsABC
            elif key == "labelsNumbers":
                labelsNumbers = value
                FaceTracker.labelsNumbers=labelsNumbers
            elif key == "labelsSpecial":
                labelsSpecial = value
                FaceTracker.labelsSpecial=labelsSpecial
            elif key == "labelsQuick":
               labelsQuick = value.split(',')
               FaceTracker.labelsQuick=labelsQuick
            elif key == "fontScale":
                fontScale = float(value)
                FaceTracker.fontScale=fontScale
            elif key == "fontThickness":
                fontThickness = int(value)
                FaceTracker.fontThickness=fontThickness
            elif key == "camSizeX":
                camSizeX = int(value)
                FaceTracker.camSizeX=camSizeX
            elif key == "camSizeY":
                camSizeY = int(value)
                FaceTracker.camSizeY=camSizeY
            elif key == "showFPS":
                if value == "False" or value == "False" or value == "0":
                    value=""
                showFPS = bool(value)
                FaceTracker.showFPS=showFPS
            elif key == "showWritten":
                if value == "False" or value == "false" or value == "0"or value == "f" or value == "F":
                    value=""
                showWritten = bool(value)
                FaceTracker.showWritten=showWritten
            elif key == "llmContextSize":
                llmContextSize = int(value)
                FaceTracker.llmContextSize=llmContextSize
            elif key == "llmBatchSize":
                llmBatchSize = int(value)
                FaceTracker.llmBatchSize=llmBatchSize
            elif key == "llmWaitTime":
                llmWaitTime = int(value)
                FaceTracker.llmWaitTime = llmWaitTime
            elif key == "maxWhatsWrittenSize":
                maxWhatsWrittenSize = int(value)
                FaceTracker.maxWhatsWrittenSize = maxWhatsWrittenSize
            elif key == "showWrittenMode":
                showWrittenMode = value
                FaceTracker.showWrittenMode = showWrittenMode


    # Print the variables
    print(f"totalOptionsN: {totalOptionsN}, mouseSpeed: {mouseSpeed}, selectionWaitTime: {selectionWaitTime}"
          f", labelsABC: {labelsABC}, labelsQuick: {labelsQuick}, fontScale: {fontScale}"
          f", fontThickness: {fontThickness}, camSizeX: {camSizeX}, camSizeY: {camSizeY}"
          f", showFPS: {showFPS}, showWritten: {showWritten}"
          f", llmContextSize: {llmContextSize}, llmBatchSize: {llmBatchSize}, llmWaitTime: {llmWaitTime}"
          f", maxWhatsWrittenSize: {maxWhatsWrittenSize}, showWrittenMode: {showWrittenMode}")
    return totalOptionsN,mouseSpeed,selectionWaitTime,labelsABC,labelsNumbers,labelsSpecial,labelsQuick,fontScale\
        ,fontThickness,camSizeX,camSizeY, showFPS, showWritten,llmContextSize,llmBatchSize,llmWaitTime,maxWhatsWrittenSize,showWrittenMode

def GetAreaPoints(totalN,centerOfFaceX,centerOfFaceY,areaSize, rotationAngle):
    #(0,0) being center of face
    #m=rise/run
    #y=mx+b
    #x=(y-b)/m
    degreesOfEach = math.radians(360/(totalN))
    contours=[]
    centerOfContours=[]

    for i in range(0,totalN,1):
        #get contour endpoints for drawing the contour
        x1 = centerOfFaceX + (math.cos(degreesOfEach * i))*areaSize
        y1 = centerOfFaceY + (math.sin(degreesOfEach * i))*areaSize
        x2 = centerOfFaceX + (math.cos(degreesOfEach * (i+1)))*areaSize
        y2 = centerOfFaceY + (math.sin(degreesOfEach * (i+1)))*areaSize

        #rotate endpoints by rotationAngle around the origin:
        rotatedX1 = ((x1-centerOfFaceX)*math.cos(math.radians(rotationAngle)))-((y1-centerOfFaceY)*math.sin(math.radians(rotationAngle)))+centerOfFaceX
        rotatedY1 = ((x1-centerOfFaceX) * math.sin(math.radians(rotationAngle))) + ((y1-centerOfFaceY) * math.cos(math.radians(rotationAngle))) + centerOfFaceY
        rotatedX2 = ((x2-centerOfFaceX) * math.cos(math.radians(rotationAngle))) - ((y2-centerOfFaceY) * math.sin(math.radians(rotationAngle)))+centerOfFaceX
        rotatedY2 = ((x2-centerOfFaceX) * math.sin(math.radians(rotationAngle))) + ((y2-centerOfFaceY) * math.cos(math.radians(rotationAngle)))+centerOfFaceY

        #points = [[centerOfFaceX, centerOfFaceY], [x1, y1], [x2, y2]]
        points = [[centerOfFaceX, centerOfFaceY], [rotatedX1, rotatedY1], [rotatedX2, rotatedY2]]
        ctr = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
        contours.append(ctr)

        #get center of contour to place labels
        centerOfContourX = centerOfFaceX + (math.cos(degreesOfEach * (i+0.5)))*areaSize
        centerOfContourY = centerOfFaceY + (math.sin(degreesOfEach * (i+0.5))) * areaSize

        rotatedXCenterOfContour = ((centerOfContourX-centerOfFaceX) * math.cos(math.radians(rotationAngle))) - ((centerOfContourY-centerOfFaceY) * math.sin(math.radians(rotationAngle)))+centerOfFaceX
        rotatedYCenterOfContour = ((centerOfContourX-centerOfFaceX) * math.sin(math.radians(rotationAngle))) + ((centerOfContourY-centerOfFaceY) * math.cos(math.radians(rotationAngle)))+centerOfFaceY

        centerOfContour=(int(rotatedXCenterOfContour),int(rotatedYCenterOfContour))
        centerOfContours.append(centerOfContour)

    return contours, centerOfContours


def GetFaceSizeAndCenter(shape,landMarks):
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


def GetGUI(theUIFrame,radiusAsPercent,totalN,centerFaceX,centerFaceY,nosePosition):
    theContours, centerContours = GetAreaPoints(totalN, centerFaceX, centerFaceY,100,360/(totalN*2))  # area number, total areas
    # set center of face
    for i in range(totalN):
        if i %2 ==0:
            cv2.fillPoly(theUIFrame, [theContours[i]], [250, 250, 250])
        else:
            cv2.fillPoly(theUIFrame, [theContours[i]], [150, 150, 150])
    #cv2.circle(theUIFrame, (centerFaceX, centerFaceY), radiusAsPercent, blueFrameColor, -1)
    cv2.ellipse(theUIFrame, (centerFaceX, centerFaceY), (radiusAsPercent, int(radiusAsPercent * 0.5)), 0, 0, 360,
                greenFrameColor, -1)
    polyEllipse = cv2.ellipse2Poly((centerFaceX, centerFaceY), (radiusAsPercent, int(radiusAsPercent * 0.5)),
                                   0, 0, 360, 1)
    # set center of nose as a controller
    cv2.circle(theUIFrame, nosePosition, int(radiusAsPercent * 0.25), redFrameColor, -1)

    return polyEllipse,theContours,centerContours

def getLLM(queue,llmWaitTime,lastWord):
    llm = loadLLM("zephyr-7b-beta.Q4_K_M.gguf", llmContextSize, llmBatchSize)
    #print("calling LLM: ")
    prompt = f"Give me a list of only {totalOptionsN - len(labelsLLMOptions)} words with no explanation that go after: \"{lastWord}\""
    #print(f"prompt: {prompt}")
    prompt = generate_prompt_from_template(prompt)
    generatedText = generate_text(llm, prompt)
    generatedText = re.sub(r'\d+\.?\s*', '', generatedText)
    #print(f"generated Text: {generatedText}")
    result = generatedText.splitlines()
    #print(f"generated Text: {FaceTracker.labelsLMM}")
    queue.put(result)

def showWhatsWritten(theTopFrame,dimensionsTop):
    #show up until max numbers of characters
    if len(FaceTracker.whatsWritten)>FaceTracker.maxWhatsWrittenSize:
        FaceTracker.whatsWritten=FaceTracker.whatsWritten[-FaceTracker.maxWhatsWrittenSize:]
    #start options
    if "mirror" in FaceTracker.showWrittenMode.lower():
        percentageTakenByWritten = .1
        top_height = int(dimensionsTop[0] * percentageTakenByWritten)
        # splitIntoHalves
        top_half = theTopFrame[:top_height, :]
        bottom_half = theTopFrame[top_height:, :]
        topHalfSize = top_half.shape
        #put black background
        top_half = np.zeros((topHalfSize[0], topHalfSize[1], 3), dtype=np.uint8)

        if FaceTracker.showWrittenMode=="CloneAndMirror":
            left_half = top_half[:, :topHalfSize[1] // 2]  # Left half
            right_half = top_half[:, topHalfSize[1] // 2:]  # Right half
            sideHalfSize = left_half.shape
            # put white text on black background
            (text_width, text_height), baseline = cv2.getTextSize(FaceTracker.whatsWritten, FaceTracker.font,
                                                                  FaceTracker.fontScale, FaceTracker.fontThickness)
            cv2.putText(right_half, FaceTracker.whatsWritten,
                        (int((sideHalfSize[1] / 2) - text_width / 2), text_height + 20), FaceTracker.font,
                        FaceTracker.fontScale, (255, 255, 255), FaceTracker.fontThickness, cv2.LINE_AA)
            cv2.putText(left_half, FaceTracker.whatsWritten,
                        (int((sideHalfSize[1] / 2) - text_width / 2), text_height + 20), FaceTracker.font,
                        FaceTracker.fontScale, (255, 255, 255), FaceTracker.fontThickness, cv2.LINE_AA)
            # Create a mirrored version of the image
            mirrored_right_half = cv2.flip(right_half, 1)
            combined_TopHalf = cv2.hconcat([left_half, mirrored_right_half])
            theTopFrame = cv2.vconcat([combined_TopHalf, bottom_half])
        #mirror
        else:
            #put white text on black background
            (text_width, text_height), baseline = cv2.getTextSize(FaceTracker.whatsWritten, FaceTracker.font, FaceTracker.fontScale,FaceTracker.fontThickness)
            cv2.putText(top_half, FaceTracker.whatsWritten, (int((dimensionsTop[1] / 2)-text_width/2), text_height+20), FaceTracker.font, FaceTracker.fontScale, (255, 255, 255), FaceTracker.fontThickness, cv2.LINE_AA)
            # Create a mirrored version of the image
            mirrored_top_half = cv2.flip(top_half, 1)
            theTopFrame = cv2.vconcat([mirrored_top_half, bottom_half])
    else:
        FaceTracker.prettyPrintInCamera(theTopFrame, FaceTracker.whatsWritten, (int(dimensionsTop[1] / 2), 20),
                                        FaceTracker.redFrameColor, onCenter=True)  # x and y are inverted in call
    return theTopFrame


def GetMenuSystem(queue, theTopFrame, totalN,theCurrentSelection,theCreatedLabelList,centerOfContours,color,lettersColor,dimensionsTop):
    if FaceTracker.showFPS:
        cv2.putText(theTopFrame, "FPS: " + str(int(fps)), org=(int(dimensionsTop[0] / 2), 20),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
    endTime = time.time()
    totalTime = math.ceil(endTime - FaceTracker.startTime)

    if FaceTracker.lastWord !=FaceTracker.prevLastWord:
        FaceTracker.prevLastWord = FaceTracker.lastWord
        if queue.empty() and not FaceTracker.llmIsWorkingFlag:  # Check if we need to call the slow method
            FaceTracker.startTime = time.time()
            llmCall = multiprocessing.Process(target=getLLM, args=(queue,FaceTracker.llmWaitTime,FaceTracker.lastWord,))
            llmCall.start()
            FaceTracker.llmIsWorkingFlag= True
        else:
            FaceTracker.prettyPrintInCamera(theTopFrame,
                                f"LLM has been called with \"{FaceTracker.lastWord}\", aprox time: {totalTime} seconds out of {FaceTracker.llmWaitTime}",
                                (int(dimensionsTop[0] / 2), 20), color)

    if not queue.empty():
        FaceTracker.llmIsWorkingFlag = False
        result = queue.get()
        print(f"result: {result}")
        if totalTime > FaceTracker.llmWaitTime:
            FaceTracker.llmWaitTime = totalTime
            changeLlmWaitTime(totalTime)
        #print(f"LLM Call took: {totalTime} seconds")
        FaceTracker.labelsLMM=result

    if FaceTracker.llmIsWorkingFlag:
        warningText=f"LLM: \"{FaceTracker.lastWord}\",aprox {int(totalTime/FaceTracker.llmWaitTime*100)}% done, {int(FaceTracker.llmWaitTime-totalTime)} seconds left"
        #print(dimensionsTop)
        FaceTracker.prettyPrintInCamera(theTopFrame, warningText,(int(dimensionsTop[1] / 2),dimensionsTop[0]-30 ), FaceTracker.redFrameColor,onCenter=True)#x and y are inverted in call

    if FaceTracker.showWritten:
        theTopFrame = showWhatsWritten(theTopFrame,dimensionsTop)

    #--------------------------
    # Menu System--------------
    #--------------------------
    if theCurrentSelection[0] < 0:#no option has been chosen
        if theCurrentSelection[1]=="MainMenu":
            GetMainMenu(totalN, theTopFrame,theCurrentSelection,centerOfContours,color,lettersColor)
        elif(theCurrentSelection[1]=="MultipleLetters"):
            GetCharacterDivisionMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection,centerOfContours,color,lettersColor)
        elif (theCurrentSelection[1] == "MultipleNumbers"):
            GetCharacterDivisionMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection,centerOfContours,color,lettersColor)
        elif (theCurrentSelection[1] == "MultipleSpecialChars"):
            GetCharacterDivisionMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection,centerOfContours,color,lettersColor)
        elif(theCurrentSelection[1]=="Quick"):
            DisplayOtherMenus(labelsQuick,labelsQuickOptions, totalN, theTopFrame,centerOfContours,color)
        elif (theCurrentSelection[1] == "MouseControl"):
            DisplayMouseMenu(labelsMouse, labelsMouseMenu, totalN, theTopFrame,centerOfContours,color)
            theCurrentSelection[0] = -1
        elif (theCurrentSelection[1] == "LLM"):
            DisplayOtherMenus(FaceTracker.labelsLMM,labelsLLMOptions, totalN, theTopFrame,centerOfContours,color)
    else:  # if an option has been chosen
        if(theCurrentSelection[1]=="MainMenu"):
            theCreatedLabelList=GetMainMenu(totalN,theTopFrame,theCurrentSelection,centerOfContours,color,lettersColor)
            # reset value and select new menu
            if theCurrentSelection[0]==0:#Quick
                print("Menu: " + str(labelsMainMenu[theCurrentSelection[0]]))
                theCurrentSelection = [-1, "Quick"]
            elif theCurrentSelection[0] == 1:#Backspace
                theCurrentSelection[0] = -1
                print("pressed: Backspace")
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
                if len(FaceTracker.whatsWritten)>0:
                    FaceTracker.whatsWritten = FaceTracker.whatsWritten[:-1]
            elif theCurrentSelection[0]==2:#LLM
                print(f"Menu: {str(labelsMainMenu[theCurrentSelection[0]])}, selection: {theCurrentSelection[0]}")
                theCurrentSelection = [-1, "LLM"]
            elif theCurrentSelection[0] == 3:#Space
                theCurrentSelection[0] = -1
                print("pressed: Space")
                keyboard.press(Key.space)
                keyboard.release(Key.space)
                FaceTracker.whatsWritten = FaceTracker.whatsWritten+" "
            elif theCurrentSelection[0] == 4:#Mouse Control
                theCurrentSelection = [-1, "MouseControl"]
                print("menu: " + str(labelsMainMenu[theCurrentSelection[0]]))
            elif theCurrentSelection[0] == 5:#ABC
                print(f"Letters Menu: {labelsABC}")
                theCurrentSelection = [-1,"MultipleLetters"]
                theCreatedLabelList = labelsABC
                theCreatedLabelList, theCurrentSelection = GetCharacterDivisionMenu(
                    labelsABC, totalN, theTopFrame, theCurrentSelection,centerOfContours,color,lettersColor)
            elif theCurrentSelection[0] == 6:  # Numbers
                print(f"Letters Menu: {labelsNumbers}")
                theCurrentSelection = [-1,"MultipleNumbers"]
                theCreatedLabelList = labelsNumbers
                theCreatedLabelList, theCurrentSelection = GetCharacterDivisionMenu(
                    labelsNumbers, totalN, theTopFrame, theCurrentSelection,centerOfContours,color,lettersColor)
            elif theCurrentSelection[0] == 7:#Special Characters
                print(f"Letters Menu: {labelsSpecial}")
                theCurrentSelection = [-1,"MultipleSpecialChars"]
                theCreatedLabelList = labelsSpecial
                theCreatedLabelList, theCurrentSelection = GetCharacterDivisionMenu(
                    labelsSpecial, totalN, theTopFrame, theCurrentSelection,centerOfContours,color,lettersColor)
        elif(theCurrentSelection[1]=="MouseControl"):
            DisplayMouseMenu(labelsMouse, labelsMouseMenu, totalN, theTopFrame,centerOfContours,color)
            #"Click","Back","Double Click","RightClick"
            if theCurrentSelection[0] == 2:  #"Back",
                theCurrentSelection = [-1, "MainMenu"]
                print("Selected Option: MainMenu")
            elif theCurrentSelection[0] == 1:#down
                mouse.move(0,mouseSpeed)
                print("mouse moved down: "+str(mouseSpeed))
            elif theCurrentSelection[0] == 0:  # Click
                mouse.press(Button.left)
                mouse.release(Button.left)
                theCurrentSelection[0] = -1
                print("mouse Left Clicked")
            elif theCurrentSelection[0] == 3:  # Left
                mouse.move(-mouseSpeed, 0)
                print("mouse moved left: " + str(mouseSpeed))
            elif theCurrentSelection[0] == 4:  # Double Click
                mouse.click(Button.left,2)
                theCurrentSelection[0] = -1
                print("mouse double clicked")
            elif theCurrentSelection[0] == 5:  # Up
                mouse.move(0,-mouseSpeed)
                print("mouse moved up: " + str(mouseSpeed))
            elif theCurrentSelection[0] == 6:  # RightClick
                mouse.press(Button.right)
                mouse.release(Button.right)
                theCurrentSelection[0] = -1
            elif theCurrentSelection[0] == 7:  # Right
                mouse.move(mouseSpeed, 0)
                print("mouse moved left: " + str(mouseSpeed))
        elif(theCurrentSelection[1]=="MultipleLetters" or theCurrentSelection[1]=="MultipleNumbers" or theCurrentSelection[1]=="MultipleSpecialChars"):
            #select character set

            print(f"label list 1: {theCreatedLabelList}")
            if theCurrentSelection[1]== -1:
                return theCurrentSelection,theCreatedLabelList
            elif (theCreatedLabelList[0]=="" and theCurrentSelection[1]=="MultipleLetters"):
                theCreatedLabelList=labelsABC
            elif (theCreatedLabelList[0]=="" and theCurrentSelection[1]=="MultipleNumbers"):
                theCreatedLabelList=labelsNumbers
            elif (theCreatedLabelList[0]=="" and theCurrentSelection[1]=="MultipleSpecialChars"):
                theCreatedLabelList=labelsSpecial

            #when selecting the letter menu
            if theCurrentSelection[0]==0:#"Quick", "", ""]
                theCurrentSelection = [-1, "Quick"]
                print("menu: "+str(labelsMainMenu[theCurrentSelection[0]]))
            elif theCurrentSelection[0] == 1:#BackSpace
                theCurrentSelection[0] = -1
                print("pressed: Backspace")
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
                if len(FaceTracker.whatsWritten)>0:
                    FaceTracker.whatsWritten = FaceTracker.whatsWritten[:-1]
            elif theCurrentSelection[0] == 2:#Back
                theCurrentSelection = [-1,"MainMenu"]
            #when selecting a letter or group of letters

            sizeOfLabelText=len(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)])
            if(sizeOfLabelText==1):
                print("pressed: " + str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))
                DisplayCharactersMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection,centerOfContours,color)
                if (str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))=='_':
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)
                    FaceTracker.whatsWritten = FaceTracker.whatsWritten + " "
                else:
                    keyboard.press(str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))
                    keyboard.release(str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))
                    FaceTracker.whatsWritten = FaceTracker.whatsWritten + str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)])
            elif (sizeOfLabelText>1):
                selectedFromListIndex=theCurrentSelection[0] - len(labelsLettersMenu)
                if(selectedFromListIndex<0):
                    selectedFromListIndex=0
                print(f"label list 2: {theCreatedLabelList}, selected from list: {selectedFromListIndex}")
                theCreatedLabelList, theCurrentSelection = GetCharacterDivisionMenu(
                    theCreatedLabelList[selectedFromListIndex], totalN, theTopFrame,theCurrentSelection,centerOfContours,color,lettersColor)
            theCurrentSelection[0] = -1
        elif(theCurrentSelection[1]=="Quick"):#"LLM","BackSpace","Back"
            DisplayOtherMenus(labelsQuick, labelsQuickOptions,totalN, theTopFrame,centerOfContours,color)
            if theCurrentSelection[0] == 0:
                theCurrentSelection = [-1, "LLM"]
            elif theCurrentSelection[0] == 1:
                print("pressed: Backspace")
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
                if len(FaceTracker.whatsWritten)>0:
                    FaceTracker.whatsWritten = FaceTracker.whatsWritten[:-1]
            elif theCurrentSelection[0] == 2:
                theCurrentSelection = [-1, "MainMenu"]
            elif(theCurrentSelection[0]>=len(labelsQuickOptions)):
                print("Typed Quick: "+labelsQuick[theCurrentSelection[0]-len(labelsQuickOptions)])
                keyboard.type(" "+labelsQuick[theCurrentSelection[0]-len(labelsQuickOptions)]+" ")
                FaceTracker.whatsWritten = FaceTracker.whatsWritten + " "+labelsQuick[theCurrentSelection[0]-len(labelsQuickOptions)]+" "
            theCurrentSelection[0] = -1
        elif(theCurrentSelection[1]=="LLM"):
            DisplayOtherMenus(labelsLMM,labelsLLMOptions, totalN, theTopFrame,centerOfContours,color)
            if theCurrentSelection[0] == 0:
                theCurrentSelection = [-1, "Quick"]
            elif theCurrentSelection[0] == 1:
                print("pressed: Backspace")
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
                if len(FaceTracker.whatsWritten)>0:
                    FaceTracker.whatsWritten = FaceTracker.whatsWritten[:-1]
            elif theCurrentSelection[0] == 2:
                theCurrentSelection = [-1, "MainMenu"]
            elif (theCurrentSelection[0] >= len(labelsQuickOptions)):
                print("Typed Quick LLM: " + FaceTracker.labelsLMM[theCurrentSelection[0] - len(labelsLLMOptions)])
                keyboard.type(" " + FaceTracker.labelsLMM[theCurrentSelection[0] - len(labelsLLMOptions)] + " ")
                FaceTracker.whatsWritten = FaceTracker.whatsWritten + " " + FaceTracker.labelsLMM[theCurrentSelection[0] - len(labelsLLMOptions)] + " "
                FaceTracker.lastWord=FaceTracker.labelsLMM[theCurrentSelection[0] - len(labelsLLMOptions)]
            theCurrentSelection[0] = -1
    return theCurrentSelection,theCreatedLabelList, theTopFrame

def DisplayMouseMenu(theLabelsList,theLabelsOptions,totalN,theTopFrame,centerOfContours,color):
    for i in range(totalN):
        # set option labels on topFrame to make them not transparent
        if i % 2 == 0:#If even, show options
            prettyPrintInCamera(theTopFrame, theLabelsOptions[int(math.floor(i/2))], centerOfContours[i], color)
        else:
            prettyPrintInCamera(theTopFrame, theLabelsList[int(math.floor(i/2))], centerOfContours[i], redFrameColor)

def DisplayCharactersMenu(theLabelsList, totalN, theTopFrame, theCurrentSelection,centerOfContours,color):
    #lettersPerArea = len(theLabelsList) / (totalN - len(labelsLettersMenu))
    lettersPerArea = math.ceil(len(theLabelsList) / (totalN - len(labelsLettersMenu)))
    for i in range(0,totalN):
        # set option labels on topFrame to make them not transparent
        if i < len(labelsLettersMenu):
            prettyPrintInCamera(theTopFrame, labelsLettersMenu[i], centerOfContours[i], color)
        else:
            prettyPrintInCamera(theTopFrame, theLabelsList[i-len(labelsLettersMenu)], centerOfContours[i], redFrameColor)

def DisplayOtherMenus(theLabelsList,theLabelsOptions, totalN, theTopFrame,centerOfContours,color):
    for i in range(totalN):
        # set option labels on topFrame to make them not transparent
        if i < len(theLabelsOptions):
            prettyPrintInCamera(theTopFrame, theLabelsOptions[i], centerOfContours[i], color)
        else:
            if i< (len(theLabelsOptions)+len(theLabelsList)):
                prettyPrintInCamera(theTopFrame,  theLabelsList[i-len(theLabelsOptions)], centerOfContours[i], redFrameColor)


def GetCharacterDivisionMenu(theLabelsList, totalN, theTopFrame, theCurrentSelection,centerOfContours,color,lettersColor):
    createdLabel = []
    lettersPerArea = math.ceil(len(theLabelsList) / (totalN - len(labelsLettersMenu)))
    if lettersPerArea>0:
        for i in range(totalN):
            # set option labels on topFrame to make them not transparent
            if i < len(labelsLettersMenu):
                prettyPrintInCamera(theTopFrame, labelsLettersMenu[i], centerOfContours[i],color)
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
                prettyPrintInCamera(theTopFrame, createdLabel[loopText-len(labelsLettersMenu)], centerOfContours[loopText], lettersColor)
            elif isinstance(createdLabel[loopText - len(labelsLettersMenu)][0], str):
                prettyPrintInCamera(theTopFrame, createdLabel[loopText-len(labelsLettersMenu)][0], centerOfContours[loopText], lettersColor)
                #print(f"createdLabel: {createdLabel}")
    return createdLabel,theCurrentSelection

def prettyPrintInCamera(topFrame, text, theOrg, theColor, lineType=cv2.LINE_AA, onCenter=False):
    text_size = cv2.getTextSize(text, font, fontScale, fontThickness)[0]
    paddingPercentage=.1
    proportionalPadding=[math.floor(text_size[0]*paddingPercentage),math.floor(text_size[1]*paddingPercentage)]
    # Create a background rectangle
    background_x1 = theOrg[0] - proportionalPadding[0]
    background_y1 = theOrg[1] - text_size[1] - proportionalPadding[1]
    background_x2 = theOrg[0] + text_size[0] + proportionalPadding[0]
    background_y2 = theOrg[1] + proportionalPadding[1]
    if onCenter:
        pixelsToSubstract = (math.floor(text_size[0]/2),math.floor(text_size[1]/2))
        cv2.rectangle(topFrame, (background_x1-pixelsToSubstract[0], background_y1-pixelsToSubstract[1]),
                      (background_x2-pixelsToSubstract[0], background_y2-pixelsToSubstract[1]), (255, 255, 255), -1)
        cv2.putText(topFrame, text, (theOrg[0]-pixelsToSubstract[0],theOrg[1]-pixelsToSubstract[1]), font, fontScale, theColor, fontThickness, lineType)
    else:
        cv2.rectangle(topFrame, (background_x1, background_y1), (background_x2, background_y2), (255, 255, 255), -1)
        cv2.putText(topFrame, text, theOrg, font, fontScale, theColor, fontThickness,lineType)
    return text_size


def GetMainMenu(totalN,theTopFrame,theCurrentSelection,centerOfContours,color,lettersColor):

    createdLabel = []
    for i in range(totalN):
        # set option labels on topFrame to make them not transparent
        if (totalN <= len(labelsMainMenu) and theCurrentSelection[1] == "MultipleNumbers"):
            createdLabel.append(labelsNumbers)
        elif (totalN <= len(labelsMainMenu) and theCurrentSelection[1] == "MultipleSpecialChars"):
            createdLabel.append(labelsSpecial)
        elif(totalN<=len(labelsMainMenu) and theCurrentSelection[1]=="MultipleLetters"):
            createdLabel.append(labelsABC)
        elif(totalN<=len(labelsMainMenu)):
            createdLabel.append(labelsABC)
        if i < len(labelsMainMenu):
            prettyPrintInCamera(theTopFrame, labelsMainMenu[i], centerOfContours[i], color, cv2.LINE_AA)
        else:
            # divide ABC between remaining areas
            lettersPerArea = math.ceil(len(labelsABC) / (totalN - len(labelsMainMenu)))
            startABCIndex = math.floor((i - len(labelsMainMenu)) * lettersPerArea)
            endABCIndex = math.ceil((i - len(labelsMainMenu)) * lettersPerArea + lettersPerArea)
            createdLabel.append(labelsABC[startABCIndex:endABCIndex])
            prettyPrintInCamera(theTopFrame, createdLabel[i - len(labelsMainMenu)], centerOfContours[i], lettersColor, cv2.LINE_AA)
    while len(createdLabel) < totalN:
        createdLabel.append("")
    return createdLabel

def GetSelectionLogic(theSelectionCurrentTime,theCurrentSelection,theSelected,thePrevSelected,ellipsePoly,nosePosition,contours):
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
                theSelectionCurrentTime = selectionCurrentTime + 1 / fps
                theSelected = idx
                if thePrevSelected != theSelected:
                    # change timer depending on area change
                    theSelectionCurrentTime = 0
                    thePrevSelected=theSelected
                    # print("Timer Reset Area Changed")
                elif theSelectionCurrentTime >= selectionWaitTime:
                    theSelected = idx
                    thePrevSelected = theSelected
                    break
    if (theSelected == -1) and (thePrevSelected != theSelected):
        theCurrentSelection[0] = thePrevSelected
        print("Selected: " + str(theCurrentSelection[0]))
        thePrevSelected = theSelected
    return theSelected,thePrevSelected,theCurrentSelection,theSelectionCurrentTime


def mainLoop(queue):

    totalOptionsN,mouseSpeed,selectionWaitTime,\
    labelsABC,labelsNumbers,labelsSpecial,labelsQuick,\
    fontScale,fontThickness,\
    camSizeX,camSizeY,showFPS,showWritten, llmContextSize,llmBatchSize,llmWaitTime,maxWhatsWrittenSize,showWrittenMode=GetConfigSettings()

    #FaceTracker.llm=loadLLM("zephyr-7b-beta.Q4_K_M.gguf",llmContextSize,llmBatchSize)
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    #Set camera
    if imageOrVideo:
        sourceTop=0
        if len(sys.argv) > 1:
            sourceTop = sys.argv[1]
        #print("SourceTop: "+str(sourceTop)+", sourceSide: "+str(sourceSide))
        cameraTopView = cv2.VideoCapture(sourceTop)
        # cameraTopView.set(cv2.CAP_PROP_FPS, 30.0)
        cameraTopView.set(cv2.CAP_PROP_FRAME_WIDTH, 640);#640
        cameraTopView.set(cv2.CAP_PROP_FRAME_HEIGHT, 640);#480
    else:
        cameraTopView = cv2.imread("testImages/Sofa2.jpg")

    while cv2.waitKey(1) != 27:  # Escape
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
        nosePosition = (50, 50)
        color = (255, 0, 0)# Blue color in BGR
        lettersColor=(50,50,255)

        if results.multi_face_landmarks:
            for faceLandMarks in results.multi_face_landmarks:
                #mpDraw.draw_landmarks(topFrame,faceLandMarks,mpFaceMesh.FACEMESH_FACE_OVAL, drawSpecs,drawSpecs)
                #get max size of face
                shape = topFrame.shape
                faceXmin,faceXmax,faceYmin,faceYmax,centerOfFaceX,centerOfFaceY=GetFaceSizeAndCenter(shape,faceLandMarks.landmark)

                for idx, landMark in enumerate(faceLandMarks.landmark):
                    if idx>44 and idx<46:#nose points:44 and 45
                        x = landMark.x
                        y = landMark.y
                        relative_x = int(x * shape[1])
                        relative_y = int(y * shape[0])
                        nosePosition=(relative_x, relative_y)

                        #circle on center of face
                        sizeOfFace=math.sqrt(math.pow(shape[1]*(faceXmax-faceXmin),2)+math.pow(shape[0]*(faceYmax-faceYmin),2))
                        radiusAsPercentage=int(0.1*sizeOfFace)
                        #print(centerOfFaceX,centerOfFaceY, faceXmin,faceXmax,faceYmin,faceYmax)
                        # cv2.putText(topFrame,str(idx), org, font,fontScale, color, thickness, cv2.LINE_AA)

                        # ---------GUI--------------
                        # create the n zones for buttons, geometry must be created by placing points in clockwise order
                        # -------------------------
                        ellipsePoly,contours,centerOfContours=GetGUI(uiFrame,radiusAsPercentage,totalOptionsN,centerOfFaceX,centerOfFaceY,nosePosition)
                        FaceTracker.currentSelection,FaceTracker.createdLabelsList,topFrame = GetMenuSystem (queue,topFrame,FaceTracker.totalOptionsN,
                                                                                                    FaceTracker.currentSelection,FaceTracker.createdLabelsList,
                                                                                                    centerOfContours,color,lettersColor,dimensionsTop)
                        # -------------------------
                        # Selection Logic----------
                        # -------------------------
                        FaceTracker.selected,FaceTracker.prevSelected,FaceTracker.currentSelection,FaceTracker.selectionCurrentTime =\
                            GetSelectionLogic(FaceTracker.selectionCurrentTime,FaceTracker.currentSelection,FaceTracker.selected,FaceTracker.prevSelected,ellipsePoly,nosePosition,contours)


        # FPS calculations
        FaceTracker.new_frame_time = time.time()
        if FaceTracker.new_frame_time != FaceTracker.prev_frame_time:
            FaceTracker.prevFps = FaceTracker.fps
            FaceTracker.fps = 1 / (FaceTracker.new_frame_time - FaceTracker.prev_frame_time)
            FaceTracker.prev_frame_time = FaceTracker.new_frame_time

        #Display
        #cv2.imshow("top frame", topFrame)
        combinedCalibImage = topFrame.copy()
        uiFrame = cv2.addWeighted(uiFrame, alpha, combinedCalibImage, 1 - alpha, 0)
        cv2.imshow("UIframe", uiFrame)

if __name__ == '__main__':
    api_url = 'https://api.example.com/data'  # Replace with your API URL
    FaceTracker.llmIsWorkingFlag=True
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=mainLoop, args=(queue,))
    p.start()
    p.join()