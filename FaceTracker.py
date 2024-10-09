import math

import cv2
import mediapipe as mp
import time
import sys

import numpy as np
#from numpy.array_api import astype

import pynput
from pynput.keyboard import Key, Controller as keyController
from pynput.mouse import Button, Controller as mouseController

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
labelsMainMenu=["LLM","Quick", "BackSpace","Space","Mouse", "ABC","Numbers","Special Chars"]
labelsLettersMenu=["Quick", "BackSpace", "Back"]
#------------------
#------------------
labelsMouseMenu=["Back","Click","Double Click","RightClick"]
labelsMouse=["Down","Left","Up","Right"]
#------------------
labelsQuickOptions=["Back","BackSpace"]
#------------------
configFilePath='./config.txt'

#------------------
#menu selection init hidden variables
createdLabelsList=[]
selectionCurrentTime=0
currentSelection=[-1,"MainMenu"]#first is option chosen, second is how deep
selected = -1
prevSelected = -1

#ui variables
greenFrameColor = (0, 255, 0)  # BGR
redFrameColor = (0, 0, 255)  # BGR
alpha = 0.3
font=cv2.FONT_HERSHEY_SIMPLEX


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
    with open(configFilePath, 'r') as file:
        for line in file:
            # Strip whitespace and check if the line starts with #
            line = line.strip()
            if line.startswith('#') or not line:
                continue  # Skip comments and empty lines

            key, value = line.strip().split('=',1)
            if key == "totalOptionsN":
                totalOptionsN = int(value)
            elif key == "mouseSpeed":
                mouseSpeed = int(value)
            elif key == "selectionWaitTime":
                selectionWaitTime = float(value)
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

    # Print the variables
    print(f"totalOptionsN: {totalOptionsN}, mouseSpeed: {mouseSpeed}, selectionWaitTime: {selectionWaitTime}"
          f", labelsABC: {labelsABC}, labelsQuick: {labelsQuick}, fontScale: {fontScale}"
          f", fontThickness: {fontThickness}, camSizeX: {camSizeX}, camSizeY: {camSizeY}")
    return totalOptionsN,mouseSpeed,selectionWaitTime,labelsABC,labelsNumbers,labelsSpecial,labelsQuick,fontScale,fontThickness,camSizeX,camSizeY

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


def GetGUI(theUIFrame,radiusAsPercent,totalN,centerFaceX,centerFaceY):
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


def GetMenuSystem(theUIFrame, theTopFrame, totalN,theCurrentSelection,theCreatedLabelList):
    cv2.putText(theTopFrame, "FPS: " + str(int(fps)), org=(int(dimensionsTop[0] / 2), 20),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
    #--------------------------
    # Menu System--------------
    #--------------------------
    if theCurrentSelection[0] < 0:#no option has been chosen
        # ["Quick", "BackSpace","Space","Mouse", "ABC","Numbers","Special Chars"]
        if theCurrentSelection[1]=="MainMenu":
            GetMainMenu(totalN, theTopFrame,theCurrentSelection)
        elif(theCurrentSelection[1]=="MultipleLetters"):
            GetLettersMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection)
        elif (theCurrentSelection[1] == "MultipleNumbers"):
            GetLettersMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection)
        elif (theCurrentSelection[1] == "MultipleSpecialChars"):
            GetLettersMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection)
        elif(theCurrentSelection[1]=="Quick"):
            DisplayOtherMenus(labelsQuick,labelsQuickOptions, totalN, theTopFrame)
        elif (theCurrentSelection[1] == "MouseControl"):
            DisplayMouseMenu(labelsMouse, labelsMouseMenu, totalN, theTopFrame)
            theCurrentSelection[0] = -1
        elif (theCurrentSelection[1] == "LLM"):
            #todo
            theCurrentSelection[1] == "MainMenu"
            theCurrentSelection[0] = -1
    else:  # if an option has been chosen
        # ["LLM","Quick", "BackSpace","Space","Mouse", "ABC","Numbers","Special Chars"]
        if(theCurrentSelection[1]=="MainMenu"):
            theCreatedLabelList=GetMainMenu(totalN,theTopFrame,theCurrentSelection)
            # reset value and select new menu
            if theCurrentSelection[0]==0:#LLM
                theCurrentSelection = [-1, "LLM"]
                print("menu: " + str(labelsMainMenu[theCurrentSelection[0]]))
            if theCurrentSelection[0]==1:#Quick
                theCurrentSelection = [-1, "Quick"]
                print("menu: " + str(labelsMainMenu[theCurrentSelection[0]]))
            elif theCurrentSelection[0] == 2:#Backspace
                theCurrentSelection[0] = -1
                print("pressed: Backspace")
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
            elif theCurrentSelection[0] == 3:#Space
                theCurrentSelection[0] = -1
                print("pressed: Space")
                keyboard.press(Key.space)
                keyboard.release(Key.space)
            elif theCurrentSelection[0] == 4:#Mouse Control
                theCurrentSelection = [-1, "MouseControl"]
                print("menu: " + str(labelsMainMenu[theCurrentSelection[0]]))
            elif theCurrentSelection[0] == 5:#ABC
                print(f"Letters Menu: {labelsABC}")
                theCurrentSelection = [-1,"MultipleLetters"]
                theCreatedLabelList = labelsABC
            elif theCurrentSelection[0] == 6:  # Numbers
                print(f"Letters Menu: {labelsNumbers}")
                theCurrentSelection = [-1,"MultipleNumbers"]
                theCreatedLabelList = labelsNumbers
            elif theCurrentSelection[0] == 7:#Special Characters
                print(f"Letters Menu: {labelsSpecial}")
                theCurrentSelection = [-1,"MultipleSpecialChars"]
                theCreatedLabelList = labelsSpecial
        elif(theCurrentSelection[1]=="MouseControl"):
            if theCurrentSelection[0] == 0:  #"Back",
                theCurrentSelection = [-1, "MainMenu"]
                print("Selected Option: MainMenu")
            elif theCurrentSelection[0] == 1:#down
                mouse.move(0,mouseSpeed)
                print("mouse moved down: "+str(mouseSpeed))
            elif theCurrentSelection[0] == 2:  # Click
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
                theCurrentSelection[0] = -1
                print("menu: "+str(labelsMainMenu[theCurrentSelection[0]]))
            elif theCurrentSelection[0] == 1:#BackSpace
                theCurrentSelection[0] = -1
                print("pressed: Backspace")
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
            elif theCurrentSelection[0] == 2:#Back
                theCurrentSelection = [-1,"MainMenu"]
            #when selecting a letter or group of letters

            sizeOfLabelText=len(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)])
            if(sizeOfLabelText==1):
                print("pressed: " + str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))
                DisplayCharactersMenu(theCreatedLabelList, totalN, theTopFrame, theCurrentSelection)
                if (str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))=='_':
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)
                else:
                    keyboard.press(str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))
                    keyboard.release(str(theCreatedLabelList[theCurrentSelection[0] - len(labelsLettersMenu)]))
            elif (sizeOfLabelText>1):
                selectedFromListIndex=theCurrentSelection[0] - len(labelsLettersMenu)
                if(selectedFromListIndex<0):
                    selectedFromListIndex=0
                print(f"label list 2: {theCreatedLabelList}, selected from list: {selectedFromListIndex}")
                theCreatedLabelList, theCurrentSelection = GetLettersMenu(
                    theCreatedLabelList[selectedFromListIndex], totalN, theTopFrame,theCurrentSelection)
            theCurrentSelection[0] = -1
        elif(theCurrentSelection[1]=="Quick"):
            DisplayOtherMenus(labelsQuick, labelsQuickOptions,totalN, theTopFrame)
            if theCurrentSelection[0] == 0:  # ["BackSpace","Back"]
                print("pressed: Backspace")
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
            elif theCurrentSelection[0] == 1:
                theCurrentSelection = [-1, "MainMenu"]
            elif(theCurrentSelection[0]>=len(labelsQuickOptions)):
                print("Typed Quick: "+labelsQuick[theCurrentSelection[0]-len(labelsQuickOptions)])
                keyboard.type(" "+labelsQuick[theCurrentSelection[0]-len(labelsQuickOptions)]+" ")
            theCurrentSelection[0] = -1
        elif(theCurrentSelection[1]=="LLM"):
            #todo
            theCurrentSelection = [-1, "MainMenu"]

    return theCurrentSelection,theCreatedLabelList

def DisplayMouseMenu(theLabelsList,theLabelsOptions,totalN,theTopFrame):
    for i in range(totalN):
        # set option labels on topFrame to make them not transparent
        if i % 2 == 0:#If even, show options
            prettyPrintInCamera(theTopFrame, theLabelsOptions[int(math.floor(i/2))], centerOfContours[i], color)
        else:
            prettyPrintInCamera(theTopFrame, theLabelsList[int(math.floor(i/2))], centerOfContours[i], redFrameColor)

def DisplayCharactersMenu(theLabelsList, totalN, theTopFrame, theCurrentSelection):
    #lettersPerArea = len(theLabelsList) / (totalN - len(labelsLettersMenu))
    lettersPerArea = math.ceil(len(theLabelsList) / (totalN - len(labelsLettersMenu)))
    for i in range(0,totalN):
        # set option labels on topFrame to make them not transparent
        if i < len(labelsLettersMenu):
            prettyPrintInCamera(theTopFrame, labelsLettersMenu[i], centerOfContours[i], color)
        else:
            prettyPrintInCamera(theTopFrame, theLabelsList[i-len(labelsLettersMenu)], centerOfContours[i], redFrameColor)

def DisplayOtherMenus(theLabelsList,theLabelsOptions, totalN, theTopFrame):
    for i in range(totalN):
        # set option labels on topFrame to make them not transparent
        if i < len(theLabelsOptions):
            prettyPrintInCamera(theTopFrame, theLabelsOptions[i], centerOfContours[i], color)
        else:
            if i< (len(theLabelsOptions)+len(theLabelsList)):
                prettyPrintInCamera(theTopFrame,  theLabelsList[i-len(theLabelsOptions)], centerOfContours[i], redFrameColor)


def GetLettersMenu(theLabelsList,totalN,theTopFrame,theCurrentSelection):
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
        print(f"createdLabel: {createdLabel}")
        for loopText in range(len(labelsLettersMenu),totalN):
            #print(loopText)
            prettyPrintInCamera(theTopFrame, createdLabel[loopText-len(labelsLettersMenu)], centerOfContours[loopText], lettersColor)
    return createdLabel,theCurrentSelection

def prettyPrintInCamera(topFrame, text, theOrg, theColor, lineType=cv2.LINE_AA):
    text_size = cv2.getTextSize(text, font, fontScale, fontThickness)[0]
    paddingPercentage=.1
    proportionalPadding=[math.floor(text_size[0]*paddingPercentage),math.floor(text_size[1]*paddingPercentage)]
    # Create a background rectangle
    background_x1 = theOrg[0] - proportionalPadding[0]
    background_y1 = theOrg[1] - text_size[1] - proportionalPadding[1]
    background_x2 = theOrg[0] + text_size[0] + proportionalPadding[0]
    background_y2 = theOrg[1] + proportionalPadding[1]
    cv2.rectangle(topFrame, (background_x1, background_y1), (background_x2, background_y2), (255, 255, 255), -1)
    cv2.putText(topFrame, text, theOrg, font, fontScale, theColor, fontThickness,lineType)


def GetMainMenu(totalN,theTopFrame,theCurrentSelection             ):
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

def GetSelectionLogic(theSelectionCurrentTime,theCurrentSelection,theSelected,thePrevSelected):
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



totalOptionsN,mouseSpeed,selectionWaitTime,labelsABC,labelsNumbers,labelsSpecial,labelsQuick,fontScale,fontThickness,camSizeX,camSizeY=GetConfigSettings()

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
                    ellipsePoly,contours,centerOfContours=GetGUI(uiFrame,radiusAsPercentage,totalOptionsN,centerOfFaceX,centerOfFaceY)
                    currentSelection,createdLabelsList = GetMenuSystem (uiFrame,topFrame,totalOptionsN,currentSelection,createdLabelsList)
                    # -------------------------
                    # Selection Logic----------
                    # -------------------------
                    selected,prevSelected,currentSelection,selectionCurrentTime =GetSelectionLogic(selectionCurrentTime,currentSelection,selected,prevSelected)


    # FPS calculations
    new_frame_time = time.time()
    if new_frame_time != prev_frame_time:
        prevFps = fps
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time



    #Display
    #cv2.imshow("top frame", topFrame)
    combinedCalibImage = topFrame.copy()
    uiFrame = cv2.addWeighted(uiFrame, alpha, combinedCalibImage, 1 - alpha, 0)
    cv2.imshow("UIframe", uiFrame)
