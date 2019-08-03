# coding=utf-8
from naoqi import ALProxy
from PIL import Image
import numpy as np
import cv2
import copy



#预定义量
localIp = ""
#1>部件名称列表
#1.1>键
#1.1.1>肢体
(head, lArm, rArm, lLeg, rLeg) = (
 "Head", "LArm", "RArm", "LLeg", "RLeg")
#1.1.2>部位
(shoulder, elbow, wrist, hand,
 hip, knee, ankle) = range(0, 7)
#1.1.3器官，待补充
(roll, yaw, pitch, yawPitch,
 eye0, eye1, 
 mouth, ear, skin, joint) = range(0, 10)
#1.2>字典
body = {
    head: {#头部
        pitch: "HeadPitch",
        yaw: "HeadYaw",
		eye0: "0",
		eye1: "1",
		mouth: {},#待补充
		ear: {},#待补充
		skin: {}},#待补充
    lArm: {#左臂
		shoulder: {	
            roll: "LShoulderRoll",
            pitch: "LShoulderPitch"},
		elbow: {
            roll: "LElbowRoll",
			yaw: "LElbowYaw"},
        wrist: {yaw: "LWristYaw"},
        hand: "LHand",
		skin: {}},#待补充
    rArm: {#右臂
		shoulder: {
            roll: "RShoulderRoll",
            pitch: "RShoulderPitch"},
		elbow: {
            roll: "RElbowRoll",
			yaw: "RElbowYaw"},
        wrist: {yaw: "RWristYaw"},
        hand: "RHand",
		skin: {}},#待补充
    lLeg: {#左腿
		hip: {
            roll: "LHipRoll",
            pitch: "LHipPitch",
            yawPitch: "LHipYawPitch"},
        knee: "LKneePitch",
		ankle: {
            roll: "LAnkleRoll",
            pitch: "LAnklePitch"},
		skin: {}},#待补充
    rLeg: {#右腿
		hip: {
            roll: "RHipRoll",
            pitch: "RHipPitch",
            yawPitch: "RHipYawPitch"},
        knee: "RKneePitch",
		ankle: {
            roll: "RAnkleRoll",
            pitch: "RAnklePitch"},
		skin: {}},#待补充
	mouth: {},#胸前的超声波发生器，待补充
	ear: {}}#胸前的超声波接收器，待补充
#2>预定义姿势列表，P65
posture = [
    "StandInit",
    "SitRelax",
    "StandZero",
    "LyingBelly",
    "Stand",
    "Crouch",
    "Sit",
    "LyingBack"]
#3>步态参数名列表，P78
moveConfig = [
    "MaxStepX",
    "MinStepX",
    "MaxStepY",
    "MaxStepTheta",
    "MaxStepFrequency",
    "MinStepPeriod",
    "MazStepPeriod",
    "StepHeight",
    "TorsoWx",
    "TorsoWy",
    "FootSeparation",
    "MinFootSeparation"]
#4>模块列表
module = [
	"ALPhotoCapture",
	"ALVideoRecorder",
	"ALVideoDevice",
    "ALAudioPlayer",
    "ALAudioDevice",
    "ALTextToSpeech",
    "ALRobotPosture",
    "ALMotion",
    "ALFaceDetection"]
#5>分辨率列表
resolution = {
    "40x30": 8,
    "40x60": 7,
    "160x120": 0,
    "320x240": 1,
    "640x480": 2,
    "1280x960": 3}
resolutionRank = {
    0: "40x30",
    1: "40x60",
    2: "160x120",
    3: "320x240",
    4: "640x480",
    5: "1280x960"}
#6>色彩空间列表
colorSpace = {
    "Yuv": 0,
    "yUv": 1,
    "yuV": 2,
    "Rgb": 3,
    "rGb": 4,
    "rgB": 5,
    "Hsy": 6,
    "hSy": 7,
    "hsY": 8,
    "YUV422": 9,
    "YUV": 10,
    "RGB": 11,
    "HSY": 12,
    "BGR": 13,
    "YYCbCr": 14}
codeToColorSpace = {
    0: "Yuv",
    1: "yUv",
    2: "yuV",
    3: "Rgb",
    4: "rGb",
    5: "rgB",
    6: "Hsy",
    7: "hSy",
    8: "hsY",
    9: "YUV422",
    10: "YUV",
    11: "RGB",
    12: "HSY",
    13: "BGR",
    14: "YYCbCr"}





#标记类
class Mark(object):
    "定义为可以标记NAO中所有的对象的类"
    def __init__(self, ip, port = 9559, module = "", device = "", handle = ""):
        self.ip = ip
        self.port = port
        self.module = module
        self.device = device
        self.handle = handle





#全局函数支持
#1>萝卜模块
def robotModule(mark):
    "得到基础模块，如果该模块原来已经创建过则返回原来的返回的那个模块（不过该功能尚无时间实现）"
    if mark.module == "":
        print("Error in robotModule()\n")
        raise#抛出异常
    if mark.ip == localIp:
        return ALProxy(mark.module)
    else:
        return ALProxy(mark.module, mark.ip, mark.port)
#2>图像识别
def colorDetection(image, interval):
    """过滤连颜色，只获取interval范围内的颜色
       interval类型：([B, R, G], [B, R, G])，分别表示，下限和上限"""
    (lower, upper) = interval
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)
def edgeDetection(image):
    "边缘检测"
    blurred = cv2.GaussianBlur(image,(3,3),0)
    gray=cv2.cvtColor(blurred,cv2.COLOR_RGB2GRAY)
    xgrad=cv2.Sobel(gray,cv2.CV_16SC1,1,0)
    ygrad=cv2.Sobel(gray,cv2.CV_16SC1,0,1)
    #50和150参数必须符合1：3或者1：2
    return cv2.Canny(xgrad,ygrad,50,150)





#器官类
class Organ(object):
    def __init__(self, mark):
        self.mark = copy.copy(mark)

#眼类
class Eye(Organ):
    "所有视频接收器"
    def __init__(self, mark):
        super(Eye, self).__init__(mark)
        mark.module = "ALPhotoCapture"
        self.photoCaptureModule = robotModule(mark)
        mark.module = "ALVideoRecorder"
        self.videoRecorderModule = robotModule(mark)
        mark.module = "ALVideoDevice"
        self.videoDeviceModule = robotModule(mark)
        self.resolution = resolution[resolutionRank[4]]
        self.colorSpace = colorSpace["RGB"]
        self.open()
    def __del__(self):
       self.close()
    #以下封装了部分厂商的接口，可随时补充
    def open(self):
        id = int(self.mark.device)
        self.photoCaptureModule.setCameraID(id)#P141
        self.videoRecorderModule.setCameraID(id)#P142
        self.videoDeviceModule.setActiveCamera(id)#P145
        self.videoDeviceModule.openCamera(id)#P145
        self.subscribe()
    def close(self):
        self.videoDeviceModule.closeCamera(self.getID())#P145
        self.unsubscribe()
    def getID(self):
        return self.photoCaptureModule.getCameraID()#P141
    def setID(self, ID):
        self.close()
        self.mark.device = str(ID)
        self.open()
    def getCaptureInterval(self):
        return self.photoCaptureModule.getCaptureInterval()#P141
    def setCaptureInterval(self, interval):
        self.photoCaptureModule.setCaptureInterval(interval)#P141
    def setColorSpace(self, colorSpace):
        self.colorSpace = colorSpace
        self.photoCaptureModule.setColorSpace(colorSpace)#P141
        self.videoRecorderModule.setColorSpace(colorSpace)#P142
    def getColorSpaceCode(self):
        return self.colorSpace
    def getColorSpaceName(self):
        return codeToColorSpace[self.getColorSpaceCode()]
    def setPictureFormat(self, pictureFormat):
        self.photoCaptureModule.setPictureFormat(pictureFormat)#P141
    def setResolution(self, resolution):
        self.resolution = resolution
        self.photoCaptureModule.setResolution(resolution)#P141
        self.videoRecorderModule.setResolution(resolution)#P142
    def getResolution(self):
        return self.resolution
    def takePicture(self, folderPath, fileName):
        self.photoCaptureModule.takePicture(folderPath, fileName)#P141
    def takePictures(self, numberOfPictures, folderPath, fileName):
        self.photoCaptureModule.takePictures(numberOfPictures, folderPath, fileName)#P141
    def getFrameRate(self):
        return self.videoRecorderModule.getFrameRate()#P142
    def setFrameRate(self, frameRate):
        self.videoRecorderModule.setFrameRate(frameRate)#P142
    def setVideoFormat(self, videoFormat):
        self.videoRecorderModule.setVideoFormat(videoFormat)#P142
    def startRecording(self):
        self.videoRecorderModule.startRecording()#P142
    def isRecording(self):
        return self.videoRecorderModule.isRecording()#P142
    def stopRecording(self):
        self.videoRecorderModule.stopRecording()#P142
    def subscribe(self):
        self.mark.handle = self.videoDeviceModule.subscribeCamera("Eye" + self.mark.device, self.getID(), self.getResolution(), self.getColorSpaceCode(), self.getFrameRate())#P145
        print(self.mark.handle)
    def getImage(self):
        if self.mark.ip == localIp:
            return self.videoDeviceModule.getImageLocal(self.mark.handle)#P145
        else:
            return self.videoDeviceModule.getImageRemote(self.mark.handle)#P145
    def releaseImage(self):
        self.videoDeviceModule.releaseImage(self.mark.handle)
    def unsubscribe(self):
        self.videoDeviceModule.unsubscribe(self.mark.handle)#P145
        print("Deleted Eye" + self.mark.device + " Handle: " + self.mark.handle)
    #以下为自定义接口
    def getImageToPIL(self):
        "仅实现了RGB类型的"
        self.releaseImage()
        naoImage = self.getImage()
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]
        return Image.frombytes(self.getColorSpaceName(), (imageWidth, imageHeight), array)
    def getImageToCV2(self):
        "仅实现了BRG类型的"
        return cv2.cvtColor(np.array(self.getImageToPIL()),cv2.COLOR_RGB2BGR)

#口类
class Mouth(Organ):
    "所有音频发生器，包括超声波发生器等"
    def __init__(self, mark):
        super(Mouth, self).__init__(mark)
        mark.module = "ALAudioPlayer"
        self.audioPlayerModule = robotModule(mark)
        mark.module = "ALAudioDevice"
        self.audioDeviceModule = robotModule(mark)
        mark.module = "ALTextToSpeech"
        self.textToSpeechModule = robotModule(mark)
    #以下封装了部分厂商的接口，可随时补充
    def playFile(self, filename):
        self.audioPlayerModule.playFile(filename)#P100
    def goTo(self, taskId, position):
        self.audioPlayerModule.goTo(taskId, position)#P100
    def getFileLength(self, taskId):
        return self.audioPlayerModule.getFileLength(taskId)#P100
    def loadFile(self, fileName):
        self.audioPlayerModule.loadFile(fileName)#P100
    def play(self, taskId, volume, pan):
        self.audioPlayerModule.play(taskId, volume, pan)#P100
    def playFileFromPosition(self, fileName, position, volume, pan):
        self.audioPlayerModule.playFileFromPosition(fileName, position, volume, pan)#P100
    def enableEnergyComputation(self):
        self.audioDeviceModule.enableEnergyComputation()#P113
    def disableEnergyComputation(self):
        self.audioDeviceModule.disableEnergyComputation()#P113
    def getMicEnergy(self):#P113
        if self.mark.device == "Front":
            return self.audioDeviceModule.getFrontMicEnergy()
        elif self.mark.device == "Rear":
            return self.audioDeviceModule.getRearMicEnergy()
        elif self.mark.device == "Left":
            return self.audioDeviceModule.getLeftMicEnergy()
        elif self.mark.device == "Right":
            return self.audioDeviceModule.getRightEnergy()
        else:
            print("Error in getMicEnergy()")
            raise
    def muteAudioOut(self, mute):
        self.audioDeviceModule.muteAudioOut(mute)#P113
    def setVolume(self, volume):
        self.textToSpeechModule.setVolume(volume)#P125
    #以下为自定义接口

#耳类
class Ear(Organ):
    "所有音频接收器，包括超声波接收器等"
    def __init__(self, mark):
        super(Ear, self).__init__(mark)
        mark.module = "ALAudioPlayer"
        self.audioPlayerModule = robotModule(mark)
        mark.module = "ALAudioDevice"
        self.audioDeviceModule = robotModule(mark)
    #以下封装了部分厂商的接口，可随时补充
    def getCurrentPosition(self, taskId):
        return self.audioPlayerModule.getCurrentPosition(taskId)#P100
    def getOutputVolume(self):
        return self.audioDeviceModule.getOutputVolume()#P113
    def setFileAsInput(self, fileName):
        self.audioDeviceModule.setFileAsInput(fileName)#P113
    def startMicrophonesRecording(self, fileName):
        self.audioDeviceModule.startMicrophonesRecording(fileName)#P113
    def stopMicrophonesRecording(self):
        self.audioDeviceModule.stopMicrophoneRecording()#P113
    #以下为自定义接口

#皮肤类
class Skin(Organ):
    "所有压力接收器"
    def __init__(self, mark):
        super(Skin, self).__init__(mark)
    #以下封装了部分厂商的接口，可随时补充
    #以下为自定义接口

#关节类
class Joint(Organ):
    "所有控制运动的发生器和接收器"
    def __init__(self, mark):
        super(Joint, self).__init__(mark)
        mark.module = "ALMotion"
        self.motionModule = robotModule(mark)
    #以下封装了部分厂商的接口，可随时补充
    def setStiffness(self, stiffness):
        self.motionModule.setStiffnesses(self.mark.device, stiffness)#非阻塞调用，P68
    def getStiffness(self):
        self.motionModule.getStiffnesses(self.mark.device)#P68
    def angleInterpolation(self, angleLists, timeLists, isAbsolute):
        self.motionModule.angleInterpolation(self.mark.device, angleLists, timeLists, isAbsolute)#阻塞调用，P71
    def angleInterpolationWithSpeed(self, targetAngle, maxSpeedFraction):
        self.motionModule.angleInterpolationWithSpeed(self.mark.device, targetAngle, maxSpeedFraction)#阻塞调用，P71
    def angleInterpolationBezier(self, times, controlPoint):
        self.motionModule.angleInterpolationBezier(self.mark.device, times, controlPoint)#阻塞调用，P71
    def setAngle(self, angle, maxSpeedFraction):
        self.motionModule.setAngles(self.mark.device, angle, maxSpeedFraction)#非阻塞调用，P72
    def changeAngle(self, angle, maxSpeedFraction):
        self.motionModule.changeAngles(self.mark.device, angle, maxSpeedFraction)#非阻塞调用，P72
    def getAngle(self, useSensors):
        self.motionModule.getAngles(self.mark.device, useSensors)#P72
    #以下为自定义接口







#基础系统类
class BaseSystem(object):
    "运动系统和感知系统共有的部分放在这"
    def __init__(self, mark):
        self.mark = copy.copy(mark)
        mark.module = "ALRobotPosture"
        self.postureModule = robotModule(mark)
        mark.module = "ALMotion"
        self.motionModule = robotModule(mark)
        mark.module = "ALTextToSpeech"
        self.textToSpeechModule = robotModule(mark)
        #关节类
		#  头部关节
        mark.device = body[head][yaw]
        self.headYaw = Joint(mark)
        mark.device = body[head][pitch]
        self.headPitch = Joint(mark)
		#  左臂关节
        mark.device = body[lArm][shoulder][pitch]
        self.lArmShoulderPitch = Joint(mark)
        mark.device = body[lArm][shoulder][roll]
        self.lArmShoulderRoll = Joint(mark)
        mark.device = body[lArm][elbow][yaw]
        self.lArmElbowYaw = Joint(mark)
        mark.device = body[lArm][elbow][roll]
        self.lArmElbowRoll = Joint(mark)
        mark.device = body[lArm][wrist][yaw]
        self.lArmWristYaw = Joint(mark)
		#  右臂关节
        mark.device = body[rArm][shoulder][pitch]
        self.rArmShoulderPitch = Joint(mark)
        mark.device = body[rArm][shoulder][roll]
        self.rArmShoulderRoll = Joint(mark)
        mark.device = body[rArm][elbow][yaw]
        self.rArmElbowYaw = Joint(mark)
        mark.device = body[rArm][elbow][roll]
        self.rArmElbowRoll = Joint(mark)
        mark.device = body[rArm][wrist][yaw]
        self.lArmWristYaw = Joint(mark)
		#  左腿关节
        mark.device = body[lLeg][hip][yawPitch]
        self.lLegYawPitch = Joint(mark)
        mark.device = body[lLeg][hip][roll]
        self.lLegHipRoll = Joint(mark)
        mark.device = body[lLeg][hip][pitch]
        self.lLegHipPitch = Joint(mark)
        mark.device = body[lLeg][knee][pitch]
        self.lLegKneePitch = Joint(mark)
        mark.device = body[lLeg][ankle][pitch]
        self.lLegAnklePitch = Joint(mark)
        mark.device = body[lLeg][ankle][roll]
        self.lLegAnkleRoll = Joint(mark)
		#  右腿关节
        mark.device = body[rLeg][hip][yawPitch]
        self.rLegYawPitch = Joint(mark)
        mark.device = body[rLeg][hip][roll]
        self.rLegHipRoll = Joint(mark)
        mark.device = body[rLeg][hip][pitch]
        self.rLegHipPitch = Joint(mark)
        mark.device = body[rLeg][knee][pitch]
        self.rLegKneePitch = Joint(mark)
        mark.device = body[rLeg][ankle][pitch]
        self.rLegAnklePitch = Joint(mark)
        mark.device = body[rLeg][ankle][roll]
        self.rLegAnkleRoll = Joint(mark)

#运动系统类
class ExerciseSystem(BaseSystem):
    "需要发生类器官协同运动的放这"
    def __init__(self, mark):
        super(ExerciseSystem, self).__init__(mark)
        #口类
        self.mouth = Mouth(mark)
    #以下封装了部分厂商的接口，可随时补充
    def wakeUp(self):
        self.motionModule.wakeUp()#P68
    def rest(self):
        self.motionModule.rest()#P68
    def moveInit(self):
        self.motionModule.moveInit()#阻塞调用，P84
    def goToPosture(self, postureName, speed = 1.0):
        self.postureModule.goToPosture(postureName, speed)#阻塞调用，P66
    def applyPosture(self, postureName, speed = 1.0):
        self.postureModule.applyPosture(postureName, speed)#阻塞调用，P66
    def stopMove(self):
        self.postureModule.stopMove()#P66
    def setStiffnesses(self, names, stiffnesses):
        self.motionModule.setStiffnesses(names, stiffnesses)#非阻塞调用，P68
    def stiffnessInterpolation(self, names, stiffnessLists, timeLists):
        self.motionModule.stiffnessInterpolation(names, stiffnessLists, timeLists)#阻塞调用，P68
    def angleInterpolationBezier(self, jointNames, times, controlPoint):
        self.motionModule.angleInterpolationBezier(jointNames, times, controlPoint)#阻塞调用，P71
    def setAngles(self, jointNames, angles, maxSpeedFraction):
        self.motionModule.setAngles(jointNames, angles, maxSpeedFraction)#非阻塞调用，P72
    def changeAngles(self, jointNames, angles, maxSpeedFraction):
        self.motionModule.changeAngles(jointNames, angles, maxSpeedFraction)#非阻塞调用，P72
    def getAngles(self, jointNames, useSensors):
        self.motionModule.getAngles(jointNames, useSensors)#P72
    def closeHand(self, handName):
        self.motionModule.closeHand(handName)#阻塞调用，P72
    def openHand(self, handName):
        self.motionModule.openHand(handName)#阻塞调用，P72
    def moveTo(self, controlPoints):
        self.motionModule.moveTo(controlPoints)#阻塞调用，P81
    def move(self, x, y, theta, moveConfig):
        self.motionModule.move(x, y, theta, moveConfig)#非阻塞调用，P82
    def moveToward(self, x, y, theta, moveConfig):
        self.motionModule.moveToward(x, y, theta, moveConfig)#非阻塞调用，P83
    def setFootSteps(self, legName, footSteps, timeList, clearExisting):
        self.motionModule.setFootSteps(legName, footSteps, timeList, clearExisting)#非阻塞调用，P85
    def setFootSetpsWithSpeed(self, legName, footSteps, fractionMaxSpeed, clearExisting):
        self.motionModule.setFootStepsWithSpeed(legName, footSteps, fractionMaxSpeed, clearExisting)#阻塞调用，P85
    def waitUntilMoveIsFinished(self):
        self.motionModule.waitUntilMoveIsFinished()#阻塞调用，P84
    def setMoveArmsEnabled(self, leftArmEnable, rightArmEnable):
        self.motionModule.setMoveArmsEnabled(leftArmEnable, rightArmEnable)#阻塞调用，P84
    def say(self, text):
        self.textToSpeechModule.say(text)
    def setLanguage(self, language):
        self.textToSpeechModule.setLanguage(language)#P125
    def sayToFile(self, stringToSay, fileName):
        self.textToSpeechModule.sayToFile(stringToSay, fileName)#P125
    def setVoice(self, voiceID):
        self.textToSpeechModule.setVoice(voiceID)#P125
    #以下为自定义接口
    def takeRol(self):
        "拿杆动作"
        pass
    def liftingRol(self):
        "举杆动作"
        pass
    def batting(self, length):
        "击球动作(开环控制)，将球打length长度，单纯的击打动作"
        pass

#感知系统类
class PerceptualSystem(BaseSystem):
    "需要接收类器官协同感知的放这"
    def __init__(self, mark):
        super(PerceptualSystem, self).__init__(mark)
        self.motionModule = robotModule(mark)
        mark.module = "ALSpeechRecognition"
        self.speechRecognitionModule = robotModule(mark)
        mark.module = "ALMemory"
        self.textToSpeechModule = robotModule(mark)
        mark.module = "ALFaceDetection"
        self.faceDetectionModule = robotModule(mark)
        #眼类
        mark.device = body[head][eye0]
        self.eye0 = Eye(mark)
        mark.device = body[head][eye1]
        self.eye1 = Eye(mark)
        #耳类
        self.ear = Ear(mark)
        #皮肤类#待补充
    #以下封装了部分厂商的接口，可随时补充
    def getPosture(self):
        return self.postureModule.getPosture()#阻塞调用，P66
    def robotIsWakeUp(self):
        return self.motionModule.robotIsWakeUp()#P68
    def moveIsActive(self):
        return self.motionModule.moveIsActive()#阻塞调用，P84
    def getMoveConfig(self, config):
        return self.motionModule.getMoveConfig(config)#阻塞调用，P84
    def getRobotPosition(self, useSensors):
        return self.motionModule.getRobotPosition(useSensors)#阻塞调用，P84
    def getRobotVelocity(self):
        return self.motionModule.getRobotVelocity()#阻塞调用，P84
    def getMoveArmsEnabled(self, chainName):
        return self.motionModule.getMoveArmsEnabled()#阻塞调用，P84
    def setVocabulary(self, vocabulary, enableWordSpotting):
        self.speechRecognitionModule.setVocabulary(vocabulary, enableWordSpotting)#P121
    def setLanguage(self, language):
        self.speechRecognitionModule.setLanguage(language)#P121
    def runSpeechRecognition(self):
        self.speechRecognitionModule.subscribe(self.mark.ip)#P121
    def stopSpeechRecognition(self):
        self.speechRecognitionModule.unsubscribe(self.mark.ip)#P121
    def getVocabularyOfRecognition(self):
        return self.memoryModule.getData("WordRecognized")#P121
    def getAvailableLanguages(self):
        return self.textToSpeechModule.getAvailableLanguages()#P125
    def getVolume(self):
        return self.textToSpeechModule.getVolume()#P125
    def getAvailableVoices(self):
        return self.textToSpeechModule.getAvailableVoices()#P125
    def clearDatabaseOfFaceDetection(self):
        self.faceDetectionModule.clearDatabase()#P162
    def forgetPorson(self, name):
        self.faceDetectionModule.forgetPersion(name)#P162
    def getLearnedFacesList(self):
        return self.faceDetectionModule.getLearnedFacesList()#P162
    def getRecognitionConfidenceThreshould(self):
        return self.faceDetectionModule.getRecognitionConfidenceThreshould()#P162
    def setRecognitionConfidenceThreshould(self, threshould):
        self.faceDetectionModule.setRecognitionConfidenceThreshould(threshould)#P162
    def isRecognitionEnable(self):
        return self.faceDetectionModule.isRecognition()#P162
    def setRecognitionEnable(self, enable):
        self.faceDetectionModule.setRecognitionEnable(enable)#P162
    def isTrackingEnable(self):
        return self.faceDetectionModule.isTrackingEnable()#P162
    def setTrackingEnable(self, enable):
        self.faceDetectionModule.setTrackingEnable(enable)#P162
    def learnFace(self, name):
        self.faceDetectionModule.learnFace(name)#P162
    def reLearnFace(self, name):
        self.faceDetectionModule.reLearnFace(name)#P162
    #以下为自定义接口
    def straightLineDetection(self, colorSet1, colorSet2):
        "直线检测，检测被两种颜色集的分界直线，如绿色集和白色集检测的球场边线"
        pass
    def circleDetection(self, colorSet1, colorSet2):
        "圆检测"
        pass
    def rectangleDetection(self, colorSet1, colorSet2):
        "矩形检测"
        pass
    def landMarkDetection(self):
        "地标检测"
        pass







#萝卜类
class Robot():
    "需要运动系统和感知系统协同完成的放这"
    def __init__(self, mark):
        self.exerciseSystem = ExerciseSystem(mark)#运动系统
        self.perceptualSystem = PerceptualSystem(mark)#感知系统
    #以下封装了部分厂商的接口，可随时补充
    def restart(self):
        pass
    #以下为自定义接口
    def moveTo(self, position):
        "移到position坐标"
        pass
    def rotateTo(self, vector):
        "面向vector方向"
        pass
    def getPosition(self):
        "获得萝卜的绝对坐标"
        pass
    def getVector(self):
        "获得萝卜面向的方向"
        pass
    def getBallPosition(self):
        "获得球的绝对坐标"
        pass
    def getHolePosition(self):
        "获得洞的绝对坐标"
        pass
    def getObstacleRange(self):
        "获得障碍范围，既物体的坐标在此范围内非法"
        pass
    def getCourtRange(self):
        "获得球场范围，既萝卜或球的坐标只在此范围内合法"
        pass
    def batting(self, position, vector):
        "击球动作(闭环控制)，position为球的绝对坐标，vector指向击打后位置的绝对坐标"
        pass
    def takeTheFirstRound(self):
        "第一关代码放这"
        pass
    def takeTheSecondRound(self):
        "第二关代码放这"
        pass
    def takeTheThirdRound(self):
        "第三关代码放这"
        pass







#===================================================================
#代码版本    编辑后长度(行)    编辑者                             日期      
#1                  700     湖南工学院.机械学院.17级.杜孟锦       19年7月
#
#
#===================================================================
#测试代码
if __name__ == "__main__":
    robot = Robot(Mark("192.168.1.104", 9559))
    #robot.exerciseSystem.say("test")
    #robot.perceptualSystem.eye0.getImageToPIL().show()
    #robot.perceptualSystem.eye1.getImageTo().show()
    
    cv2.imshow('Eye0', robot.perceptualSystem.eye0.getImageToCV2())
    cv2.imshow('Eye1', robot.perceptualSystem.eye1.getImageToCV2())
    cv2.waitKey(0)
