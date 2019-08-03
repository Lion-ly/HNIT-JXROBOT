import math
import time


class ComputeHitBallPosition(object):

    def __init__(self, ip, port, landmarkTheoreticalSize=0.105, HitBallD=0.10):
        self.ip = ip
        self.port = port
        self.landmarkTheoreticalSize = landmarkTheoreticalSize
        self.HitBall = HitBallD

    def computeMarkDistance(self):
        """
        :return: HoleX, HoleY, MarkDistance
        """
        currentCamera = "CameraTop"
        memoryProxy = ALProxy("ALMemory", self.ip, self.port)
        landmarkProxy = ALProxy("ALLandMarkDetection", self.ip, self.port)
        landmarkProxy.subscribe("landmarkTest")
        markData = memoryProxy.getData("LandmarkDetected")
        while markData is None or len(markData) == 0:
            time.sleep(0.1)
            markData = memoryProxy.getData("LandmarkDetected")
        wzCamera = markData[1][0][0][1]
        wyCamera = markData[1][0][0][2]
        angularSize = markData[1][0][0][3]
        distanceFromCameraToLandmark = landmarkTheoreticalSize / (2 * math.tan(angularSize / 2))
        motionProxy = ALProxy("ALMotion", self.ip, self.port)
        transform = motionProxy.getTransform(currentCamera, 2, True)
        transformList = almath.vectorFloat(transform)
        robotToCamera = almath.Transform(transformList)
        cameraToLandmarkRotationTransform = almath.Transform_from3DRotation(0, wyCamera, wzCamera)
        cameraToLandmarkTranslationTransform = almath.Transform(distanceFromCameraToLandmark, 0, 0)
        robotToLandmark = robotToCamera * cameraToLandmarkRotationTransform * cameraToLandmarkTranslationTransform
        x = robotToLandmark
        y = robotToLandmark
        # print "x " + str(robotToLandmark.r1_c4) + " (in meters)"
        # print "y " + str(robotToLandmark.r2_c4) + " (in meters)"
        # print "z " + str(robotToLandmark.r3_c4) + " (in meters)"
        distance = math.sqrt(robotToLandmark.r1_c4 ** 2 + robotToLandmark.r2_c4 ** 2)
        landmarkProxy.unsubscribe("landmarkTest")
        return x, y, distance

    def computeHitBallPosition2(self, ballX, ballY):
        """
        :param ballX: ballX
        :param ballY: ballY
        :return: A dict, HoleX, HoleY, HitX, HitY
        """
        hb = self.HitBallD  # The distance from the hitting point to the ball
        HoleX, HoleY, d = self.computeMarkDistance()

        k = (HoleY - ballY) / (HoleX - ballX)

        HitX = math.sqrt(math.sqrt(hb) / (k + 1)) + ballX  # Hit ball position
        HitY = k * (math.sqrt(math.sqrt(hb) / (k + 1))) + ballY

        return {"HoleX": HoleX, "HoleY": HoleY, "HitX": HitX, "HitY": HitY}
