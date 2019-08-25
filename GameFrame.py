# coding=utf-8
import threading
import time
import inspect
import ctypes
from naoqi import ALProxy
import functools
from ConfigureNao import ConfigureNao


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stopThread(thread):
    _async_raise(thread.ident, SystemExit)


class GolfGame(ConfigureNao):
    def __init__(self, IP, PORT=9559):
        super(GolfGame, self).__init__(IP, PORT)
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.__start()

    def __start(self):
       """
        第一次开启任务，只需要触摸机器人头部前中后即可开始。
        之后重启任务需要，先同时触摸机器人右手背部和三个
        任务对应的头部(比如当前正在执行第一个任务，若要重新
        开始，应先同时触摸头的前部和右手背部，关闭当前正在
        执行的任务，之后才可以触摸机器人头部，选择任务开始）。

        """
        self.__task1_start()
        self.__task2_start()
        self.__task3_start()
        while True:
            # 同时触摸FrontHead和RHand,强制结束任务线程,重新开始新线程
            flagData = self.touchData()
            if flagData == [1.0, 0, 0, 1.0]:
                print("task1 is terminate")
                stopThread(self.t1)
                self.__task1_start()

            # 同时触摸MiddleHead和RHand
            if flagData == [0, 1.0, 0, 1.0]:
                print("task2 is terminate")
                stopThread(self.t2)
                self.__task2_start()

            # 同时触摸RearHead和RHand
            if flagData == [0, 0, 1.0, 1.0]:
                print("task3 is terminate")
                stopThread(self.t3)
                self.__task3_start()

            time.sleep(1)

    def __task1_start(self):
        self.t1 = threading.Thread(target=self.__run1)
        self.t1.start()

    def __task2_start(self):
        self.t2 = threading.Thread(target=self.__run2)
        self.t2.start()

    def __task3_start(self):
        self.t3 = threading.Thread(target=self.__run3)
        self.t3.start()

    def __run1(self):
        flagData = self.touchData()
        # 触摸FrontHead开始
        if flagData == [1.0, 0, 0, 0]:
            # 将要执行的第一关任务放在此处
            print("task1 is running\n")
            # end
        time.sleep(1)

    def __run2(self):
        flagData = self.touchData()
        # 触摸MiddleHead开始
        if flagData == [0, 1.0, 0, 0]:
            # 将要执行的第二关任务放在此处
            print("task1 is running\n")
            # end
        time.sleep(1)

    def __run3(self):
        flagData = self.touchData()
        # 触摸RearHead开始
        if flagData == [0, 0, 1.0, 0]:
            # 将要执行的第三关任务放在此处
            print("task1 is running\n")
            # end
        time.sleep(1)

    def touchData(self):
        FrontFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")
        MiddleFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Middle/Sensor/Value")
        RearFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Rear/Sensor/Value")
        RArmFlag = self.memoryProxy.getData("Device/SubDeviceList/RHand/Touch/Back/Sensor/Value")

        return [FrontFlag, MiddleFlag, RearFlag, RArmFlag]


if __name__ == '__main__':
    robotIp = "192.168.137.55"
    port = 9559
    game = GolfGame(robotIp, port)
