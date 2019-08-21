# coding =utf-8

class WalkConfiguration(object):
    def __init__(self):
        super(WalkConfiguration, self).__init__()
        self.MaxStepX = 0.013
        self.MaxStepY = 0.11
        self.MaxStepTheta = 0.075
        self.MaxStepFrequency = 0.98
        self.StepHeight = 0.011
        self.TorsoWx = 0.00
        self.TorsoWy = 0.00

    def WalkLineBig_blue(self):
        walkLineMaxStepX_Big = 0.025
        walkLineMaxStepFrequency_Big = 0.9
        walkLineStepHeight_Big = 0.012

        WalkLineBig_blue_parameters = [["MaxStepX", walkLineMaxStepX_Big],
                                       ["MaxStepY", self.MaxStepY],
                                       ["MaxStepTheta", self.MaxStepTheta],
                                       ["MaxStepFrequency", walkLineMaxStepFrequency_Big],
                                       ["StepHeight", walkLineStepHeight_Big],
                                       ["TorsoWx", self.TorsoWx],
                                       ["TorsoWy", self.TorsoWy]
                                       ]
        return WalkLineBig_blue_parameters

    def WalkLineMiddle_blue(self):
        walkLineMaxStepX_Little = 0.019
        walkLineMaxStepFrequency_Little = 0.9
        walkLineStepHeight_Little = 0.012

        WalkLineMiddle_blue_parameters = [["MaxStepX", walkLineMaxStepX_Little],
                                          ["MaxStepY", self.MaxStepY],
                                          ["MaxStepTheta", self.MaxStepTheta],
                                          ["MaxStepFrequency", walkLineMaxStepFrequency_Little],
                                          ["StepHeight", walkLineStepHeight_Little],
                                          ["TorsoWx", self.TorsoWx],
                                          ["TorsoWy", self.TorsoWy]
                                          ]
        return WalkLineMiddle_blue_parameters

    def WalkLineLittle_blue(self):
        walkLineMaxStepX_Little = 0.013
        walkLineMaxStepFrequency_Little = 0.9
        walkLineStepHeight_Little = 0.012

        WalkLineLittle_blue_parameters = [["MaxStepX", walkLineMaxStepX_Little],
                                          ["MaxStepY", self.MaxStepY],
                                          ["MaxStepTheta", self.MaxStepTheta],
                                          ["MaxStepFrequency", walkLineMaxStepFrequency_Little],
                                          ["StepHeight", walkLineStepHeight_Little],
                                          ["TorsoWx", self.TorsoWx],
                                          ["TorsoWy", self.TorsoWy]
                                          ]
        return WalkLineLittle_blue_parameters

    def WalkSideBig_blue(self):
        walkSideMaxStepY_Big = 0.12
        walkSideMaxStepFrequency_Big = 0.9
        walkSideStepHeight_Big = 0.01

        WalkSideBig_blue_parameters = [["MaxStepX", self.MaxStepX],
                                       ["MaxStepY", walkSideMaxStepY_Big],
                                       ["MaxStepTheta", self.MaxStepTheta],
                                       ["MaxStepFrequency", walkSideMaxStepFrequency_Big],
                                       ["StepHeight", walkSideStepHeight_Big],
                                       ["TorsoWx", self.TorsoWx],
                                       ["TorsoWy", self.TorsoWy]
                                       ]

        return WalkSideBig_blue_parameters

    def WalkSideLittle_blue(self):
        walkSideMaxStepY_Little = 0.11
        walkSideMaxStepFrequency_Little = 0.9
        walkSideStepHeight_Little = 0.01

        WalkSideLittle_blue_parameters = [["MaxStepX", self.MaxStepX],
                                          ["MaxStepY", walkSideMaxStepY_Little],
                                          ["MaxStepTheta", self.MaxStepTheta],
                                          ["MaxStepFrequency", walkSideMaxStepFrequency_Little],
                                          ["StepHeight", walkSideStepHeight_Little],
                                          ["TorsoWx", self.TorsoWx],
                                          ["TorsoWy", self.TorsoWy]
                                          ]
        return WalkSideLittle_blue_parameters

    def WalkCircleBig_blue(self):
        walkCircleMaxStepTheta_Big = 0.1
        walkCircleMaxStepFrequency_Big = 0.9
        walkCirclesStepHeight_Big = 0.01

        WalkCircleBig_blue_parameters = [["MaxStepX", self.MaxStepX],
                                         ["MaxStepY", self.MaxStepY],
                                         ["MaxStepTheta", walkCircleMaxStepTheta_Big],
                                         ["MaxStepFrequency", walkCircleMaxStepFrequency_Big],
                                         ["StepHeight", walkCirclesStepHeight_Big],
                                         ["TorsoWx", self.TorsoWx],
                                         ["TorsoWy", self.TorsoWy]
                                         ]

        return WalkCircleBig_blue_parameters

    def WalkCircleLittle_blue(self):
        walkCircleMaxStepTheta_Little = 0.07
        walkCircleMaxStepFrequency_Little = 1.0
        walkCircleStepHeight_Little = 0.01

        WalkCircleLittle_blue_parameters = [["MaxStepX", self.MaxStepX],
                                            ["MaxStepY", self.MaxStepY],
                                            ["MaxStepTheta", walkCircleMaxStepTheta_Little],
                                            ["MaxStepFrequency", walkCircleMaxStepFrequency_Little],
                                            ["StepHeight", walkCircleStepHeight_Little],
                                            ["TorsoWx", self.TorsoWx],
                                            ["TorsoWy", self.TorsoWy]
                                            ]
        return WalkCircleLittle_blue_parameters

    def WalkTriangleBig_blue(self):
        walkTriangleMaxStepX_Big = 0.01
        walkTriangleMaxStepY_Big = 0.15
        walkTriangleMaxStepTheta_Big = 0.18
        walkTriangleMaxStepFrequency_Big = 0.98
        walkTriangleStepHeight_Big = 0.007

        WalkTriangleBig_blue_parameters = [["MaxStepX", walkTriangleMaxStepX_Big],
                                           ["MaxStepY", walkTriangleMaxStepY_Big],
                                           ["MaxStepTheta", walkTriangleMaxStepTheta_Big],
                                           ["MaxStepFrequency", walkTriangleMaxStepFrequency_Big],
                                           ["StepHeight", walkTriangleStepHeight_Big],
                                           ["TorsoWx", self.TorsoWx],
                                           ["TorsoWy", self.TorsoWy]
                                           ]
        return WalkTriangleBig_blue_parameters

    def WalkTriangleLittle_blue(self):
        walkTriangleMaxStepX_Little = 0.01
        walkTriangleMaxStepY_Little = 0.14
        walkTriangleMaxStepTheta_Little = 0.15
        walkTriangleMaxStepFrequency_Little = 0.98
        walkTriangleStepHeight_Little = 0.007

        WalkTriangleLittle_blue_parameters = [["MaxStepX", walkTriangleMaxStepX_Little],
                                              ["MaxStepY", walkTriangleMaxStepY_Little],
                                              ["MaxStepTheta", walkTriangleMaxStepTheta_Little],
                                              ["MaxStepFrequency", walkTriangleMaxStepFrequency_Little],
                                              ["StepHeight", walkTriangleStepHeight_Little],
                                              ["TorsoWx", self.TorsoWx],
                                              ["TorsoWy", self.TorsoWy]
                                              ]
        return WalkTriangleLittle_blue_parameters

    def WalkLineBig_red(self):
        walkLineMaxStepX_Big = 0.025
        walkLineMaxStepFrequency_Big = 0.98
        walkLineStepHeight_Big = 0.15

        WalkLineBig_red_parameters = [["MaxStepX", walkLineMaxStepX_Big],
                                      ["MaxStepY", self.MaxStepY],
                                      ["MaxStepTheta", self.MaxStepTheta],
                                      ["MaxStepFrequency", walkLineMaxStepFrequency_Big],
                                      ["StepHeight", walkLineStepHeight_Big],
                                      ["TorsoWx", self.TorsoWx],
                                      ["TorsoWy", self.TorsoWy]
                                      ]
        return WalkLineBig_red_parameters

    def WalkLineLittle_red(self):
        walkLineMaxStepX_Little = 0.013
        walkLineMaxStepFrequency_Little = 0.98
        walkLineStepHeight_Little = 0.015

        WalkLineLittle_red_parameter = [["MaxStepX", walkLineMaxStepX_Little],
                                        ["MaxStepY", self.MaxStepY],
                                        ["MaxStepTheta", self.MaxStepTheta],
                                        ["MaxStepFrequency", walkLineMaxStepFrequency_Little],
                                        ["StepHeight", walkLineStepHeight_Little],
                                        ["TorsoWx", self.TorsoWx],
                                        ["TorsoWy", self.TorsoWy]
                                        ]
        return WalkLineLittle_red_parameter

    def WalkSideBig_red(self):
        walkSideMaxStepY_Big = 0.12
        walkSideMaxStepFrequency_Big = 0.9
        walkSideStepHeight_Big = 0.01

        WalkSideBig_red_parameters = [["MaxStepX", self.MaxStepX],
                                      ["MaxStepY", walkSideMaxStepY_Big],
                                      ["MaxStepTheta", self.MaxStepTheta],
                                      ["MaxStepFrequency", walkSideMaxStepFrequency_Big],
                                      ["StepHeight", walkSideStepHeight_Big],
                                      ["TorsoWx", self.TorsoWx],
                                      ["TorsoWy", self.TorsoWy]
                                      ]

        return WalkSideBig_red_parameters

    def WalkSideLittle_red(self):
        walkSideMaxStepY_Little = 0.11
        walkSideMaxStepFrequency_Little = 0.9
        walkSideStepHeight_Little = 0.01

        WalkSideLittle_red_parameters = [["MaxStepX", self.MaxStepX],
                                         ["MaxStepY", walkSideMaxStepY_Little],
                                         ["MaxStepTheta", self.MaxStepTheta],
                                         ["MaxStepFrequency", walkSideMaxStepFrequency_Little],
                                         ["StepHeight", walkSideStepHeight_Little],
                                         ["TorsoWx", self.TorsoWx],
                                         ["TorsoWy", self.TorsoWy]
                                         ]

        return WalkSideLittle_red_parameters

    def WalkCircleBig_red(self):
        walkCircleMaxStepTheta_Big = 0.1
        walkCircleMaxStepFrequency_Big = 0.98
        walkCircleStepHeight_Big = 0.015

        WalkCircleBig_red_parameters = [["MaxStepX", self.MaxStepX],
                                        ["MaxStepY", self.MaxStepY],
                                        ["MaxStepTheta", walkCircleMaxStepTheta_Big],
                                        ["MaxStepFrequency", walkCircleMaxStepFrequency_Big],
                                        ["StepHeight", walkCircleStepHeight_Big],
                                        ["TorsoWx", self.TorsoWx],
                                        ["TorsoWy", self.TorsoWy]
                                        ]
        return WalkCircleBig_red_parameters

    def WalkCircleLittle_red(self):
        walkCircleMaxStepTheta_Little = 0.07
        walkCircleMaxStepFrequency_Little = 0.98
        walkCirclesStepHeight_Little = 0.015

        WalkCircleLittle_red_parameters = [["MaxStepX", self.MaxStepX],
                                           ["MaxStepY", self.MaxStepY],
                                           ["MaxStepTheta", walkCircleMaxStepTheta_Little],
                                           ["MaxStepFrequency", walkCircleMaxStepFrequency_Little],
                                           ["StepHeight", walkCirclesStepHeight_Little],
                                           ["TorsoWx", self.TorsoWx],
                                           ["TorsoWy", self.TorsoWy]
                                           ]

        return WalkCircleLittle_red_parameters

    def WalkTriangleBig_red(self):
        walkTriangleMaxStepX_Big = 0.01
        walkTriangleMaxStepY_Big = 0.15
        walkTriangleMaxStepTheta_Big = 0.18
        walkTriangleMaxStepFrequency_Big = 0.98
        walkTriangleStepHeight_Big = 0.01

        WalkTriangleBig_red_parameters = [["MaxStepX", walkTriangleMaxStepX_Big],
                                          ["MaxStepY", walkTriangleMaxStepY_Big],
                                          ["MaxStepTheta", walkTriangleMaxStepTheta_Big],
                                          ["MaxStepFrequency", walkTriangleMaxStepFrequency_Big],
                                          ["StepHeight", walkTriangleStepHeight_Big],
                                          ["TorsoWx", self.TorsoWx],
                                          ["TorsoWy", self.TorsoWy]
                                          ]

        return WalkTriangleBig_red_parameters

    def WalkTriangleLittle_red(self):
        walkTriangleMaxStepX_Little = 0.01
        walkTriangleMaxStepY_Little = 0.14
        walkTriangleMaxStepTheta_Little = 0.15
        walkTriangleMaxFrequency_Little = 0.98
        walkTriangleStepHeight_Little = 0.007

        WalkTriangleLittle_red_parameters = [["MaxStepX", walkTriangleMaxStepX_Little],
                                             ["MaxStepY", walkTriangleMaxStepY_Little],
                                             ["MaxStepTheta", walkTriangleMaxStepTheta_Little],
                                             ["MaxStepFrequency", walkTriangleMaxFrequency_Little],
                                             ["StepHeight", walkTriangleStepHeight_Little],
                                             ["TorsoWx", self.TorsoWx],
                                             ["TorsoWy", self.TorsoWy]
                                             ]

        return WalkTriangleLittle_red_parameters
