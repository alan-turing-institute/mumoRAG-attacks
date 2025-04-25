from enum import Enum


class AttackMask(Enum):
    Full = (512,512,0,0)
    FirstQuadrant = (256,256,0,0)
    SecondQuadrant = (256, 256, 256, 0)
    ThirdQuadrant = (256, 256, 0, 256)
    FourthQuadrant = (256, 256, 256, 256)