import sys
import time

import cv2

sys.path.append(r"src/ops/device/hikvision_camera/MvImport")
from src.ops.device.hikvision_camera.MvImport.MvCameraControl_class import *

from loguru import logger
import numpy as np


class CameraWrapper:
    def __init__(self):
        # 获得设备信息
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

        # ch: 枚举设备 | en:Enum device
        # nTLayerType[IN] 枚举传输层 ，pstDevList[OUT] 设备列表
        self.device_list = {}
        self.Enum_device(tlayerType, self.deviceList)

    def _get_image(self, data_buf, nPayloadSize):
        """
        获取图像
        :param data_buf:
        :param nPayloadSize:
        :return: 图像
        """
        # 输出帧的信息
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        # void *memset(void *s, int ch, size_t n);
        # 函数解释:将s中当前位置后面的n个字节 (typedef unsigned int size_t )用 ch 替换并返回 s
        # memset:作用是在一段内存块中填充某个给定的值，它是对较大的结构体或数组进行清零操作的一种最快方法
        # byref(n)返回的相当于C的指针右值&n，本身没有被分配空间
        # 此处相当于将帧信息全部清空了
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        # print(stFrameInfo.fExposureTime)

        # 采用超时机制获取一帧图片，SDK内部等待直到有数据时返回，成功返回0
        ret = self.cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 2000)
        if ret == 0:
            # print(time.time())
            # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
            #     stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            pass
        else:
            logger.info("no data[0x%x]" % ret)

        image = np.asarray(data_buf)  # 将c_ubyte_Array转化成ndarray得到（3686400，）
        image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, -1))  # 根据自己分辨率进行转化

        # image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_Y422)
        # default image format should be RGB8
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def close_device(self):
        """
        关闭设备
        :param cam:
        :param data_buf:
        """
        logger.info("close device")
        # ch:停止取流 | en:Stop grab image
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            logger.info("stop grabbing fail! ret[0x%x]" % ret)
            del self.data_buf
            sys.exit()

        # ch:关闭设备 | Close device
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            logger.info("close device fail! ret[0x%x]" % ret)
            del self.data_buf
            sys.exit()

        # ch:销毁句柄 | Destroy handle
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            logger.info("destroy handle fail! ret[0x%x]" % ret)
            del self.data_buf
            sys.exit()
        logger.info("destroy handle success!")
        del self.data_buf

    def Enum_device(self, tlayerType, deviceList):
        """
        ch:枚举设备 | en:Enum device
        nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
        """
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            logger.info("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            logger.info("find no device!")
            raise RuntimeError("find no device!")

        logger.info("Find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                logger.info("\ngige device: [%d]" % i)
                # 输出设备名字
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                logger.info("device model name: %s" % strModeName)
                # 输出设备用户自定义名
                userDefinedName = ''
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                    userDefinedName = userDefinedName + chr(per)
                logger.info("user defined name: %s" % userDefinedName)
                # 输出设备ID
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                # logger.info("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                device_ip = f'{nip1}.{nip2}.{nip3}.{nip4}'
                logger.info(f"device ip: {device_ip}")
                self.device_list[device_ip] = i
            # 输出USB接口的信息
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                logger.info("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                logger.info("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                logger.info("user serial number: %s" % strSerialNumber)

    def enable_device(self, nConnectionNum):
        """
        设备使能
        :param nConnectionNum: 设备编号
        :return: 相机, 图像缓存区, 图像数据大小
        """
        # ch:创建相机实例 | en:Creat Camera Object
        cam = MvCamera()

        # ch:选择设备并创建句柄 | en:Select device and create handle
        # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
        stDeviceList = cast(self.deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            logger.info("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            logger.info("open device fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    logger.info("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                logger.info("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            logger.info("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:设置图片为RGB8 | en:Set image format as RGB8
        # ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Planar)
        ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
        # ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)
        if ret != 0:
            logger.info("set PixelFormat fail! ret[0x%x]" % ret)
            sys.exit()

        # 从这开始，获取图片数据
        # ch:获取数据包大小 | en:Get payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        # MV_CC_GetIntValue，获取Integer属性值，handle [IN] 设备句柄
        # strKey [IN] 属性键值，如获取宽度信息则为"Width"
        # pIntValue [IN][OUT] 返回给调用者有关相机属性结构体指针
        # 得到图片尺寸，这一句很关键
        # payloadsize，为流通道上的每个图像传输的最大字节数，相机的PayloadSize的典型值是(宽x高x像素大小)，此时图像没有附加任何额外信息
        ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            logger.info("get payload size fail! ret[0x%x]" % ret)
            sys.exit()

        nPayloadSize = stParam.nCurValue

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            logger.info("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()
        #  返回获取图像缓存区。
        data_buf = (c_ubyte * nPayloadSize)()
        #  date_buf前面的转化不用，不然报错，因为转了是浮点型

        self.cam, self.data_buf, self.nPayloadSize = cam, data_buf, nPayloadSize

    def enable_device_by_ip(self, ip):
        assert ip in self.device_list, f"ip {ip} not in device list"
        self.enable_device(self.device_list[ip])

    def get_image(self):
        return self._get_image(self.data_buf, self.nPayloadSize)

    def set_exposure_time(self, exposure_time):
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        if ret != 0:
            logger.info("set ExposureTime fail! ret[0x%x]" % ret)
            sys.exit()

    def set_width(self, width):
        ret = self.cam.MV_CC_SetIntValue("Width", width)
        if ret != 0:
            logger.info("set Width fail! ret[0x%x]" % ret)
            sys.exit()

    def set_height(self, height):
        ret = self.cam.MV_CC_SetIntValue("Height", height)
        if ret != 0:
            logger.info("set Height fail! ret[0x%x]" % ret)
            sys.exit()

    def set_offset_x(self, offset_x):
        ret = self.cam.MV_CC_SetIntValue("OffsetX", offset_x)
        if ret != 0:
            logger.info("set OffsetX fail! ret[0x%x]" % ret)
            sys.exit()

    def set_offset_y(self, offset_y):
        ret = self.cam.MV_CC_SetIntValue("OffsetY", offset_y)
        if ret != 0:
            logger.info("set OffsetY fail! ret[0x%x]" % ret)
            sys.exit()


class FakeCameWrapper:
    def __init__(self):
        pass

    def enable_device(self, nConnectionNum):
        pass

    def enable_device_by_ip(self, ip):
        pass

    def get_image(self, fp='data/dsdata/2024-03-26/ds_1/20000'):
        '''
        从fp中读取图片
        yield

        '''
        for file in os.listdir(fp):
            if not file.endswith('.jpg'):
                continue
            image = cv2.imread(os.path.join(fp, file))
            yield image

    def set_exposure_time(self, exposure_time):
        pass

    def close_device(self):
        pass


if __name__ == '__main__':
    cam = CameraWrapper()
    # cam.enable_device(0)
    cam.enable_device_by_ip('192.168.0.66')
    time.sleep(0.5)
    for i in range(1):
        image = cam.get_image()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        logger.info(f"image.shape{image.shape}")

        # 展示图片
        cv2.imshow('image', image)
        cv2.waitKey(0)
        # 存图
        # cv2.imwrite(f"image{i}.jpg", image)
    cam.close_device()
