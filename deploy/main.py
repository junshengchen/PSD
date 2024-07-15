# coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import threading
from ultralytics import YOLO
import logging
import time
import modbus_tk
import serial
from modbus_tk import modbus_rtu
import modbus_tk.defines as cst

class Modbus():
    def __init__(self):
        self.modbusSerialPort = r'COM5'  # 串口信息
        self.modbusSerialBaudrate = 115200
        self.modbusSite = 1
        while True:
            try:
                master = modbus_rtu.RtuMaster(
                    serial.Serial(port=self.modbusSerialPort, baudrate=self.modbusSerialBaudrate, bytesize=8,
                                  parity='N',
                                  stopbits=1, xonxoff=0)
                )
                master.set_timeout(0.1)  # 0.1s timeout # 这个数据建议根据需要调整
                master.set_verbose(True)  # 正式使用的时候请注释掉
                break
            except modbus_tk.modbus.ModbusError as exc:
                logging.error(exc)
                del master
                time.sleep(3)  # 等待3秒
                continue
        self.master = master

    def read_register(self):
        modbusErrorCount = 0
        while True:
            try:
                # 读取plc程序，寄存器的值
                res = self.master.execute(self.modbusSite, cst.READ_HOLDING_REGISTERS, 4096, 5)
                # print("res:----",res)
                if len(res) == 5:
                    return res[0]
            except Exception:
                modbusErrorCount += 1
                if modbusErrorCount > 100:
                    logging.error('本次寄存器读取失败，重新初始化一个master')
                    del self.master
                    time.sleep(3)  # 等待3秒
                    self.__init__()  # 重新初始化寄存器

    def write_register(self, currnet_litchi_grade, current_bowl_number):  # 往寄存器写等级
        writeDelay = 0
        while writeDelay <= 3:
            try:
                self.master.execute(self.modbusSite, cst.WRITE_MULTIPLE_REGISTERS, 4106,
                                    output_value=[current_bowl_number, currnet_litchi_grade])
                print("current_number:", current_bowl_number, "   grade:", currnet_litchi_grade)
                return
            except Exception:
                writeDelay += 1
                continue
        print("本次写入寄存器失败！")

def grade_detection(classes):
    if classes.size == 0:
        return 2  # 等级 2，当没有检测到类别时
    if 0 in classes:
        return 1  # 等级 1
    elif 2 in classes:
        return 3  # 等级 3
    elif 1 in classes:
        return 4  # 等级 4

def process_camera(hCamera, pFrameBuffer, cap, model, window_name, target_classes, modbus):
    # 获取相机特性描述
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 从相机取一帧图片
        try:
            start_time = time.time()  # 开始计时
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            capture_time = time.time()  # 图像捕获完成

            # 使用YOLO模型进行推理
            results = model(frame)
            inference_time = time.time()  # 推理完成

            # 获取推理结果
            predictions = results[0]
            boxes = predictions.boxes.xyxy.cpu().numpy()  # 获取检测框坐标
            scores = predictions.boxes.conf.cpu().numpy()  # 获取置信度
            classes = predictions.boxes.cls.cpu().numpy().astype(int)  # 获取类别

            # 打印检测到的所有类别
            detected_classes = set(classes)
            print(f"Detected classes for {window_name}:", detected_classes)

            # 定义类别标签
            labels = {0: 'pear', 1: 'bruise', 2: 'twig', 3: 'rot'}

            # 绘制检测结果
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{labels.get(cls, 'unknown')} {score:.2f}"
                # 绘制边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制标签
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示结果图像
            cv2.imshow(window_name, frame)

            # 对检测结果进行分级
            grade = grade_detection(classes)
            grading_time = time.time()  # 分级完成

            # 读取当前位置果盘编号
            current_bowl_number = modbus.read_register()

            # 将分级结果和当前果盘编号写入寄存器
            if current_bowl_number is not None:
                modbus.write_register(currnet_litchi_grade=grade, current_bowl_number=current_bowl_number)
            modbus_time = time.time()  # 写入Modbus完成

            # 打印各个步骤的时间
            print(f"Capture time: {capture_time - start_time:.4f}s")
            print(f"Inference time: {inference_time - capture_time:.4f}s")
            print(f"Grading time: {grading_time - inference_time:.4f}s")
            print(f"Modbus time: {modbus_time - grading_time:.4f}s")
            print(f"Total time: {modbus_time - start_time:.4f}s")

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"CameraGetImageBuffer failed for {window_name}({e.error_code}): {e.message}")

    # 保持显示最终检测结果
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭相机
    mvsdk.CameraUnInit(hCamera)

    # 释放帧缓存
    mvsdk.CameraAlignFree(pFrameBuffer)

def main():
    modbus = Modbus()  # 初始化Modbus实例

    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    cameras = []
    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    camera_indices = input("Select cameras (space separated indices): ")
    camera_indices = [int(idx) for idx in camera_indices.split()]

    # 加载YOLO模型
    model = YOLO('best.pt')  # 请替换为您的模型路径

    # 设定目标类别，类别标签可以根据您的需求修改
    target_classes = {0, 1}  # 示例：0表示'pear', 1表示'bruise'

    for idx in camera_indices:
        DevInfo = DevList[idx]
        print(DevInfo)

        # 打开相机
        hCamera = 0
        try:
            hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print(f"CameraInit Failed({e.error_code}): {e.message}")
            continue

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(hCamera, 2)

        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

        # 启动线程处理每个相机的图像
        window_name = f"YOLO Detection - Camera {idx}"
        thread = threading.Thread(target=process_camera, args=(hCamera, pFrameBuffer, cap, model, window_name, target_classes, modbus))
        thread.start()
        cameras.append((thread, hCamera, pFrameBuffer))

    try:
        for thread, hCamera, pFrameBuffer in cameras:
            thread.join()
    finally:
        for _, hCamera, pFrameBuffer in cameras:
            # 关闭相机
            mvsdk.CameraUnInit(hCamera)
            # 释放帧缓存
            mvsdk.CameraAlignFree(pFrameBuffer)
        cv2.destroyAllWindows()

main()
