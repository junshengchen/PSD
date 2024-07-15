# 开发时间：2024/6/19 23:40
import logging
import time
import modbus_tk
import serial
from modbus_tk import modbus_rtu
import modbus_tk.defines as cst
import numpy as np



class Modbus():
    def __init__(self):
        self.modbusSerialPort = r'COM4';  # 串口信息
        self.modbusSerialBaudrate = 115200;
        self.modbusSite = 1;
        while True:
            try:
                master = modbus_rtu.RtuMaster(
                    serial.Serial(port=self.modbusSerialPort, baudrate=self.modbusSerialBaudrate, bytesize=8, parity='N',
                                  stopbits=1,
                                  xonxoff=0)
                )
                master.set_timeout(0.1)  # 0.1s timeout # 这个数据建议根据需要调整
                master.set_verbose(True)  # 正式使用的时候请注释掉
                break
            except modbus_tk.modbus.ModbusError as exc:
                logging.error(exc)
                del master
                time.sleep(3);  # 等待3秒
                continue;
        self.master = master
    def read_register(self):
        while True:
            modbusErrorCount = 0
            try:
                # 读取plc程序，寄存器的值
                res = self.master.execute(self.modbusSite, cst.READ_HOLDING_REGISTERS, 4096, 5);
            # print("res:----",res)
            except Exception:
                modbusErrorCount = modbusErrorCount + 1;
                if modbusErrorCount > 100:
                    logging.error('本次寄存器读取失败，重新初始化一个master')
                    del self.master
                    time.sleep(3);  # 等待3秒
                    self.__init__() #重新初始化寄存器
            # 读取寄存器的值，返回最新当前位置的果盘编号
            if len(res) == 5:
                return res[0];
    def write_register(self,currnet_litchi_grade,current_bowl_number):#往寄存器写等级
        writeDelay = 0
        while writeDelay <= 3:
            try:
                rand_int = np.random.randint(1, 10)
                self.master.execute(self.modbusSite, cst.WRITE_MULTIPLE_REGISTERS, 4106,
                               output_value=[current_bowl_number, currnet_litchi_grade])
                print("current_number:", current_bowl_number, "   grade:", currnet_litchi_grade)
                return
            except:
                writeDelay = writeDelay + 1;
                continue;
        print("本次写入寄存器失败！")