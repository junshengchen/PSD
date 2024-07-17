Deploy the trained model to the classification device, which captures images through four industrial cameras. The images are then sent to the model for defect detection. Based on the detected defects, the defects are classified into different levels. The defect levels are then transmitted to the register, which controls the conveyor belt to sort the pears according to their classification.

*  `mvsdk.py  ` contains the SDK camera parameters.
*  `modbus_part.py  `is used for connecting to the register and transmitting classification information.
*  `main.py  `is the main program for classification.
