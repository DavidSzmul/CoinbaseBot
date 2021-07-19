* Install Python 3.8
* Create virutal environment
python -m venv D:\Python\Virtual_Environments\RL-env
Use .bat to activate venv on cmd (add cmd -k to keep window opened)

* Upgrade pip
python -m pip install --upgrade pip

* Install Requirement txt or normal version
python -m pip install -r requirements.txt
OR
python -m pip install tensorforce

* Tensorflow version:
Problem of GPU with version 2.3.1
Try downgrade 2.2.0

* Installation of TF-GPU
Install CUDA + Set Path of CUDA in cmd
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include;%PATH%
SET PATH=C:\cuda\bin;%PATH%  ---> cuDNN

WARNING careful between -> Python version + TF version + CUDA and cuDNN version

* Verification of TF-GPU
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
--> Enable to get if dll are loaded
+
tf.debugging.set_log_device_placement(True)
# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c) 
--> Verify if execution on GPU

* Python hierarchy
https://docs.python-guide.org/writing/structure/
https://blog.finxter.com/python-how-to-import-modules-from-another-folder/

* Add Project path in virtualenv
Create PYTHONPATH in environment variables
Add project in PYTHONPATH (manually for the moment, can create .bat for each project)