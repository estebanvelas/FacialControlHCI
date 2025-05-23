# **FacialControlHCI**  
A facial gesture-based control system for Human-Computer Interaction (HCI) using the Estimated Cervical Rotation point as an Anchor.  

## **Overview**  
**FacialControlHCI** is an open-source project that enables users to interact with computers using facial expressions. It leverages **computer vision** and **machine learning** to recognize facial gestures and map them to system controls.  

## **Features**  
✅ Real-time facial gesture recognition  
✅ Hands-free interaction  
✅ Supports multiple facial expressions  
✅ Built using **OpenCV** and **Python**  
✅ Easily extensible for custom applications  

## **Installation**  

### **Prerequisites**  
Ensure you have the following installed:  
- Python 3.x  
- OpenCV  
- NumPy  

### **Setup**  
Clone the repository and install dependencies:  
```sh
git clone https://github.com/estebanvelas/FacialControlHCI.git
cd FacialControlHCI
pip install -r requirements.txt
```

## **Usage**  
Run the facial control system:  
```sh
python main.py
```
Adjust settings in the configuration file (`config.json`) to fine-tune recognition sensitivity. 

## **Windows Deployment**
Modify FacialControlHMI.spec with your requirements.
Run in CMD:
```sh
pyinstaller FacialControlHMI.spec
```

## **Contributing**  
We welcome contributions! Feel free to submit **issues**, create **pull requests**, or suggest improvements.  

## **License**  
This project is licensed under the **GNU General Public License v3.0**.  
You are free to use, modify, and distribute this software under the terms of the [GPL v3 License](LICENSE).  

## **Acknowledgments**  
Special thanks to **Sara Pyszka**, **Dr. Stephanie Valencia** and all other contributors for developing this project.
