---
# **Unitree Go1 Edu Sidewalk Navigation System**

This project explores the development of a robust sidewalk navigation system for **telepresence robots**, specifically tailored to the **Unitree Go1 Edu Robot Dog**. By leveraging advanced computer vision techniques and custom model development, the system addresses real-world challenges such as sidewalk detection, obstacle avoidance, and seamless navigation in urban environments.

---

## **Project Overview**

The main goal of this project was to create a **reliable sidewalk navigation system** for telepresence robots, focusing on enhancing the mobility of these robots in urban environments. The research primarily addressed challenges related to accurate sidewalk segmentation, real-time processing, and user interaction.

### **Research Focus**:
- **Primary Objective**: To design and implement a **sidewalk segmentation model** that enables smooth, autonomous navigation for the Unitree Go1 Edu Robot.
- **Key Challenge**: Existing sidewalk navigation systems often lack robustness, limiting their usability in complex environments.
- **Solution**: A custom instance segmentation model was developed to accurately detect sidewalks, integrated with external hardware for real-time navigation.

---

## **Methodology**

The development process involved several key steps:

### 1. **Custom Model Development**:
   - A bespoke instance segmentation model was built specifically for **sidewalk detection** using the Unitree Go1 Edu robot dog.
   - Popular models like **YOLOv8**, **SAM**, and **FastSAM** were tested, with YOLOv8 selected for its real-time capabilities.

### 2. **Computer Vision Techniques**:
   - Advanced computer vision techniques were applied to achieve precise segmentation of sidewalks in urban environments.
   - The model successfully distinguished between sidewalks and background elements.

### 3. **External Camera Integration**:
   - An external **IP Webcam** was integrated to overcome the onboard camera limitations of the Unitree robot, enabling more accurate image capture and processing.

### 4. **Dataset Creation**:
   - A **diverse dataset** of Australian sidewalks was curated, featuring a range of textures, colors, and patterns to improve the model's generalization capabilities.
   - The dataset helped the model perform well across various sidewalk conditions.

### 5. **Model Testing**:
   - The system was tested for performance and accuracy using popular models, with **YOLOv8** delivering an average inference time of **10ms** on a GPU-enabled system.
   - Metrics like the **confusion matrix** were used to assess the model's precision and recall.

---

## **Key Results**

- **Segmentation Accuracy**: The trained model exhibited high accuracy in detecting sidewalks, efficiently distinguishing them from background elements.
- **Inference Time**: With a GPU-enabled system, the YOLOv8 model demonstrated an average inference time of **10ms**, ensuring real-time navigation.
- **Robustness**: The system was tested in various urban environments, successfully navigating sidewalks while avoiding obstacles.
- **Obstacle Avoidance**: The robot demonstrated smooth navigation, even in environments with varying sidewalk textures and potential obstructions.

---

## **Conclusion**

This research project successfully addressed the challenge of developing a sophisticated **sidewalk navigation system** for telepresence robots. By implementing a custom instance segmentation model, the project demonstrated how **computer vision techniques** can be applied to enable robust and efficient navigation in complex urban environments.

The insights and results from this project provide a foundation for future research and practical applications in the field of **autonomous systems** and **robotics**. The work opens up new avenues for further exploration in real-world applications of telepresence robots.

---

## **Project Resources**

- **Annotated Australian Sidewalk Dataset**: [Link](https://universe.roboflow.com/projectnlr2u/sidewalk-segmentationv4gpn/dataset/)
- **General Sidewalk Dataset**: [Link](https://universe.roboflow.com/projects5k1o6/sidewalk-dlu6l/dataset/1)
- **Robot Navigation Demo**: [Watch Demo](https://youtu.be/w4D5bM6zMF8)
- **Robot Navigation Demo with Model Inference**: [Watch Inference Demo](https://youtu.be/5iUNyrXyqjY)
- **Source Code**: [GitHub Repository](https://github.com/cyber-panther/Unitree-go1-edu-sidewalk-navigation)

---

## **Acknowledgements**

The development of this system was supported by various resources, including:

- **Halbe, S. (2020)** - Overview of Object Detection and Instance Segmentation.
- **Rath, S. (2024)** - Comprehensive guide on YOLOv5 for instance segmentation.
- **Unitree Go1 Documentation** - Reference for robot integration.
- **Roboflow** - Dataset management platform for model training.
- **IP Webcam** - App for external camera integration.

For detailed information about the project, feel free to explore the provided links and resources.

---
