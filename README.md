# Posture Detection & Pose Classification Web Application

This project is a posture detection and pose classification web application developed using Python, Flask, and various libraries such as Mediapipe and Matplotlib. It utilizes machine learning algorithms to accurately detect and classify human poses, including poses like Tree Pose, T Pose, Warrior II, etc. This application is particularly useful for applications like fitness monitoring and gesture-based interfaces.

## Features

- **Pose Detection**: Utilizes Mediapipe library to detect human poses in real-time through the webcam.
- **Pose Classification**: Employs machine learning algorithms to classify detected poses into predefined categories such as Tree Pose, T Pose, Warrior II, etc.
- **Web Interface**: Provides a user-friendly web interface for interacting with the application.
- **Fitness Monitoring**: Enables users to monitor their posture during fitness activities such as yoga, pilates, etc.
- **Gesture-Based Interfaces**: Can be integrated into gesture-based interfaces for controlling applications or devices.

## Screenshots

![Screenshot 1](Screenshots/home-page.png)

![Screenshot 2](Screenshots/detection.png)

## Requirements

- Flask==3.0.2
- Mediapipe==0.10.5
- opencv_contrib_python==4.8.1.78
- opencv_python==4.8.1.78

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/username/posture-detection-web-app.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Posture-Detection-Pose-Classification-Project
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:

   ```bash
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Allow access to the webcam if prompted.

4. Follow the instructions on the web interface to detect and classify human poses.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
