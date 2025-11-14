#  AI-Driven Yoga Posture Correction Using Computer Vision

## Project Overview
This project applies **computer vision** and **AI** to guide yoga practice in real time.  
It uses **MediaPipe** to detect 33 human body landmarks (shoulders, elbows, knees, etc.) from each video frame.  
Joint angles are then computed from these landmarks and fed into a trained **machine learning model** (e.g., a Random Forest classifier) to identify the user’s yoga pose.

The application provides **immediate feedback**:
- It overlays the user’s skeleton on the webcam feed (using **OpenCV**).
- Displays the recognized pose.
- Issues **spoken instructions** via a text-to-speech engine (**pyttsx4**) to correct misalignments.

In summary, the system aims to mimic a **virtual instructor** by continuously analyzing posture and giving **visual/audio cues** for safe, accurate, and consistent yoga form.

---

##  Features

###  Real-Time Pose Detection
Uses **MediaPipe’s pose estimation** to extract 33 key body landmarks from the webcam stream.  
Angles between joints are computed on-the-fly for all major limbs.

###  Pose Classification
A **scikit-learn model** (e.g., Random Forest or SVM) is trained on joint-angle data to recognize yoga asanas.  
The classifier predicts the current pose (e.g., “Warrior II”, “Tree”, etc.) from the detected landmarks.

###  Visual Feedback
The **OpenCV window** shows the live video with a skeleton overlay and pose label.  
If misalignment is detected, visual cues (highlighted joints or text) appear to indicate needed corrections.

###  Voice Guidance
Using the **pyttsx4** library, the system speaks corrective instructions (e.g., “Straighten your left knee”) at regular intervals to guide the user.  
This **multi-modal feedback** (audio + visual) helps users adjust their posture without constantly watching the screen.

###  Performance Evaluation
After training, the system can output evaluation metrics (confusion matrix, accuracy) for the pose classifier.  
A **confusion matrix** is plotted using **Seaborn/Matplotlib** to summarize classification results on test data.

---

##  System Requirements

- **Python:** Version 3.8 or higher (tested on Python 3.9+ recommended).  
  The project relies on standard Python 3 support.
- **Libraries:** Requires Python packages listed in `requirements.txt`, including  
  OpenCV, MediaPipe, NumPy, Pandas, scikit-learn, seaborn, matplotlib, and pyttsx4.  
- **Hardware:** A computer with a webcam for live video capture.  
  A modern CPU is sufficient, but a GPU can speed up MediaPipe’s pose processing.
- **Operating System:** Compatible with Windows, macOS, or Linux (any OS where Python and OpenCV can run).

---

##  Installation

### 1️. Clone the Repository
```bash
git clone https://github.com/yourusername/AI-Yoga-Posture-Correction.git
cd AI-Yoga-Posture-Correction
```
### 2. Create a Virtual Environment

It’s recommended to use Python’s built-in **venv** module to isolate dependencies.

```bash
python3 -m venv venv
```

###  Setup Instructions

### 3. Activate the Virtual Environment

**On Unix/macOS:**
```bash
source venv/bin/activate
```
**On Windows**
```bash
.\venv\Scripts\activate
```

### 4. Install Dependencies

Use `pip` to install all required packages from `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Usage

Once the environment is set up, run the main application:

```bash
python main.py
```

### Usage Details

This script will:

- Load the training data (`train_angle.csv`).
- Train the pose classification model.
- Begin the real-time feedback loop using your webcam.

The **OpenCV window** will pop up, showing your live video feed with the **skeleton overlay** and **pose name**.  
If any joint is outside the ideal range for the identified pose, the system will **highlight it** and **speak a correction**.  
For example: *“Bend your right knee more”* during the Warrior pose.

By default, `main.py` uses **webcam input (camera index 0)**.  
You can modify the code to use a **video file** or change the **camera index** as needed.

After exiting the video window (e.g., pressing `Esc`),  
the program will also display a **confusion matrix** of classification results for the test dataset.

---

###  File Structure

| File | Description |
|------|--------------|
| `main.py` | The main script. Loads data, trains the model (SVM/Random Forest) on the joint-angle dataset, and runs the live pose correction routine. |
| `utils.py` | Helper functions for angle calculation, landmark extraction, and prediction. |
| `demo.py` | Handles real-time correction and voice feedback loop. |
| `give_angle_teacher.py` | Optional utility for generating joint-angle CSV data from labeled yoga images. |
| `train_angle.csv` / `test_angle.csv` | Datasets of joint angles and pose labels. |
| `requirements.txt` | Contains all dependency specifications. |

---

### Example Run Command

```bash
python main.py
```

This starts the **live posture correction session** and displays results **visually and audibly**.

---

### Summary

The **AI-Driven Yoga Posture Correction Using Computer Vision** project leverages **pose estimation**, **machine learning**, and **text-to-speech feedback** to create a virtual yoga instructor capable of:

- Recognizing and correcting yoga postures in **real time**  
- Providing **visual + audio feedback** for proper alignment  
- Helping users practice yoga **safely, effectively, and independently**
