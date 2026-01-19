## 📸 Face Recognition Attendance System
📋 Overview
A simple, lightweight face recognition attendance system built with Python and OpenCV. No complex installations required - just 2 Python packages!

## ✨ Features
✅ No dlib installation - Uses only OpenCV

✅ Real-time face recognition from webcam

✅ Automatic attendance marking in CSV format

✅ Live face capture during runtime

✅ Simple template matching - No complex AI models

✅ Cross-platform - Works on Windows, Mac, Linux

✅ No training required - Learn faces on the fly

## 🚀 Quick Start
- Download all project files to a folder
- Open with your code editor
- Run: python attendance.py

Press 'C' to capture faces of people

## 🎯 How to Use
1. First Time Setup in bash

 ` pip install opencv-python numpy`

 ` python attendance.py`
 
2. Adding People to the System
Method A: Live Capture (Easiest)
Run python attendance.py

When person is in front of camera, press 'T'

Enter person's name when prompted

System automatically saves and learns the face

Method B: Manual Photo Addition
Add photos to faces/ folder

Name format: Name_anything.jpg

Example: John_office.jpg, Sarah_home.jpg

System uses part before first underscore as name

3. Taking Attendance
Run python attendance.py

System automatically detects and recognizes faces

Attendance is marked automatically when confidence > 60%

Press 'Q' to quit

## 📊 Attendance Records
File: attendance.csv

Format: Name, Date, Time, Status

Example: John, 2024-01-15, 09:30:15, Present

No duplicate entries for the same day

## 🔧 Requirements
Python 3.6 or higher

OpenCV 4.x

NumPy

Webcam
 
