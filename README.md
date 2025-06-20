# Crowd Detection using AI

This project aims to **detect persons in a video** and further **analyze crowd formations** 
using computer vision and deep learning. A **crowd** is defined as **three or more people** 
standing **close together** for **10 consecutive frames**.

---

## User Story

> Train an AI model to detect persons in a video and further analyze the detected data to identify crowds.
> This involves leveraging pre-trained object detection models (like YOLOv5) and applying custom logic to detect and log crowd events.

---

##  Crowd Detection Logic

- A **crowd** = **≥3 persons** standing **close together** in a single frame.
- If the **same group** persists for **10 consecutive frames**, it's marked as a **crowd event**.
- Logs include:
  - `Frame Number`
  - `Person Count in Crowd`
- All logs are saved to `crowd_events.csv`

---

## 🛠️ Tech Stack

- Python
- OpenCV
- YOLOv5 (pre-trained)
- NumPy & Pandas

---


