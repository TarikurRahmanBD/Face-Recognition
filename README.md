# Face Recognition

## Project Overview

This is a simple face recognition demo written in Python. It detects faces in an input image and attempts to identify them using a trained model. The repository contains two approaches:

- Original approach (commented in earlier commits) uses the `face_recognition` library (which depends on `dlib`).
- Fallback approach (currently implemented in `main.py`) uses OpenCV's Haar cascade for detection and the LBPH face recognizer — this does not require compiling native C++ extensions and works with just `opencv-python` and `numpy`.

The project expects training images in the `train/` folder (one image per person, filename used as the label) and a test image at `test/test.jpg`.

## Project Structure

- `main.py` — entry point (OpenCV LBPH recognizer by default).
- `train/` — directory for training images (one image per person).
- `test/` — directory for test images (expects `test.jpg`).
- `output.jpg` — generated output image after running the script.

## How to Install & Run

Below are two options depending on whether you want to use the original `face_recognition` (dlib) approach or the OpenCV-only fallback that is already in `main.py`.

1) Recommended / Fast: Run the OpenCV-only fallback (no native build)

- Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- Install required packages and run:

```powershell
pip install --upgrade pip
pip install opencv-python numpy
python main.py
```

The script will train an LBPH recognizer on images found in `train/` and write the result to `output.jpg`.

2) Optional: Use `face_recognition` (dlib) — more accurate but requires native build tools

- On Windows you must have Visual Studio Build Tools (C++ workload) and CMake installed, or use Anaconda/Miniconda which often provides prebuilt `dlib` packages.

- If you prefer to install manually with pip (may require compiling dlib):

```powershell
# Install Visual Studio Build Tools (Desktop development with C++) and CMake first
python -m pip install face_recognition opencv-python numpy
```

- Conda alternative (if you have conda):

```bash
conda install -c conda-forge dlib face_recognition opencv numpy
```

Note: If `dlib` fails to build on Windows, install the Visual Studio Build Tools including the C++ toolchain and CMake, then retry the pip install.

## Usage Notes

- Place one clear face image per person inside the `train/` folder. The filename (without extension) will be used as the person's label.
- Put a test image at `test/test.jpg` and run `python main.py`.
- The script writes `output.jpg` with rectangles and labels drawn.
- If you see many `Unknown` labels, try adding more training images or tuning the LBPH confidence threshold in `main.py`.

## Features

- Detects faces in a test image using Haar cascades.
- Trains an LBPH face recognizer from images in `train/`.
- Writes annotated output to `output.jpg`.
- Two setup paths: lightweight OpenCV-only or the higher-accuracy `face_recognition` (dlib) option.

## Troubleshooting

- `ModuleNotFoundError: No module named 'face_recognition'` — install `face_recognition` or use the OpenCV fallback.
- `Failed building wheel for dlib` — install Visual Studio Build Tools (C++), CMake, or use conda to obtain a prebuilt wheel.
- Faces not detected: ensure training and test images are clear, front-facing, and reasonably sized. Haar cascades can be sensitive to scale — try resizing images.

## About the Developer

- **Name:** Tarikur Rahman
- **GitHub:** https://github.com/tarikurrahmanbd
- **Portfolio:** https://yourtarikur.netlify.app/
- **Social/Handle:** tarikurrahman08
- **Email:** tarikurrahman2008@gmail.com

## License

This project is released under the **MIT License**.

---

If you want, I can:

- Add a `requirements.txt` or `pyproject.toml`.
- Improve `main.py` to support multiple training images per person and a simple CLI.
- Revert to the original `face_recognition`-based script and add install instructions tailored to your environment.

Choose next steps and I'll implement them.
