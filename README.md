# Computer Vision Module 3 - Assignments

This repository contains various computer vision assignments and applications.

## Structure

- `e_sig_app/` - Digital E-Signature & Watermark Streamlit App (deployable)
- `notebooks/` - Jupyter notebooks for various CV assignments
- `cv2_files/` - Course materials and resources
- `opencv_course/` - OpenCV course files

## Running the E-Signature App Locally

```bash
source ~/opencv_env/bin/activate
cd e_sig_app
streamlit run e_sig_app.py
```

## Deploying to Streamlit Cloud

When deploying via Streamlit Community Cloud:
- Set the main file path as: `e_sig_app/e_sig_app.py`
- Requirements are in: `e_sig_app/requirements.txt`
