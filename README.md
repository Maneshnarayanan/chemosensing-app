# ChemoSense: Computer-Assisted Image-Based Chemosensing System

ChemoSense is a modular Python-based pipeline for converting colorimetric responses of paper-based sensors into quantitative analytical data. It utilizes CIELAB color space and ΔE (CIE76) parameters for high-sensitivity transduction.

## Features
- **Modular Design**: Independent modules for Image IO, ROI extraction, and Color Science.
- **CIELAB Transformation**: Standardized sRGB to CIELAB (D65) conversion.
- **Quantitative Analysis**: Linear and Polynomial regression modeling for concentration calibration.
- **Interactive Web App**: Built with Streamlit for real-time analysis and visualization.
- **Chemometrics Ready**: Integrated support for PCA and supervised ML classification.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/project1.git
   cd project1
   ```

2. **Setup virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Web App
```bash
streamlit run app.py
```

### Run Demonstration Script
```bash
python main.py
```

## Project Structure
- `src/chemosense/`: Core packages.
- `notebooks/`: Jupyter notebooks for demonstration.
- `tests/`: Unit tests for verification.
- `app.py`: Streamlit web interface.
