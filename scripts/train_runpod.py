"""Run the original trainer with OpenCV threads bounded for the Pod CPU quota."""
import os
from pathlib import Path
import runpy
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import cv2
cv2.setNumThreads(1)
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
sys.argv[0] = str(root / 'src/train.py')
runpy.run_path(sys.argv[0], run_name='__main__')
