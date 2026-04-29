# EECS 351 Image Compression

# Overview
This project focuses on recreating our own image compression pipeline Python. It's workflow is heavily inspired by JPEG.

# Test data
All test images, greyscale and color, came from [image compression RGB 16-bit](https://imagecompression.info/test_images/).

# Example MATLAB Scripts
The core of the compression algorithm is the **Discrete Cosine Transform**. You can you see how it affects image quality by running the following MATLAB live script: `theory/dct_compression.mlx`.

You can also find a similar example for **Single Value Decomposition** based compression at `theory/svd_compression.mlx`.

# Python Compression Pipeline
The main tool in this repository is the custom Python compression pipeline. There are multiple different pipelines. To run the pipelines you must first install the required packages with `pip install -r requirements.txt`. After that, you can run pipeline by running `python -m codec.pipelines.[codec name]` where `[codec name]` is `dct_ae_codec.py`, `dct_ans_codec.py`, `dct_hf_codec.py`, or `hybrid_hf_codec.py`.