<div align="center">
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Orbitron&size=28&duration=3000&pause=1000&color=00F7FF&center=true&vCenter=true&width=700&lines=Neural+Style+Transfer;AdaIN+Powered+Image+Generation;PyTorch+%7C+Flask+%7C+Computer+Vision;Transforming+Photos+Into+Artwork" />
</p>
</div>

## 🎨 NEURALFORGE AI

NeuralForge AI is a deep learning project that transforms ordinary photographs into artistic masterpieces by transferring the visual style of one image onto another. The project implements the **Adaptive Instance Normalization (AdaIN)** algorithm using **PyTorch**, trains a custom decoder network, and provides a user-friendly **Flask web application** for real-time style transfer.

Project Link [Deployed on Azure] : <i style="color : blue"> [https://neuralforge-ai.onrender.com](https://neuralforgeai-c0gyctecbbcaaad2.southindia-01.azurewebsites.net/) </i>
---

## 🚀 Features

* Neural Style Transfer using AdaIN
* Custom-trained decoder network
* VGG-19 based feature encoder
* Real-time image stylization through a Flask web interface
* Adjustable style intensity using Alpha blending
* Support for multiple artistic styles
* GPU acceleration with CUDA support
* Resume training from saved checkpoints
* Automatic model checkpointing and output generation

---

## 📸 How It Works

NeuralForge AI accepts:

* **Content Image** → The image whose structure and objects should be preserved.
* **Style Image** → The image whose artistic appearance should be transferred.

The model:

1. Extracts feature maps using a pretrained VGG Encoder.
2. Applies Adaptive Instance Normalization (AdaIN) to align content features with style statistics.
3. Uses a custom-trained Decoder network to reconstruct the final stylized image.
4. Allows blending between content and style using an Alpha parameter.

---

## 🏗️ Project Architecture

```text
Content Image ──► VGG Encoder → Content Features──┐
                                                  ▼
                                                AdaIN ──► Decoder ──► Stylized Output
                                                  ▲
Style Image ───► VGG Encoder → Style Features  ───┘
```

---

## 🧠 Model Architecture

### Encoder

* Based on VGG-19
* Pretrained weights loaded from `vgg_normalised.pth`
* Feature extraction layers:

  * ReLU1_1
  * ReLU2_1
  * ReLU3_1
  * ReLU4_1

### AdaIN Layer

Adaptive Instance Normalization transfers style by matching:

* Channel-wise Mean
* Channel-wise Standard Deviation
between content and style feature maps.
---
<p align="center">
  <img width="379" height="148"
       alt="ReferenceDLModelPipeline"
       src="https://github.com/user-attachments/assets/f2dd2bf0-b061-4e13-b19e-154770a96f8e">
</p>

---


### Decoder

Custom convolutional decoder consisting of:

* Reflection Padding
* Convolution Layers
* ReLU Activations
* Upsampling Layers

The decoder learns to reconstruct images from AdaIN-transformed feature maps.

---

## 📂 Dataset

| Source | Description | Count | Size |
|---------|------------|--------:|--------:|
| COCO 2017 Test Dataset | Content Images for Structure Preservation | ~40,670 Images | ~6.65 GB |
| WikiArt Dataset Subset | Artistic Style Reference Images | ~8,476 Paintings | ~3.95 GB |
| **Total Dataset Used** | Content + Style Images | **~49,000 Images** | **~10.6 GB** |

**Total Storage Size:** ~10.6 GB
---

## 📁 Project Structure

```text
NeuralForgeAI/
│
├── app.py
├── train.py
├── vgg_normalised.pth
│
├── utils/
│   ├── models.py
│   └── utils.py
│
├── templates/
│   └── index.html
│
├── static_uploads/
│
├── examples/
│
└── experiment/
    └── checkpoints/
```

---

## ⚙️ Training

### Train from Scratch

```bash
python train.py \
--batch_size=4 \
--epochs=10 \
--experiment="train1" \
--vgg="vgg_normalised.pth" \
--content_dir="content_dataset" \
--style_dir="style_dataset"
```

### Resume Training

```bash
python train.py \
--resume \
--decoder_path="decoder_10.pth" \
--optimizer_path="optimizer_10.pth"
```

---

## 📉 Loss Functions

### Content Loss

Measures similarity between generated image features and AdaIN-transformed features.

```text
Lcontent = MSE(Fgenerated, Ftarget)
```

### Style Loss

Matches feature statistics across multiple VGG layers.

```text
Lstyle = Σ(MSE(μg, μs) + MSE(σg, σs))
```

### Total Loss

```text
Ltotal = λc * Lcontent + λs * Lstyle
```

Default:

```text
Content Weight = 1.0
Style Weight = 5.0
```

---

## 🌐 Web Application

The Flask application allows users to:

* Upload a content image
* Upload a style image
* Adjust style strength using Alpha
* Generate stylized artwork instantly
* Download the generated image

### Alpha Parameter

| Alpha | Result              |
| ----- | ------------------- |
| 0.0   | Original Content    |
| 0.5   | Balanced Blend      |
| 1.0   | Full Style Transfer |

---

## 🛠️ Tech Stack

| Layer | Technologies Used |
|--------|------------------|
| 🧠 Deep Learning | PyTorch, Torchvision |
| ⚙️ Backend Framework | Flask |
| 📝 Form Handling | Flask-WTF, WTForms |
| 🎨 Frontend | HTML5, CSS3, Bootstrap, JavaScript |
| 🖼️ Image Processing | Pillow (PIL) |
| 🚀 Model Architecture | VGG Encoder + AdaIN + Custom Decoder |
---

## 📊 Training Outputs

During training the framework automatically:

* Logs Content Loss
* Logs Style Loss
* Saves model checkpoints
* Saves optimizer checkpoints
* Generates sample stylized outputs

Example output:

```text
Loss: 125.84
Content Loss: 2.16
Style Loss: 123.67
```

---

## 🔮 Future Improvements

* Multiple style blending
* Video style transfer
* REST API deployment
* Docker support
* ONNX export
* Mobile application
* Cloud deployment on AWS/Azure

---

## 👩‍💻 Author

**Mrunalini Pachpute**

Deep Learning | Computer Vision | Generative AI

NeuralForge AI was built to explore artistic image generation through neural style transfer and demonstrate practical applications of deep learning in creative AI.

---

## ⭐ Acknowledgements

This project is inspired by:

* Gatys et al. — Neural Style Transfer
* Huang & Belongie — Adaptive Instance Normalization (AdaIN)
* PyTorch Deep Learning Community

---

## License

MIT License

Feel free to use, modify, and contribute to the project.
