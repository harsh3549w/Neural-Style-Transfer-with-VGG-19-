
# Neural Style Transfer via Feature Space Optimization 🎨🤖

A deep learning pipeline built in PyTorch that implements Neural Style Transfer (NST). Instead of training a model to classify images, this project uses a pre-trained, frozen VGG-19 Convolutional Neural Network as a feature extractor to perform **pixel-level optimization** on a blank canvas. 

It blends the global geometric structure of a "Content" image (e.g., a photograph) with the multi-scale textural patterns of a "Style" image (e.g., a painting).

## 🧠 Architecture & Approach

This implementation dynamically deconstructs and rebuilds a `torchvision` VGG-19 model, injecting custom Loss modules directly into the computational graph.

* **Content Loss (`conv_4`):** Computes the Mean Squared Error (MSE) of raw feature maps at a deep layer to preserve high-level semantic shapes while discarding photographic textures.
* **Style Loss (`conv_1` through `conv_5`):** Computes the MSE of **Gram Matrices** across multiple network depths to capture and transfer texture correlations (brushstrokes, color patterns) at both microscopic and macroscopic scales.
* **Optimization:** Uses the **L-BFGS** second-order optimizer to iteratively update the input image's pixels via a custom `closure()` function, resulting in faster and sharper convergence compared to standard first-order optimizers like Adam.

## 🚀 Key Technical Highlights

* **Dynamic Graph Injection:** Extracted specific layers from VGG-19 and rebuilt an `nn.Sequential` pipeline to intercept feature maps mid-forward pass.
* **Compute & Memory Optimization:** Explicitly amputated all unused VGG layers after `conv_5` to minimize GPU memory footprint and accelerate the L-BFGS optimization loop.
* **Gradient Management:** Applied `.detach()` to target reference images to prevent unnecessary gradient tracking, and utilized `.requires_grad_()` on the input image tensor to shift the optimization target from network weights to raw pixels.
* **Tensor Broadcasting:** Built a custom `Normalization` layer utilizing `.view(-1, 1, 1)` to efficiently broadcast 1D ImageNet statistics across 4D image tensors natively within the `nn.Sequential` graph.
* **In-Place Override:** Forced ReLUs to `inplace=False` to preserve uncorrupted tensor states for accurate loss calculations during backpropagation.

## ⚙️ Requirements

* Python 3.8+
* PyTorch & Torchvision
* Pillow (PIL)
* Matplotlib

## 🛠️ Usage

1. Place your target photograph in the `Images/` directory and name it `content.jpg`.
2. Place your target artwork in the `Images/` directory and name it `style.jpg`.
3. Run the notebook/script. The script will automatically detect CUDA availability and adjust tensor resolutions accordingly (512px for GPU, 128px for CPU).

## 🖼️ Results
*(Note: Add your Before & After generated images here)*
