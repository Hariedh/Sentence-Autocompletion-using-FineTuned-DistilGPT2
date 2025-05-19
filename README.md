Sentence Autocompletion with DistilGPT2
This project implements a sentence autocompletion system using a fine-tuned DistilGPT2 model on the AG News dataset. It supports training the model to predict the continuation of news-related text and provides a Gradio interface for interactive text generation. The code includes model training, saving/loading of weights, and a user-friendly UI for generating completions.
Features

Model Training: Fine-tunes a DistilGPT2 model on a subset of the AG News dataset for text autocompletion.
Inference: Generates text continuations based on user-provided prompts.
Gradio Interface: Offers an interactive web UI with adjustable parameters (max length, temperature).
Model Persistence: Saves and loads model weights for future use.
GPU/CPU Support: Automatically detects and uses CUDA if available.

Prerequisites

Python 3.8 or higher
CUDA-enabled GPU (optional, for faster training/inference)
Git (for cloning the repository)

Installation

Clone the Repository:
git clone https://github.com/your-username/sentence-autocompletion.git
cd sentence-autocompletion


Create a Virtual Environment (recommended):
python -m venv autocompletion_env
source autocompletion_env/bin/activate  # On Windows: autocompletion_env\Scripts\activate


Install Dependencies:
pip install torch torchvision transformers datasets gradio tqdm

If using a GPU, ensure torch is installed with the correct CUDA version:
pip install torch --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version



Usage
Training the Model
To train the model, run the script:
python autocompletion.py


The script trains a DistilGPT2 model on a subset of the AG News dataset for 10 epochs.
The best model weights (based on lowest loss) are saved as best_model.pt in the project directory.

Running Inference with Gradio
To use the trained model for inference:
python autocompletion.py


After training, run the script again to launch the Gradio interface in your default web browser.
Enter a prompt (e.g., "The stock market"), adjust max length and temperature if desired, and view the generated completion.
If best_model.pt exists, the script loads it for inference.

Saved Files
The following file is saved after training:

best_model.pt: PyTorch model weights for the best-performing model.

Note: The tokenizer is not saved separately as it uses the pretrained distilgpt2 tokenizer, which is loaded directly from Hugging Face.
Project Structure
sentence-autocompletion/
├── autocompletion.py          # Main script for training and inference
├── best_model.pt              # Saved model weights (generated after training)
├── README.md                  # Project documentation
├── requirements.txt           # Dependency list
├── .gitignore                 # Git ignore file

Example
Input Prompt: "The stock market"Output Completion: "The stock market closes at $82 a share on Wednesday after a steady rally in U.S. stocks"
Input Prompt: "Scientists discovered"Output Completion: "Scientists discovered new species of dinosaur Pestolosaurus, a dinosaur whose legs were pulled off by human hands"
Troubleshooting

Version Conflicts: Ensure compatible versions of torch, torchvision, and transformers. Update with:pip install --upgrade torch torchvision transformers


CUDA Issues: Verify your CUDA version and install the matching torch version.
Gradio Not Loading: Check if port 7860 is free, or specify a different port:iface.launch(server_port=8080)


Model File Missing: If best_model.pt is not found, train the model first by running the script.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built with Hugging Face Transformers and Gradio.
Trained on the AG News dataset.

