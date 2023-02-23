# Furniture Classification API

This project is an example of how to build a REST API for image classification using PyTorch and Flask. The API accepts image inputs of any size, pre-processes the image, and runs it through a pre-trained PyTorch model to make a prediction. The predicted class and confidence score are returned as JSON.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Flask

### Installation

1. Clone the repository:
`git clone https://github.com/username/repo.git`
2. cd repo 
3. Install the required packages:

<`pip install -r requirements.txt`>


### Model training

1. run the training command: `python main.py --mode train --save_path [PATH]`

### Model testing

1. run the training command: `python main.py --mode test --model_path [PATH]`

### Usage
1. Start the Flask app:
`python main.py --mode 'serve' --model_path 'path/to/the/checkpoint'`

2. Send a POST request to the `/classify` endpoint with an image file attached:

bash `curl -X POST -F "image=@/path/to/image.jpg" http://localhost:5000/classify`

### Configuration

- The PyTorch model is located in `model.py`.
- The Flask app is located in `app.py`.
- The image preprocessing pipeline is defined in `app.py`.
- The server configuration is defined in `config.py`.

## Contributing

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push to the branch.
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- This project was inspired by [PyTorch Image Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and [Building a Simple Flask API for Image Recognition](https://towardsdatascience.com/building-a-simple-flask-api-for-image-recognition-c2d8aad9c6eb).
