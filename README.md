# SmallTrain
The machine learning library enabling "small train", which requires lower machine power and fewer training data.

# Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Docs & Community](#docs--community)
* [Installation](#installation)
	* [Standalone Docker Application](#standalone-docker-application)
	* [Python Library](#python-library)
* [Tutorials](#tutorials)
	* [Launch Jupyter Lab Notebook](#launch-jupyter-lab-notebook)
* [Contributing](#contributing)


## Overview
SmallTrain is a machine learning library running on TensorFlow (and Keras and PyTorch in the future version).
It enables small train, which requires lower machine power and fewer training data because
you can develop your machine learning(ML) models from our pre-trained models.


## Features

*	Easy to develop for POC to production
*	Almost no programming for building your pre-trained model.
*	Available as both TensorFlow and Pytorch wrapper.
*	Always adapting to algorithms which evolve.
*	Build using state-of-the-art algorithms from Scientific and Mathematical papers
*	Accuracy is always going to be better even with minimal data and training time
*	Licensed under MIT Open Source


## Docs & Community

* [Website and Documentation](https://www.smalltrain.org/en/)
* [Release Notes](https://www.smalltrain.org/en/docs/release-logs/)


##	Installation

### Standalone Docker Application

See [`this guide`](/docker/README.md) for detailed instruction to build SmallTrain as a standalone docker container from source.

### Python Library
This is a [SmallTrain](https://www.smalltrain.org/en/) module available through the
[pypi registry](https://pypi.org/project/smalltrain/).

Installation is done using the
[`pip install` command](https://pypi.org/project/smalltrain/):

```bash
$ pip install smalltrain
```

Follow [our installing guide](https://pypi.org/project/smalltrain/)
for more information.

## Tutorials

You can run tutorial codes on Jupyter Lab Notebook.

### Launch Jupyter Lab Notebook

```bash
# Enable password
jupyter notebook password
# Run Jupyter Lab Notebook
cd /var/smalltrain/tutorials
nohup jupyter lab &
```

For example, a tutorial notebook for image detection is available on http://YOURHOST:JUPYTER_NOTEBOOK_PORT/lab/tree/image_recognition/notebooks/cifar10.ipynb
(Default JUPYTER_NOTEBOOK_PORT is 8888).

See [Tutorials](https://www.smalltrain.org/en/docs/tutorials/)  for more tutorials.


## Contributing

[Contributing Guide](https://www.smalltrain.org/en/docs/contribution-guidelines/)

[List of all contributors](https://github.com/geek-guild/smalltrain/graphs/contributors)


## Licence

[MIT Licence](https://github.com/geek-guild/smalltrain/blob/master/LICENSE)

