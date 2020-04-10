from logs import logDecorator  as lD
import jsonref
import requests
requests.packages.urllib3.disable_warnings()
import ssl
import os
import torch 
from torchvision.datasets import MNIST
import torchvision

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.module1.module1'


@lD.log(logBase + '.doSomething')
def doSomething(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    print(tf.__version__)

    return

@lD.log(logBase + '.samplecode')
def samplecode(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    pass
    return

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the 
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''

    x = torch.tensor([6,6])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        print("no usable gpus")

    return