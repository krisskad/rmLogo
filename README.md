# rmLogo
It can remove logos from multiple files in a batch. But for now it is specific for one logo only


## Requirements
[OpenCV](https://pypi.org/project/opencv-python/)
[Numpy](https://pypi.org/project/numpy/)

## Use Case
'''Python
def process_image(input_dir: Any = None,
               output_dir: Any = None,
               single_image: Any = None,
               kernel_size: Any = None) -> None
'''

## Tutorial
__In your project file__

'''Python
from main impoprt process_image
# run process_image function with given parameters
process_image("E:\project\rmLogo\INPUT_DIR", "E:\project\rmLogo\OUTPUT_DIR", None, (5,5))

'''

## Architecture

