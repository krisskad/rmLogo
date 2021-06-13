# rmLogo
It can remove logos from multiple files in a batch. But for now it is specific for one logo only

#### Basic Installation
  1. Clone the repository somewhere you're happy to have the folder:
    
    $ cd my/directory/of/choice
    $ git clone https://github.com/krisskad/rmLogo.git

  2. Install the dependencies:
  
    $ cd path/to/rmLogo
    $ python -m pip install -r requirements.txt
    
## Requirements
[OpenCV](https://pypi.org/project/opencv-python/) <br>
[Numpy](https://pypi.org/project/numpy/)

# Advanced Usage

    Computer vision program to watermark and logos. Utilises computer
    vision and image analysis to return approximated feature dimensions.
    
    positional arguments:
      input_dir             The path of the image folder which containts images [JPG, PNG].
      output_dir            The path of the output folder to save output images [It will store the image with same name as original image]
    
    optional arguments:
      single_image          You can provide a single image to test the model [just provide the image path and output_dir and make other as None]
      kernel_size 2         To perform smoothing process we need odd values [the greather the value more soft will be result. It will damage quality]


    def process_image(input_dir: Any = input_dir,
                   output_dir: Any = output_dir,
                   single_image: Any = None,
                   kernel_size: Any = None)

## Tutorial
__In your project file__

    from main impoprt process_image
    # run process_image function with given parameters
    process_image("E:\project\rmLogo\INPUT_DIR", "E:\project\rmLogo\OUTPUT_DIR", None, (5,5))

@Author krisskad