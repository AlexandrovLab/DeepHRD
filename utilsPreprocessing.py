import os
import glob
import numpy as np
import openslide
from PIL import Image
import math
import re

#	The associated functions for processing tissue slides have been adapted from an open 
#	source tutorial on working with pathology slides from IBM: 
#		[1] https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images


def getNumberOfSlides(path, extension):
	"""
	Obtain the total number of WSI training slide images.

	Returns:
		The total number of WSI training slide images.
	"""
	slides = glob.glob1(path, "*." + extension)
	return (len(slides), slides)


def writeSlideNumberSampleNameToFile (projectDir, num_train_images, train_images):
	with open(projectDir + "slideNumberToSampleName.txt", "w") as f:
		for i in range(0, num_train_images, 1):
			print("\t".join([str(i+1).zfill(3), train_images[i].split("/")[-1].split(".")[0]]), file=f)



def parseDimensionsFromImageFilename(filename):
	"""
	Parse an image filename to extract the original width and height and the converted width and height.

	Example:
		"TUPAC-TR-011-32x-97103x79079-3034x2471-tile_summary.png" -> (97103, 79079, 3034, 2471)

	Args:
		filename: The image filename.

	Returns:
		Tuple consisting of the original width, original height, the converted width, and the converted height.
	"""
	m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
	large_w = int(m.group(1))
	large_h = int(m.group(2))
	small_w = int(m.group(3))
	small_h = int(m.group(4))
	return large_w, large_h, small_w, small_h


def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.

  Args:
    np_img: The image represented as a NumPy array.

  Returns:
     The NumPy array converted to a PIL Image.
  """
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return(Image.fromarray(np_img))


def mask_rgb(rgb, mask):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  result = rgb * np.dstack([mask, mask, mask])
  return(result)


def openSlide(filename):
	"""
	Open a whole-slide image (*.svs, etc).

	Args:
		filename: Name of the slide file.

	Returns:
		An OpenSlide object representing a whole-slide image.
	"""
	try:
		slide = openslide.open_slide(filename)
	except OpenSlideError:
		slide = None
	except FileNotFoundError:
		slide = None
	return(slide)


def open_image_np(filename):
	"""
	Open an image (*.jpg, *.png, etc) as an RGB NumPy array.

	Args:
		filename: Name of the image file.

	returns:
		A NumPy representing an RGB image.
	"""
	pil_img = Image.open(filename)
	np_img = np.asarray(pil_img)
	return(np_img)


def getTrainingImagePath(slide_number, outputPath, projectPrefix, outputExtension, scaleFactor, large_w=None, large_h=None, small_w=None, small_h=None):
	"""
	Convert slide number and optional dimensions to a training image path. If no dimensions are supplied,
	the corresponding file based on the slide number will be looked up in the file system using a wildcard.

	Example:
		5 -> ../data/training_png/TUPAC-TR-005-32x-49920x108288-1560x3384.png

	Args:
		slide_number: The slide number.
		large_w: Large image width.
		large_h: Large image height.
		small_w: Small image width.
		small_h: Small image height.

	Returns:
		 Path to the image file.
	"""
	padded_sl_num = str(slide_number).zfill(3)
	if large_w is None and large_h is None and small_w is None and small_h is None:
		wildcard_path = os.path.join(outputPath, projectPrefix + padded_sl_num + "*." + outputExtension)
		img_path = glob.glob(wildcard_path)[0]
	else:
		img_path = os.path.join(outputPath, projectPrefix + padded_sl_num + "-" + str(scaleFactor) + "x-" + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + outputExtension)
	return(img_path)

def get_training_slide_path(projectPath, slide_number):
	"""
	Convert slide number to a path to the corresponding WSI training slide file.

	Example:
		5 -> ../data/training_slides/TUPAC-TR-005.svs

	Args:
		slide_number: The slide number.

	Returns:
		Path to the WSI training slide file.
	"""
	slide_filepath = os.path.join(projectPath, str(slide_number))
	return (slide_filepath)


def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
	"""
	Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
	a column tile size.

	Args:
		rows: Number of rows.
		cols: Number of columns.
		row_tile_size: Number of pixels in a tile row.
		col_tile_size: Number of pixels in a tile column.

	Returns:
		Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
		into given the row tile size and the column tile size.
	"""
	num_row_tiles = math.ceil(rows / row_tile_size)
	num_col_tiles = math.ceil(cols / col_tile_size)
	return(num_row_tiles, num_col_tiles)

def mask_percent(np_img):
	"""
	Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

	Args:
		np_img: Image as a NumPy array.

	Returns:
		The percentage of the NumPy array that is masked.
	"""
	if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
		np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
		mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
	else:
		mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
	return(mask_percentage)



def tissue_percent(np_img):
	"""
	Determine the percentage of a NumPy array that is tissue (not masked).

	Args:
		np_img: Image as a NumPy array.

	Returns:
		The percentage of the NumPy array that is tissue.
	"""
	return(100 - mask_percent(np_img))





