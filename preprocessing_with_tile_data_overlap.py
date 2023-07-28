import math
import multiprocessing
import numpy as np
import openslide
from openslide import OpenSlideError
import os
import PIL
from PIL import Image
import re
import sys
import utilsPreprocessing as util 
import sys
import skimage.morphology as sk_morphology
import pandas as pd
from enum import Enum
import skimage.color as sk_color
import random


# # BASE_DIR = "/Users/ebergstr/Desktop/lab/thesis/Aim3/histology/BRCA/debug/"
# # SKIPPED_SAMPLES = os.path.join(BASE_DIR, "skipped_samples.txt")

# # PROJECT = "BRCA"
# TRAIN_PREFIX = PROJECT + "-"
# SRC_TRAIN_DIR = os.path.join(BASE_DIR, PROJECT + "/")
# SRC_TRAIN_EXT = "svs"
# DEST_TRAIN_EXT = "png"
SCALE_FACTOR = 32
# DEST_TRAIN_DIR = os.path.join(BASE_DIR, PROJECT + "_" + DEST_TRAIN_EXT)
# TISSUE_HIGH_THRESH = 80
# TISSUE_LOW_THRESH = 10
# RESOLUTION = '5x'
# ROW_TILE_SIZE = 256
# COL_TILE_SIZE = 256
# NUM_TOP_TILES = 100000

# HSV_PURPLE = 270
# HSV_PINK = 330

# FILTER_RESULT_TEXT = "filtered"
# FILTER_DIR = os.path.join(BASE_DIR, "filter_" + DEST_TRAIN_EXT)

# TOP_TILES_SUFFIX = "top_tile_summary"
# TOP_TILES_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_" + DEST_TRAIN_EXT)
# TILE_DIR = os.path.join(BASE_DIR, "tiles_" + DEST_TRAIN_EXT)
# TILE_SUFFIX = "tile"



def get_tile_image_path(tile):
	"""
	Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
	pixel width, and pixel height.

	Args:
		tile: Tile object.

	Returns:
		Path to image tile.
	"""
	t = tile
	padded_sl_num = str(t.slide_num).zfill(3)
	tile_path = os.path.join(TILE_DIR, padded_sl_num,
													 TRAIN_PREFIX + padded_sl_num + "-" + TILE_SUFFIX + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
														 t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + "." + DEST_TRAIN_EXT)
	return(tile_path)


def get_filter_image_result(slide_number):
	"""
	Convert slide number to the path to the file that is the final result of filtering.

	Example:
		5 -> ../data/filter_png/TUPAC-TR-005-32x-49920x108288-1560x3384-filtered.png

	Args:
		slide_number: The slide number.

	Returns:
		Path to the filter image file.
	"""
	padded_sl_num = str(slide_number).zfill(3)
	training_img_path = util.getTrainingImagePath(slide_number, DEST_TRAIN_DIR, TRAIN_PREFIX, DEST_TRAIN_EXT, SCALE_FACTOR)
	large_w, large_h, small_w, small_h = util.parseDimensionsFromImageFilename(training_img_path)
	img_path = os.path.join(FILTER_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
		SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
		small_h) + "-" + FILTER_RESULT_TEXT + "." + DEST_TRAIN_EXT)
	return(img_path)


def summary_stats(tile_summary):
	"""
	Obtain various stats about the slide tiles.

	Args:
		tile_summary: TileSummary object.

	Returns:
		 Various stats about the slide tiles as a string.
	"""
	return "Original Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
				 "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
				 "Scale Factor: 1/%dx\n" % tile_summary.scale_factor + \
				 "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
				 "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
				 "Total Mask: %3.2f%%, Total Tissue: %3.2f%%\n" % (
					 tile_summary.mask_percentage(), tile_summary.tissue_percentage) + \
				 "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
				 " %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
					 tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
				 " %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
					 tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
					 TISSUE_HIGH_THRESH) + \
				 " %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
					 tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
				 " %5d (%5.2f%%) tiles =0%% tissue" % (tile_summary.none, tile_summary.none / tile_summary.count * 100)




# def parse_dimensions_from_image_filename(filename):
# 	"""
# 	Parse an image filename to extract the original width and height and the converted width and height.

# 	Example:
# 		"TUPAC-TR-011-32x-97103x79079-3034x2471-tile_summary.png" -> (97103, 79079, 3034, 2471)

# 	Args:
# 		filename: The image filename.

# 	Returns:
# 		Tuple consisting of the original width, original height, the converted width, and the converted height.
# 	"""
# 	m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
# 	large_w = int(m.group(1))
# 	large_h = int(m.group(2))
# 	small_w = int(m.group(3))
# 	small_h = int(m.group(4))
# 	return large_w, large_h, small_w, small_h


def small_to_large_mapping(small_pixel, large_dimensions):
	"""
	Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.

	Args:
		small_pixel: The scaled-down width and height.
		large_dimensions: The width and height of the original whole-slide image.

	Returns:
		Tuple consisting of the scaled-up width and height.
	"""
	small_x, small_y = small_pixel
	large_w, large_h = large_dimensions
	large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
	large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
	return(large_x, large_y)


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
	"""
	Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
	and eosin are purplish and pinkish, which do not have much green to them.

	Args:
		np_img: RGB image as a NumPy array.
		green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
		avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
		overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
		output_type: Type of array to return (bool, float, or uint8).

	Returns:
		NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
	"""
	
	g = np_img[:, :, 1]
	gr_ch_mask = (g < green_thresh) & (g > 0)
	mask_percentage = util.mask_percent(gr_ch_mask)

	if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
		new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
		# print(
		# 	"Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
		# 		mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
		gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
	np_img = gr_ch_mask

	if output_type == "bool":
		pass
	elif output_type == "float":
		np_img = np_img.astype(float)
	else:
		np_img = np_img.astype("uint8") * 255
	return(np_img)


def filter_grays(rgb, tolerance=15, output_type="bool"):
	"""
	Create a mask to filter out pixels where the red, green, and blue channel values are similar.

	Args:
		np_img: RGB image as a NumPy array.
		tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
		output_type: Type of array to return (bool, float, or uint8).

	Returns:
		NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
	"""
	(h, w, c) = rgb.shape

	rgb = rgb.astype(int)
	rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
	rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
	gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
	result = ~(rg_diff & rb_diff & gb_diff)

	if output_type == "bool":
		pass
	elif output_type == "float":
		result = result.astype(float)
	else:
		result = result.astype("uint8") * 255
	return(result)


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
							 display_np_info=False):
	"""
	Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
	red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

	Args:
		rgb: RGB image as a NumPy array.
		red_lower_thresh: Red channel lower threshold value.
		green_upper_thresh: Green channel upper threshold value.
		blue_upper_thresh: Blue channel upper threshold value.
		output_type: Type of array to return (bool, float, or uint8).
		display_np_info: If True, display NumPy array info and filter time.

	Returns:
		NumPy array representing the mask.
	"""

	r = rgb[:, :, 0] > red_lower_thresh
	g = rgb[:, :, 1] < green_upper_thresh
	b = rgb[:, :, 2] < blue_upper_thresh
	result = ~(r & g & b)
	if output_type == "bool":
		pass
	elif output_type == "float":
		result = result.astype(float)
	else:
		result = result.astype("uint8") * 255
	return(result)


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool", display_np_info=False):
	"""
	Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
	red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
	Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
	lower threshold value rather than a blue channel upper threshold value.

	Args:
		rgb: RGB image as a NumPy array.
		red_upper_thresh: Red channel upper threshold value.
		green_lower_thresh: Green channel lower threshold value.
		blue_lower_thresh: Blue channel lower threshold value.
		output_type: Type of array to return (bool, float, or uint8).
		display_np_info: If True, display NumPy array info and filter time.

	Returns:
		NumPy array representing the mask.
	"""

	r = rgb[:, :, 0] < red_upper_thresh
	g = rgb[:, :, 1] > green_lower_thresh
	b = rgb[:, :, 2] > blue_lower_thresh
	result = ~(r & g & b)
	if output_type == "bool":
		pass
	elif output_type == "float":
		result = result.astype(float)
	else:
		result = result.astype("uint8") * 255
	return(result)

def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
								display_np_info=False):
	"""
	Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
	red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

	Args:
		rgb: RGB image as a NumPy array.
		red_upper_thresh: Red channel upper threshold value.
		green_upper_thresh: Green channel upper threshold value.
		blue_lower_thresh: Blue channel lower threshold value.
		output_type: Type of array to return (bool, float, or uint8).
		display_np_info: If True, display NumPy array info and filter time.

	Returns:
		NumPy array representing the mask.
	"""

	r = rgb[:, :, 0] < red_upper_thresh
	g = rgb[:, :, 1] < green_upper_thresh
	b = rgb[:, :, 2] > blue_lower_thresh
	result = ~(r & g & b)
	if output_type == "bool":
		pass
	elif output_type == "float":
		result = result.astype(float)
	else:
		result = result.astype("uint8") * 255
	return(result)

def filter_red_pen(rgb, output_type="bool"):
	"""
	Create a mask to filter out red pen marks from a slide.

	Args:
		rgb: RGB image as a NumPy array.
		output_type: Type of array to return (bool, float, or uint8).

	Returns:
		NumPy array representing the mask.
	"""
	result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
					 filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
					 filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
					 filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
					 filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
					 filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
					 filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
					 filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
					 filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
	if output_type == "bool":
		pass
	elif output_type == "float":
		result = result.astype(float)
	else:
		result = result.astype("uint8") * 255
	return(result)


def filter_green_pen(rgb, output_type="bool"):
	"""
	Create a mask to filter out green pen marks from a slide.

	Args:
		rgb: RGB image as a NumPy array.
		output_type: Type of array to return (bool, float, or uint8).

	Returns:
		NumPy array representing the mask.
	"""
	result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
					 filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
					 filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
					 filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
					 filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
					 filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
					 filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
					 filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
					 filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
					 filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
					 filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
					 filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
					 filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
					 filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
					 filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
	if output_type == "bool":
		pass
	elif output_type == "float":
		result = result.astype(float)
	else:
		result = result.astype("uint8") * 255
	return(result)


def filter_blue_pen(rgb, output_type="bool"):
	"""
	Create a mask to filter out blue pen marks from a slide.

	Args:
		rgb: RGB image as a NumPy array.
		output_type: Type of array to return (bool, float, or uint8).

	Returns:
		NumPy array representing the mask.
	"""
	result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
					 filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
					 filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
					 filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
					 filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
					 filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
					 filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
					 filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
					 filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
					 filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
					 filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
					 filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
	if output_type == "bool":
		pass
	elif output_type == "float":
		result = result.astype(float)
	else:
		result = result.astype("uint8") * 255
	return(result)


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
	"""
	Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
	is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
	reduce the amount of masking that this filter performs.

	Args:
		np_img: Image as a NumPy array of type bool.
		min_size: Minimum size of small object to remove.
		avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
		overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
		output_type: Type of array to return (bool, float, or uint8).

	Returns:
		NumPy array (bool, float, or uint8).
	"""

	rem_sm = np_img.astype(bool)  # make sure mask is boolean

	rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
	
	mask_percentage = util.mask_percent(rem_sm)
	if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
		new_min_size = min_size / 2
		# print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
		# 	mask_percentage, overmask_thresh, min_size, new_min_size))
		rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
	np_img = rem_sm

	if output_type == "bool":
		pass
	elif output_type == "float":
		np_img = np_img.astype(float)
	else:
		np_img = np_img.astype("uint8") * 255

	return(np_img)


def apply_image_filters(np_img, slide_num=None, info=None, save=False):
	"""
	Apply filters to image as NumPy array and optionally save and/or display filtered images.

	Args:
		np_img: Image as NumPy array.
		slide_num: The slide number (used for saving/displaying).
		info: Dictionary of slide information (used for HTML display).
		save: If True, save image.

	Returns:
		Resulting filtered image as a NumPy array.
	"""
	
	rgb = np_img
		
	mask_not_green = filter_green_channel(rgb)
	rgb_not_green = util.mask_rgb(rgb, mask_not_green)
	
	mask_not_gray = filter_grays(rgb)
	rgb_not_gray = util.mask_rgb(rgb, mask_not_gray)

	mask_no_red_pen = filter_red_pen(rgb)
	rgb_no_red_pen = util.mask_rgb(rgb, mask_no_red_pen)
	
	mask_no_green_pen = filter_green_pen(rgb)	
	rgb_no_green_pen = util.mask_rgb(rgb, mask_no_green_pen)
	
	mask_no_blue_pen = filter_blue_pen(rgb)
	rgb_no_blue_pen = util.mask_rgb(rgb, mask_no_blue_pen)

	mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
	rgb_gray_green_pens = util.mask_rgb(rgb, mask_gray_green_pens)
	
	mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
	rgb_remove_small = util.mask_rgb(rgb, mask_remove_small)

	img = rgb_remove_small
	return(img)


def training_slide_to_image(slide_number, slide):
	"""
	Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.

	Args:
		slide_number: The slide number.
	"""

	img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_number, slide)
	
	img_path = util.getTrainingImagePath(slide_number, DEST_TRAIN_DIR, TRAIN_PREFIX, DEST_TRAIN_EXT, SCALE_FACTOR,  large_w, large_h, new_w, new_h)
	if not os.path.exists(DEST_TRAIN_DIR):
		os.makedirs(DEST_TRAIN_DIR)
	img.save(img_path)


def slide_to_scaled_pil_image(slide_number, slide):
	"""
	Convert a WSI training slide to a scaled-down PIL image.

	Args:
		slide_number: The slide number.

	Returns:
		Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
	"""

	slide_filepath = os.path.join(SRC_TRAIN_DIR, slide)
	slide = util.openSlide(slide_filepath)

	
	large_w, large_h = slide.dimensions
	
	new_w = math.floor(large_w / SCALE_FACTOR)
	new_h = math.floor(large_h / SCALE_FACTOR)

	try:
		if abs(0.25  - float(slide.properties['openslide.mpp-x'])) <  abs(0.5  - float(slide.properties['openslide.mpp-x'])):
			objective_power = 40
		else:
			objective_power = 20
	except:
		objective_power = 10
		print(slide, file=skippedSamps, flush=True, end="\t")
		for x in slide.properties:
			print("\t".join([str(x), str(slide.properties[x])]), end = "\t", flush=True, file=skippedSamps)
		print(file=skippedSamps, flush=True)

	with open(BASE_DIR + "objectiveInfo.txt", "a") as out:
		print("\t".join([str(slide_number).zfill(3), str(objective_power)]), file=out)

	level = slide.get_best_level_for_downsample(SCALE_FACTOR)
	whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
	whole_slide_image = whole_slide_image.convert("RGB")
	img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
	return(img, large_w, large_h, new_w, new_h)


def apply_filters_to_image(slide_num, save=True):
	"""
	Apply a set of filters to an image and optionally save and/or display filtered images.

	Args:
		slide_num: The slide number.
		save: If True, save filtered images.

	Returns:
		Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
		(used for HTML page generation).
	"""
	info = dict()

	if save and not os.path.exists(FILTER_DIR):
		os.makedirs(FILTER_DIR)
	
	img_path = util.getTrainingImagePath(slide_num, DEST_TRAIN_DIR, TRAIN_PREFIX, DEST_TRAIN_EXT, SCALE_FACTOR)#,  large_w, large_h, new_w, new_h)
	np_orig = util.open_image_np(img_path)
	filtered_np_img = apply_image_filters(np_orig, slide_num, info, save=False)
	
	if save:
		result_path = get_filter_image_result(slide_num)
		pil_img = util.np_to_pil(filtered_np_img)
		pil_img.save(result_path)

	return(filtered_np_img, info)



def get_tile_indices(rows, cols, row_tile_size, col_tile_size, stepSize):
	"""
	Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).

	Args:
		rows: Number of rows.
		cols: Number of columns.
		row_tile_size: Number of pixels in a tile row.
		col_tile_size: Number of pixels in a tile column.

	Returns:
		List of tuples representing tile coordinates consisting of starting row, ending row,
		starting column, ending column, row number, column number.
	"""
	indices = list()
	num_row_tiles, num_col_tiles = util.get_num_tiles(rows, cols, row_tile_size, col_tile_size)
	for r in np.arange(0, num_row_tiles-1+(1-OVERLAP), stepSize*(1-OVERLAP)):
	# for r in range(0, num_row_tiles, stepSize):
		start_r = r * row_tile_size
		end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
		for c in np.arange(0, num_col_tiles-1+(1-OVERLAP), stepSize*(1-OVERLAP)):
		# for c in range(0, num_col_tiles, stepSize):
			start_c = c * col_tile_size
			end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
			# indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
			indices.append((int(start_r), int(end_r), int(start_c), int(end_c), int(r) + 1, int(c) + 1))
	return(indices)


class TissueQuantity(Enum):
	NONE = 0
	LOW = 1
	MEDIUM = 2
	HIGH = 3

def tissue_quantity(tissue_percentage):
	"""
	Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.

	Args:
		tissue_percentage: The tile tissue percentage.

	Returns:
		TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
	"""
	if tissue_percentage >= TISSUE_HIGH_THRESH:
		return(TissueQuantity.HIGH)
	elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
		return(TissueQuantity.MEDIUM)
	elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
		return(TissueQuantity.LOW)
	else:
		return(TissueQuantity.NONE)


def tissue_quantity_factor(amount):
	"""
	Obtain a scoring factor based on the quantity of tissue in a tile.

	Args:
		amount: Tissue amount as a TissueQuantity enum value.

	Returns:
		Scoring factor based on the tile tissue quantity.
	"""
	if amount == TissueQuantity.HIGH:
		quantity_factor = 1.0
	elif amount == TissueQuantity.MEDIUM:
		quantity_factor = 0.2
	elif amount == TissueQuantity.LOW:
		quantity_factor = 0.1
	else:
		quantity_factor = 0.0
	return(quantity_factor)





def filter_hsv_to_h(hsv, output_type="int", display_np_info=True):
	"""
	Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
	values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
	https://en.wikipedia.org/wiki/HSL_and_HSV

	Args:
		hsv: HSV image as a NumPy array.
		output_type: Type of array to return (float or int).
		display_np_info: If True, display NumPy array info and filter time.

	Returns:
		Hue values (float or int) as a 1-dimensional NumPy array.
	"""
	h = hsv[:, :, 0]
	h = h.flatten()
	if output_type == "int":
		h *= 360
		h = h.astype("int")
	return(h)

def rgb_to_hues(rgb):
	"""
	Convert RGB NumPy array to 1-dimensional array of hue values (HSV H values in degrees).

	Args:
		rgb: RGB image as a NumPy array

	Returns:
		1-dimensional array of hue values in degrees
	"""
	hsv = sk_color.rgb2hsv(rgb)
	h = filter_hsv_to_h(hsv, display_np_info=False)
	return(h)

def hsv_purple_pink_factor(rgb):
	"""
	Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
	average is purple versus pink.

	Args:
		rgb: Image an NumPy array.

	Returns:
		Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
	"""
	hues = rgb_to_hues(rgb)
	hues = hues[hues >= 260]  # exclude hues under 260
	hues = hues[hues <= 340]  # exclude hues over 340
	if len(hues) == 0:
		return(0)  # if no hues between 260 and 340, then not purple or pink

	pu_dev = np.sqrt(np.mean(np.abs(hues - HSV_PURPLE) ** 2))
	pi_dev = np.sqrt(np.mean(np.abs(hues - HSV_PINK) ** 2))
	avg_factor = (340 - np.average(hues)) ** 2

	if pu_dev == 0:  # avoid divide by zero if tile has no tissue
		return(0)

	factor = pi_dev / pu_dev * avg_factor
	return(factor)


def hsv_saturation_and_value_factor(rgb):
	"""
	Function to reduce scores of tiles with narrow HSV saturations and values since saturation and value standard
	deviations should be relatively broad if the tile contains significant tissue.

	Example of a blurred tile that should not be ranked as a top tile:
		../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png

	Args:
		rgb: RGB image as a NumPy array

	Returns:
		Saturation and value factor, where 1 is no effect and less than 1 means the standard deviations of saturation and
		value are relatively small.
	"""

	hsv = sk_color.rgb2hsv(rgb)
	s = hsv[:, :, 1]
	s = s.flatten()
	v = hsv[:, :, 2]
	v = v.flatten()

	s_std = np.std(s)
	v_std = np.std(v)
	if s_std < 0.05 and v_std < 0.05:
		factor = 0.4
	elif s_std < 0.05:
		factor = 0.7
	elif v_std < 0.05:
		factor = 0.7
	else:
		factor = 1

	factor = factor ** 2
	return(factor)


def score_tile(np_tile, tissue_percent, slide_num, row, col):
	"""
	Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.

	Args:
		np_tile: Tile as NumPy array.
		tissue_percent: The percentage of the tile judged to be tissue.
		slide_num: Slide number.
		row: Tile row.
		col: Tile column.

	Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
	"""
	color_factor = hsv_purple_pink_factor(np_tile)
	s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
	amount = tissue_quantity(tissue_percent)
	quantity_factor = tissue_quantity_factor(amount)
	combined_factor = color_factor * s_and_v_factor * quantity_factor
	score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0
	# scale score to between 0 and 1
	score = 1.0 - (10.0 / (10.0 + score))
	return(score, color_factor, s_and_v_factor, quantity_factor)


def score_tiles(slide_num, np_img=None, dimensions=None, small_tile_in_tile=False):
	"""
	Score all tiles for a slide and return the results in a TileSummary object.

	Args:
		slide_num: The slide number.
		np_img: Optional image as a NumPy array.
		dimensions: Optional tuple consisting of (original width, original height, new width, new height). Used for dynamic
			tile retrieval.
		small_tile_in_tile: If True, include the small NumPy image in the Tile objects.

	Returns:
		TileSummary object which includes a list of Tile objects containing information about each tile.
	"""

	objective_powerInfo = pd.read_csv(BASE_DIR+"objectiveInfo.txt", header=None, names=['objective_power'], index_col=0, sep="\t")
	objective_power = int(objective_powerInfo.loc[slide_num, 'objective_power'])
	if objective_power == 40:
		if RESOLUTION == '5x':
			stepSize = 8
		elif RESOLUTION == '20x':
			stepSize = 2
		elif RESOLUTION == '2.5x':
			stepSize = 16
	elif objective_power == 20:
		if RESOLUTION == '5x':
			stepSize = 4
		elif RESOLUTION == '2.5x':
			stepSize = 8
		else:
			stepSize = 1
	else:
		if RESOLUTION == '5x':
			stepSize = 2
		elif RESOLUTION == '2.5x':
			stepSize = 4
		elif RESOLUTION == '20x':
			print(RESOLUTION + " is not supported at 10x magnification")
			sys.exit()
		else:
			stepSize = 1


	if dimensions is None:
		img_path = get_filter_image_result(slide_num)
		o_w, o_h, w, h = util.parseDimensionsFromImageFilename(img_path)
	else:
		o_w, o_h, w, h = dimensions

	
	if np_img is None:
		np_img = slide.open_image_np(img_path)


	row_tile_size = round(ROW_TILE_SIZE / SCALE_FACTOR)
	col_tile_size = round(COL_TILE_SIZE / SCALE_FACTOR)
	

	slidePath = util.get_training_slide_path(SRC_TRAIN_DIR, slide_num)

	num_row_tiles, num_col_tiles = util.get_num_tiles(h, w, row_tile_size, col_tile_size)

	tile_sum = TileSummary(slide_num=slide_num,
												 orig_w=o_w,
												 orig_h=o_h,
												 orig_tile_w=COL_TILE_SIZE,
												 orig_tile_h=ROW_TILE_SIZE,
												 scaled_w=w,
												 scaled_h=h,
												 scaled_tile_w=col_tile_size,
												 scaled_tile_h=row_tile_size,
												 tissue_percentage=util.tissue_percent(np_img),
												 num_col_tiles=num_col_tiles,
												 num_row_tiles=num_row_tiles)

	count = 0
	high = 0
	medium = 0
	low = 0
	none = 0

	tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size, stepSize)
	for t in tile_indices:
		count += 1  # tile_num
		r_s, r_e, c_s, c_e, r, c = t

		np_tile = np_img[r_s:r_e+((r_e-r_s)*(stepSize-1)), c_s:c_e+((c_e-c_s)*(stepSize-1))]

		t_p = util.tissue_percent(np_tile)
		amount = tissue_quantity(t_p)
		if amount == TissueQuantity.HIGH:
			high += 1
		elif amount == TissueQuantity.MEDIUM:
			medium += 1
		elif amount == TissueQuantity.LOW:
			low += 1
		elif amount == TissueQuantity.NONE:
			none += 1

		o_c_s, o_r_s = small_to_large_mapping((c_s, r_s), (o_w, o_h))
		o_c_e, o_r_e = small_to_large_mapping((c_e, r_e), (o_w, o_h))

		# pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
		if (o_c_e - o_c_s) > COL_TILE_SIZE:
			o_c_e -= 1
		if (o_r_e - o_r_s) > ROW_TILE_SIZE:
			o_r_e -= 1

		score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, slide_num, r, c)

		np_scaled_tile = np_tile if small_tile_in_tile else None
		tile = Tile(tile_sum, slide_num, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
								o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
		tile_sum.tiles.append(tile)

	tile_sum.count = count
	tile_sum.high = high
	tile_sum.medium = medium
	tile_sum.low = low
	tile_sum.none = none

	tiles_by_score = tile_sum.tiles_by_score()
	rank = 0
	for t in tiles_by_score:
		rank += 1
		t.rank = rank

	return(tile_sum)



def training_slide_range_to_images(start_ind, end_ind, train_images):
	"""
	Convert a range of WSI training slides to smaller images (in a format such as jpg or png).

	Args:
		start_ind: Starting index (inclusive).
		end_ind: Ending index (inclusive).

	Returns:
		The starting index and the ending index of the slides that were converted.
	"""
	count = 0
	for slide_num in range(start_ind, end_ind + 1):
		try:
			training_slide_to_image(slide_num, train_images[count])
			count += 1
		except:
			print(str(slide_num).zfill(3), file=skippedSamps, flush=True)
	return (start_ind, end_ind)


def apply_filters_to_image_range(start_ind, end_ind, save):
	"""
	Apply filters to a range of images.

	Args:
		start_ind: Starting index (inclusive).
		end_ind: Ending index (inclusive).
		save: If True, save filtered images.

	Returns:
		Tuple consisting of 1) staring index of slides converted to images, 2) ending index of slides converted to images,
		and 3) a dictionary of image filter information.
	"""
	html_page_info = dict()
	for slide_num in range(start_ind, end_ind + 1):
		try:
			_, info = apply_filters_to_image(slide_num, save=save)
			html_page_info.update(info)
		except:
			print(str(slide_num).zfill(3), file=skippedSamps, flush=True)
		
	return(start_ind, end_ind, html_page_info)



def save_display_tile(tile,imageFile, save=True):
	"""
	Save and/or display a tile image.

	Args:
		tile: Tile object.
		save: If True, save tile image.
	"""
	tile_pil_img = tile_to_pil_tile(tile, imageFile)

	if save:
		img_path = get_tile_image_path(tile)
		dir = os.path.dirname(img_path)
		if not os.path.exists(dir):
			os.makedirs(dir)
		tile_pil_img.save(img_path)



def tile_to_pil_tile(tile, imageSlide):
	"""
	Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.

	Args:
		tile: Tile object.

	Return:
		Tile as a PIL image.
	"""
	t = tile
	slide_filepath = util.get_training_slide_path(SRC_TRAIN_DIR, imageSlide)
	s = util.openSlide(slide_filepath)
	x, y = t.o_c_s, t.o_r_s
	w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
	objective_powerInfo = pd.read_csv(BASE_DIR+"objectiveInfo.txt", header=None, names=['objective_power'], index_col=0, sep="\t")
	objective_power = int(objective_powerInfo.loc[t.slide_num, 'objective_power'])

	if objective_power == 40:
		if RESOLUTION == '5x':
			w = w*8
			h = h*8
		elif RESOLUTION == '2.5x':
			w = w*16
			h = h*16				
		else:
			w = w*2
			h = h*2
	elif objective_power ==20:
		if RESOLUTION == '5x':
			w = w*4
			h = h*4		
		elif RESOLUTION == '2.5x':
			w = w*8
			h = h*8	
	else:
		if RESOLUTION == '5x':
			w = w*2
			h = h*2		
		elif RESOLUTION == '2.5x':
			w = w*4
			h = h*4		

	level = 0
	tile_region = s.read_region((x, y), level, (w, h))
	tile_region = tile_region.resize((256,256),Image.BILINEAR)
	# RGBA to RGB
	pil_img = tile_region.convert("RGB")
	return(pil_img)

def summary_and_tiles(slide_num, imageSlide, save_top_tiles, save_data):
	"""
	Generate tile summary and top tiles for slide.

	Args:
		slide_num: The slide number.
		save_top_tiles: If True, save top tiles to files.

	"""
	try:
		img_path = get_filter_image_result(slide_num)
	except:
		return(None)
	np_img = util.open_image_np(img_path)

	tile_sum = score_tiles(slide_num, np_img)

	if save_data:
		save_tile_data(tile_sum)

	if save_top_tiles:
		for tile in tile_sum.top_tiles():
			tile.save_tile(imageSlide)
	return(tile_sum)


def get_tile_data_filename(slide_number):
	"""
	Convert slide number to a tile data file name.

	Example:
		5 -> TUPAC-TR-005-32x-49920x108288-1560x3384-tile_data.csv

	Args:
		slide_number: The slide number.

	Returns:
		The tile data file name.
	"""
	padded_sl_num = str(slide_number).zfill(3)

	training_img_path = util.getTrainingImagePath(slide_number, DEST_TRAIN_DIR, TRAIN_PREFIX, DEST_TRAIN_EXT, SCALE_FACTOR)
	large_w, large_h, small_w, small_h = util.parseDimensionsFromImageFilename(training_img_path)
	data_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
		large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_DATA_SUFFIX + ".csv"

	return data_filename


def save_tile_data(tile_summary):
	"""
	Save tile data to csv file.

	Args
		tile_summary: TimeSummary object.
	"""

	# time = Time()

	csv = summary_title(tile_summary) + "\n" + summary_stats(tile_summary)

	csv += "\n\n\nTile Num,Row,Column,Tissue %,Tissue Quantity,Col Start,Row Start,Col End,Row End,Col Size,Row Size," + \
				 "Original Col Start,Original Row Start,Original Col End,Original Row End,Original Col Size,Original Row Size," + \
				 "Color Factor,S and V Factor,Quantity Factor,Score\n"

	for t in tile_summary.tiles:
		line = "%d,%d,%d,%4.2f,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%4.0f,%4.2f,%4.2f,%0.4f\n" % (
			t.tile_num, t.r, t.c, t.tissue_percentage, t.tissue_quantity().name, t.c_s, t.r_s, t.c_e, t.r_e, t.c_e - t.c_s,
			t.r_e - t.r_s, t.o_c_s, t.o_r_s, t.o_c_e, t.o_r_e, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s, t.color_factor,
			t.s_and_v_factor, t.quantity_factor, t.score)
		csv += line

	# data_path = slide.get_tile_data_path(tile_summary.slide_num)
	TILE_DATA_DIR = os.path.join(BASE_DIR, "tile_data_" + RESOLUTION + "_overlap" + str(OVERLAP).replace(".", ""))
	if not os.path.exists(TILE_DATA_DIR):
		os.makedirs(TILE_DATA_DIR)
	data_path = os.path.join(TILE_DATA_DIR, get_tile_data_filename(tile_summary.slide_num))

	csv_file = open(data_path, "w")
	csv_file.write(csv)
	csv_file.close()

	# print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Data", str(time.elapsed()), data_path))


def image_range_to_tiles(start_ind, end_ind, slides, save_top_tiles, save_data):
	"""
	Generate tile summaries and tiles for a range of images.

	Args:
		start_ind: Starting index (inclusive).
		end_ind: Ending index (inclusive).
		save_top_tiles: If True, save top tiles to files.
	"""
	image_num_list = list()
	tile_summaries_dict = dict()
	count = 0
	finished = 0
	for slide_num in range(start_ind, end_ind + 1):
		imageSlide = slides[count]


		if os.path.exists(os.path.join(TILE_DIR, str(slide_num).zfill(3))):

			continue
		tile_summary = summary_and_tiles(slide_num, imageSlide, save_top_tiles, save_data)
		if tile_summary == None:
			continue
		image_num_list.append(slide_num)
		tile_summaries_dict[slide_num] = tile_summary
		count += 1
	return(image_num_list, tile_summaries_dict)



def multiprocess_training_slides_to_images(numProcessors=None):
	"""
	Convert all WSI training slides to smaller images using multiple processes (one process per core).
	Each process will process a range of slide numbers.
	"""

	with open(BASE_DIR + "objectiveInfo.txt", "w") as out:
		pass

	# how many processes to use
	if numProcessors:
		num_processes = numProcessors
	else:
		num_processes = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(num_processes)

	num_train_images, train_images = util.getNumberOfSlides(SRC_TRAIN_DIR, SRC_TRAIN_EXT)


	if num_processes > num_train_images:
		num_processes = num_train_images
	images_per_process = num_train_images / num_processes


	util.writeSlideNumberSampleNameToFile(BASE_DIR, num_train_images, train_images)

	# # each task specifies a range of slides
	tasks = []
	for num_process in range(1, num_processes + 1):
		start_index = (num_process - 1) * images_per_process + 1
		end_index = num_process * images_per_process
		start_index = int(start_index)
		end_index = int(end_index)
		images_task = [train_images[x] for x in range(start_index-1, end_index, 1)]
		tasks.append((start_index, end_index, images_task))

	# start tasks
	results = []
	for t in tasks:
		results.append(pool.apply_async(training_slide_range_to_images, t))

	for result in results:
		result.wait()
		if not result.successful():
			result.get()


def multiprocess_apply_filters_to_images(save=True, maxProcessors=None):
	"""
	Apply a set of filters to all training images using multiple processes (one process per core).

	Args:
		save: If True, save filtered images.
	"""

	if save and not os.path.exists(FILTER_DIR):
		os.makedirs(FILTER_DIR)

	# how many processes to use
	if maxProcessors:
		num_processes = maxProcessors
	else:
		num_processes = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(num_processes)


	num_train_images, slide_labels = util.getNumberOfSlides(SRC_TRAIN_DIR, SRC_TRAIN_EXT)

	if num_processes > num_train_images:
		num_processes = num_train_images
	images_per_process = num_train_images / num_processes


	tasks = []
	for num_process in range(1, num_processes + 1):
		start_index = (num_process - 1) * images_per_process + 1
		end_index = num_process * images_per_process
		start_index = int(start_index)
		end_index = int(end_index)
		tasks.append((start_index, end_index, save))

	# start tasks
	results = []
	for t in tasks:
		results.append(pool.apply_async(apply_filters_to_image_range, t))


	for result in results:
		result.wait()
		if not result.successful():
			result.get()




def multiprocess_filtered_images_to_tiles(save_top_tiles, save_data, maxProcessors=None):
	"""
	Generate tile summaries and tiles for all training images using multiple processes (one process per core).

	Args:
		save_top_tiles: If True, save top tiles to files.
	"""

	# how many processes to use
	if maxProcessors:
		num_processes = maxProcessors
	else:
		num_processes = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(num_processes)


	num_train_images, slide_labels = util.getNumberOfSlides(SRC_TRAIN_DIR, SRC_TRAIN_EXT)

	if num_processes > num_train_images:
		num_processes = num_train_images
	images_per_process = num_train_images / num_processes


	tasks = []
	for num_process in range(1, num_processes + 1):
		start_index = (num_process - 1) * images_per_process + 1
		end_index = num_process * images_per_process
		start_index = int(start_index)
		end_index = int(end_index)
		subImages = [slide_labels[x] for x in range(start_index-1,end_index, 1)]
		tasks.append((start_index, end_index, subImages, save_top_tiles, save_data))

	# start tasks
	results = []
	for t in tasks:
		results.append(pool.apply_async(image_range_to_tiles, t))

	for result in results:
		result.wait()
		if not result.successful():
			result.get()

def summary_title(tile_summary):
	"""
	Obtain tile summary title.

	Args:
		tile_summary: TileSummary object.

	Returns:
		 The tile summary title.
	"""
	return "Slide %03d Tile Summary:" % tile_summary.slide_num



class TileSummary:
	"""
	Class for tile summary information.
	"""

	slide_num = None
	orig_w = None
	orig_h = None
	orig_tile_w = None
	orig_tile_h = None
	scale_factor = SCALE_FACTOR
	scaled_w = None
	scaled_h = None
	scaled_tile_w = None
	scaled_tile_h = None
	mask_percentage = None
	num_row_tiles = None
	num_col_tiles = None

	count = 0
	high = 0
	medium = 0
	low = 0
	none = 0

	def __init__(self, slide_num, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h, scaled_tile_w,
							 scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
		self.slide_num = slide_num
		self.orig_w = orig_w
		self.orig_h = orig_h
		self.orig_tile_w = orig_tile_w
		self.orig_tile_h = orig_tile_h
		self.scaled_w = scaled_w
		self.scaled_h = scaled_h
		self.scaled_tile_w = scaled_tile_w
		self.scaled_tile_h = scaled_tile_h
		self.tissue_percentage = tissue_percentage
		self.num_col_tiles = num_col_tiles
		self.num_row_tiles = num_row_tiles
		self.tiles = []

	def __str__(self):
		return(summary_title(self) + "\n" + summary_stats(self))

	def mask_percentage(self):
		"""
		Obtain the percentage of the slide that is masked.

		Returns:
			 The amount of the slide that is masked as a percentage.
		"""
		return(100 - self.tissue_percentage)

	def tiles_by_tissue_percentage(self):
		"""
		Retrieve the tiles ranked by tissue percentage.

		Returns:
			 List of the tiles ranked by tissue percentage.
		"""
		sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
		return(sorted_list)

	def tiles_by_score(self):
		"""
		Retrieve the tiles ranked by score.

		Returns:
			 List of the tiles ranked by score.
		"""
		sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
		return(sorted_list)

	def top_tiles(self):
		"""
		Retrieve the top-scoring tiles.

		Returns:
			 List of the top-scoring tiles.
		"""
		sorted_tiles = self.tiles_by_tissue_percentage()
		top_tiles = [x for x in sorted_tiles if x.tissue_percentage > TISSUE_HIGH_THRESH]
		random.shuffle(top_tiles)
		top_tiles = top_tiles[:NUM_TOP_TILES]
		return(top_tiles)



class Tile:
	"""
	Class for information about a tile.
	"""

	def __init__(self, tile_summary, slide_num, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
							 o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
		self.tile_summary = tile_summary
		self.slide_num = slide_num
		self.np_scaled_tile = np_scaled_tile
		self.tile_num = tile_num
		self.r = r
		self.c = c
		self.r_s = r_s
		self.r_e = r_e
		self.c_s = c_s
		self.c_e = c_e
		self.o_r_s = o_r_s
		self.o_r_e = o_r_e
		self.o_c_s = o_c_s
		self.o_c_e = o_c_e
		self.tissue_percentage = t_p
		self.color_factor = color_factor
		self.s_and_v_factor = s_and_v_factor
		self.quantity_factor = quantity_factor
		self.score = score

	def __str__(self):
		return("[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
			self.tile_num, self.r, self.c, self.tissue_percentage, self.score))

	def __repr__(self):
		return("\n" + self.__str__())

	def mask_percentage(self):
		return(100 - self.tissue_percentage)

	def tissue_quantity(self):
		return(tissue_quantity(self.tissue_percentage))

	def save_tile(self, imageSlide):
		save_display_tile(self, imageSlide, save=True)


def preprocess_images (project, projectPath, max_cpu, save_top_tiles=True, save_data=False, overlap=0):
	global BASE_DIR, PROJECT, skippedSamps, TRAIN_PREFIX, SRC_TRAIN_DIR, SRC_TRAIN_EXT, DEST_TRAIN_EXT, SCALE_FACTOR, DEST_TRAIN_DIR, TISSUE_HIGH_THRESH, TISSUE_LOW_THRESH, RESOLUTION, \
			ROW_TILE_SIZE, COL_TILE_SIZE, NUM_TOP_TILES, HSV_PURPLE, HSV_PINK, FILTER_RESULT_TEXT, FILTER_DIR, TOP_TILES_SUFFIX, TOP_TILES_DIR, TILE_DIR, TILE_SUFFIX, TILE_DATA_SUFFIX, \
			OVERLAP


	BASE_DIR = projectPath
	PROJECT = project
	SKIPPED_SAMPLES = os.path.join(BASE_DIR, "skipped_samples.txt")
	OVERLAP = overlap

	# PROJECT = "BRCA"
	TRAIN_PREFIX = PROJECT + "-"
	SRC_TRAIN_DIR = os.path.join(BASE_DIR, PROJECT + "/")
	SRC_TRAIN_EXT = "svs"
	# SRC_TRAIN_EXT = "ndpi"
	#SRC_TRAIN_EXT = "jpg"
	DEST_TRAIN_EXT = "png"
	SCALE_FACTOR = 32
	DEST_TRAIN_DIR = os.path.join(BASE_DIR, PROJECT + "_" + DEST_TRAIN_EXT)
	TISSUE_HIGH_THRESH = 80
	TISSUE_LOW_THRESH = 10
	RESOLUTION = '5x'
	ROW_TILE_SIZE = 256
	COL_TILE_SIZE = 256
	#ROW_TILE_SIZE = 224
	#COL_TILE_SIZE = 224
	NUM_TOP_TILES = 100000

	HSV_PURPLE = 270
	HSV_PINK = 330

	FILTER_RESULT_TEXT = "filtered"
	FILTER_DIR = os.path.join(BASE_DIR, "filter_" + DEST_TRAIN_EXT)

	TOP_TILES_SUFFIX = "top_tile_summary"
	TOP_TILES_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_" + DEST_TRAIN_EXT)
	TILE_DIR = os.path.join(BASE_DIR, "tiles_" + DEST_TRAIN_EXT)
	TILE_SUFFIX = "tile"
	TILE_DATA_SUFFIX = "tile_data_" + RESOLUTION + "_overlap" + str(OVERLAP).replace(".", "")


	if max_cpu:
		maxProcessors = max_cpu
	else:
		maxProcessors = multiprocessing.cpu_count()

	skippedSamps = open(SKIPPED_SAMPLES, "w")
	multiprocess_training_slides_to_images(maxProcessors)
	multiprocess_apply_filters_to_images(save=True, maxProcessors=maxProcessors)
	multiprocess_filtered_images_to_tiles(save_top_tiles, save_data, maxProcessors=maxProcessors)
	skippedSamps.close()



# if __name__ == "__main__":
# 	skippedSamps = open(SKIPPED_SAMPLES, "w")
# 	multiprocess_training_slides_to_images(32)
# 	multiprocess_apply_filters_to_images(save=True, maxProcessors=4)
# 	multiprocess_filtered_images_to_tiles(save_top_tiles=True, maxProcessors=1)
# 	skippedSamps.close()
