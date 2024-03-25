from PIL import Image, ImageFilter
import numpy as np
from multiprocessing import Pool, Array, Process, cpu_count, RawArray
import timer_wraper as tw
import os


def convert_to_black_and_white(img):
    # convert the image to grayscale
    img_gray = img.convert('L')

    # convert the grayscale image to a numpy array
    img_array = np.array(img_gray)

    # calculate the median intensity value to find approximately 50/50 split
    median_intensity = np.median(img_array)

    # convert the image to black and white using the median intensity value as threshold
    img_bw = img_gray.point(lambda x: 255 if x < median_intensity else 0).convert('1')

    return img_bw


def add_noise_to_image(img, img_bw):
    np_img = np.array(img)

    # count black pixels
    num_black_pixels = img_bw.getcolors()[0][0]

    # calculate the number of pixels to be changed
    num_noise_pixels = num_black_pixels // 10

    # generate random integers between 0 and the width of the image
    x_coords = np.random.randint(0, np_img.shape[1], num_noise_pixels)
    y_coords = np.random.randint(0, np_img.shape[0], num_noise_pixels)

    # iterate over pairs of x and y coordinates to change the color of the pixel at coordinates
    for x, y in zip(x_coords, y_coords):
        np_img[y, x] = 255 - np_img[y, x]

    # changing back to Image
    return Image.fromarray(np_img)


def process_image(filepath, filename):
    bw_output_path = 'Data/Output/images_bw'
    blur_output_path = 'Data/Output/images_blur'
    noise_output_path = 'Data/Output/images_noise'

    # original image
    img = Image.open(filepath)

    # convert to black and white
    img_bw = convert_to_black_and_white(img)
    img_bw.save(os.path.join(bw_output_path, filename))

    # apply blur
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=3))
    img_blurred.save(os.path.join(blur_output_path, filename))

    # add noise
    img_noisy = add_noise_to_image(img, img_bw)
    img_noisy.save(os.path.join(noise_output_path, filename))

    return


@tw.timeit
def sequential_application(image_dir):

    # iterate over each file in the directory
    for filename in os.listdir(image_dir):

        # extra check to see if the file is a JPEG image
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):

            # construct the full path to the image file
            filepath = os.path.join(image_dir, filename)

            # process and save result images
            process_image(filepath, filename)

    return


@tw.timeit
def parallelize_pool_application(image_dir, cpus):  # cpus=cpu_count()
    # list to store image paths
    image_paths = []

    # iterate through files in the directory
    for filename in os.listdir(image_dir):

        # extra check to see if the file is a JPEG image
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):

            # construct the full path to the image file
            filepath = os.path.join(image_dir, filename)

            # append the filepath to the list
            image_paths.append((filepath, filename))

    # initializes a pool of worker processes for parallel execution
    pool = Pool(cpus)

    print("CPU count: ", cpus)

    # maps the process_image function to each image
    # distributes the workload of processing images across the worker processes in the pool
    pool.starmap(process_image, [(image_paths[i][0], image_paths[i][1]) for i in range(len(image_paths))])

    # closes the pool of worker processes, indicating that no more tasks will be added to it
    pool.close()
    # blocks the main program's execution until all processes in the pool have completed their tasks
    pool.join()

    return


if __name__ == '__main__':
    # paths to the input directory and output directories
    input_directory = 'Data/Images'

    # Without parallelization
    print("Sequential application without parallelization:")
    sequential_application(input_directory)

    print("-----------------------------------------------")
    print("Parallelized application using Pool.apply():")
    parallelize_pool_application(input_directory, cpus=4)
    print("-----------------------------------------------")
    print("Parallelized application using Pool.apply():")
    parallelize_pool_application(input_directory, cpus=8)
    print("-----------------------------------------------")
    print("Parallelized application using Pool.apply():")
    parallelize_pool_application(input_directory, cpus=12)
    print("-----------------------------------------------")
    print("Parallelized application using Pool.apply():")
    parallelize_pool_application(input_directory, cpus=16)












