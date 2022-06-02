from typing import Dict, Any
import numpy as np
import utils
import cost_matrix
NDArray = Any


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    image_copy = np.copy(image)
    new_image, temp_image, vertical_seams,horizontal_seams = handle_height_width(image_copy, out_height, out_width, forward_implementation)
    if vertical_seams is None:
        vertical_to_show = image
    else:
        vertical_to_show = change_vertical_to_red(image, vertical_seams)
    if horizontal_seams is None:
        horizon_to_show = image
    elif horizontal_seams is not None and vertical_seams is None:
        horizon_to_show = change_horizontal_to_black(image, horizontal_seams)
    else:
        horizon_to_show = change_horizontal_to_black(temp_image, horizontal_seams)

    return {'resized': new_image, 'vertical_seams': vertical_to_show, 'horizontal_seams': horizon_to_show}


def change_vertical_to_red(image, vertical_seams):
    """
    Adds red vertical seams on to the original image
    """
    vertical_seams_image = np.copy(image)
    n = len(vertical_seams[0])
    for i in range(n):
        for j, seam in enumerate(vertical_seams):
            pix = seam[i]
            vertical_seams_image[i][pix][0] = 255
            vertical_seams_image[i][pix][1] = 0
            vertical_seams_image[i][pix][2] = 0
    return vertical_seams_image


def change_horizontal_to_black(image, horizontal_seams):
    """
    Adds black horizontal seams to original image
    """
    horizontal_seams_image = np.copy(image)
    rot_image = np.rot90(horizontal_seams_image, k=1, axes=(0, 1))
    n = len(horizontal_seams[0])
    for i in range(n):
        for j, seam in enumerate(horizontal_seams):
            pix = seam[i]
            rot_image[i][pix][0] = 0
            rot_image[i][pix][1] = 0
            rot_image[i][pix][2] = 0
    horizontal_seams_image = np.rot90(rot_image, k=-1, axes=(0, 1))
    return horizontal_seams_image


def handle_height_width(image: NDArray, out_height: int, out_width: int, forward_implementation: bool):
    """
    Changes height and width of image, according to input parameters
    """
    greyscale_image = utils.to_grayscale(image)
    in_height, in_width = greyscale_image.shape
    horizontal_seams = None
    vertical_seams = None
    temp_image = np.zeros(image.shape)
    if in_width != out_width:
        k = in_width - out_width
        vertical_seams, mapping_matrix = reduce_width(greyscale_image, np.abs(k), forward_implementation)
        if k > 0:
            # reduce
            image = generate_image_from_map(mapping_matrix, image)
        else:
            # increase
            image = duplicate_seams_in_image(vertical_seams, image)
        temp_image = np.copy(image)
    if in_height != out_height:
        rot_image = np.rot90(np.copy(image), k=1, axes=(0, 1))
        greyscale_image = utils.to_grayscale(rot_image)
        k = in_height - out_height
        horizontal_seams, mapping_matrix = reduce_width(greyscale_image, np.abs(k), forward_implementation)
        if k > 0:
            # reduce
            image = generate_image_from_map(mapping_matrix, rot_image)
        else:
            # increase
            image = duplicate_seams_in_image(horizontal_seams, rot_image)
        image = np.rot90(image, k=-1, axes=(0, 1))
    return image, temp_image, vertical_seams, horizontal_seams


def reduce_width(greyscale_image: NDArray,k, forward_implementation: bool):
    """
    Decreases the width of an image with optimal seam algorithm
    """
    in_height, in_width = greyscale_image.shape
    vertical_seams = []
    mapping_matrix = create_mapping_matrix(greyscale_image.shape)
    optimal_seam_method = cost_matrix.find_optimal_seam if forward_implementation else cost_matrix.find_optimal_seam_no_forward
    optimal_seam = None
    curr_greyscale_image = greyscale_image
    grads = utils.get_gradients(curr_greyscale_image)
    for i in range(k):
        optimal_seam, curr_greyscale_image, grads = optimal_seam_method(curr_greyscale_image,optimal_seam, grads)
        vertical_seams.append(get_seam_image_indices(mapping_matrix, optimal_seam))
        mapping_matrix = update_mapping_matrix(mapping_matrix, optimal_seam, in_width-i)

    return vertical_seams, mapping_matrix


def create_mapping_matrix(image_shape):
    """
      creates mapping matrix for the input image
    """
    n, m = image_shape
    return np.array([[j for j in range(m)] for _ in range(n)])


def update_mapping_matrix(mapping_matrix, seam_indices,current_width):
    """
    mapping_matrix - current mapping matrix
    seam_indices - a list of columns indices with len=n
    current_shape - of the edited image
    """
    m = current_width # not really used - just in case we don't delete
    for i, j in enumerate(seam_indices):
        mapping_matrix[i][int(j):-1] = mapping_matrix[i][int(j)+1:]
    return mapping_matrix[:, :m-1]


def get_seam_image_indices(mapping_matrix: NDArray,seam):
    """
    mapping_matrix - current mapping matrix
    seam - seam indices from the last iteration, a list of columns indices with len=n
    after each iteration of seam finding, saves the original indices of the seam for later use
    """
    return [mapping_matrix[i][int(loc)] for i, loc in enumerate(seam)]


def generate_image_from_map(mapping_matrix: NDArray,image: NDArray):
    """
    mapping_matrix - current mapping matrix
    image - original RGB imgae
    current_shape - shape after removing seams
    """
    n, m = mapping_matrix.shape
    image_to_plot = np.zeros((n, m, 3))
    for i in range(n):
        for j in range(m):
            loc = mapping_matrix[i][j]
            image_to_plot[i][j] = image[i][loc]
    return image_to_plot


def duplicate_seams_in_image(seams,image):
    """
    After finding k optimal seams, duplicates them in the original image
    seams - list of k seam lists with original indices, obtained by get_seam_image_indices
    image - original RGB image
    """
    im_shape = image.shape
    n, m, _ = im_shape
    mapping_matrix = np.zeros((n, m))
    rows = [image[i] for i in range(n)]
    for i in range(n):
        for j, seam in enumerate(seams):
            dup = seam[i]
            mapping_matrix[i][dup] = 1
    for i in range(n):
        for j in range(m-1, -1, -1):
            if mapping_matrix[i][j] == 1:
                rows[i] = np.insert(rows[i], j, image[i][j], axis=0)

    new_image = np.stack(tuple(rows))
    return new_image



