from typing import Dict, Any
import numpy as np
import utils

NDArray = Any




def update_current_image(current_image, seam_indices, current_width):
    """
    mapping_matrix - current mapping matrix
    seam_indices - a list of columns indices with len=n
    current_shape - of the edited image
    """
    if seam_indices is not None:
        m = current_width
        for i, j in enumerate(seam_indices):
            current_image[i][int(j):-1] = current_image[i][int(j) + 1:]
        return current_image[:, :m - 1]
    else:
        return current_image



def cal_cost_c(greyscale_image):
    """
    gets a greyscale image, returns the cv-r-l matrices* using whole-matrix calculations only.
    """
    n, m = greyscale_image.shape

    zero_column = np.broadcast_to([0.], [n, 1])
    left_shifted = np.concatenate([zero_column, greyscale_image[:, 0:-1]], axis=1)
    right_shifted = np.concatenate([greyscale_image[:, 1:], zero_column], axis=1)
    new_row = np.zeros(m)
    down_shifted = np.vstack([new_row, greyscale_image])
    down_shifted = down_shifted[0:n]

    # edges:
    left_shifted[:0] = 255.0 + right_shifted[:1]

    # greyscale_image[i, j+1] - greyscale_image[i, j-1]
    abs1_mat = np.abs(right_shifted - left_shifted)
    # np.abs(greyscale_image[i-1, j] - greyscale_image[i, j-1])
    abs2_mat = np.abs(down_shifted - left_shifted)
    # np.abs(greyscale_image[i-1, j] - greyscale_image[i, j+1])
    abs3_mat = np.abs(down_shifted - right_shifted)

    cl = abs1_mat + abs2_mat
    cl[:, 0] = 255.0
    cl[:, m - 1] = 255.0 + abs2_mat[:, m - 1]

    cv = abs1_mat
    cv[:, 0] = 255.0
    cv[:, m - 1] = 255.0

    cr = abs1_mat + abs3_mat
    cr[:, m - 1] = 255.0
    cr[:, 0] = 255.0 + abs3_mat[:, 0]

    cl[0] = np.zeros(m)
    cr[0] = np.zeros(m)
    cv[0] = np.zeros(m)

    return cl, cv, cr


def create_cost_matrix_forward(image_shape, greyscale_image,gradient_matrix):
    """
    gets an image, and calculate its cost matrix
    """

    n, m = image_shape
    cl_mat, cv_mat, cr_mat = cal_cost_c(greyscale_image)
    cost_matrix_cal = np.zeros((n, m))
    cost_matrix_cal[0] = gradient_matrix[0]

    for i in range(1, n):
        zero_single = np.broadcast_to([0.], [1])
        """ shifting the row above (i-1): """
        cost_left_shifted = np.concatenate([zero_single, cost_matrix_cal[i - 1, 0:-1]], axis=0)
        # in left, place j is actually place j-1: in cell a[j] there is now a[j-1]
        cost_right_shifted = np.concatenate([cost_matrix_cal[i - 1, 1:], zero_single], axis=0)
        # in left, place j is actually place j+1: in cell a[j] there is now a[j+1]

        """ calculating the current row (i): """
        for j in range(0, m):
            if j == 0:
                cost_matrix_cal[i][j] = min(cost_right_shifted[j] + cr_mat[i, j],
                                            cost_matrix_cal[i - 1, j] + cv_mat[i, j])
            elif j == m - 1:
                cost_matrix_cal[i][j] = min(cost_left_shifted[j] + cl_mat[i, j],
                                            cost_matrix_cal[i - 1, j] + cv_mat[i, j])
            else:
                cost_matrix_cal[i][j] = min(cost_left_shifted[j] + cl_mat[i, j],
                                            cost_matrix_cal[i - 1, j] + cv_mat[i, j],
                                            cost_right_shifted[j] + cr_mat[i, j])

        cost_matrix_cal[i] += gradient_matrix[i]

    return cost_matrix_cal,  cl_mat, cv_mat, cr_mat


def create_cost_matrix_no_forward(image_shape, greyscale_image,gradient_matrix):
    """
    gets an image, and calculate its cost matrix
    """

    n, m = image_shape
    cost_matrix_cal = np.zeros((n, m))
    cost_matrix_cal[0] = gradient_matrix[0]

    for i in range(1, n):
        zero_single = np.broadcast_to([0.], [1])
        """ shifting the row above (i-1): """
        "0,x,x,x,x  y"
        cost_left_shifted = np.concatenate([zero_single, cost_matrix_cal[i - 1, 0:-1]], axis=0)
        # in left, place j is actually place j-1: in cell a[j] there is now a[j-1]
        "y  x,x,x,x,0"
        cost_right_shifted = np.concatenate([cost_matrix_cal[i - 1, 1:], zero_single], axis=0)
        # in left, place j is actually place j+1: in cell a[j] there is now a[j+1]

        """ calculating the current row (i): """
        for j in range(0, m):
            if j == 0:
                cost_matrix_cal[i][j] = min(cost_right_shifted[j],
                                            cost_matrix_cal[i - 1, j])
            elif j == m - 1:
                cost_matrix_cal[i][j] = min(cost_left_shifted[j],
                                            cost_matrix_cal[i - 1, j])
            else:
                cost_matrix_cal[i][j] = min(cost_left_shifted[j],
                                            cost_matrix_cal[i - 1, j],
                                            cost_right_shifted[j])

        cost_matrix_cal[i] += gradient_matrix[i]

    return cost_matrix_cal






def find_optimal_seam(greyscale_image, seam_update,gradient_matrix):
    """ gets the greyscale image and the current mapping, and returns the best seam. """
    cur_image = update_current_image(greyscale_image, seam_update, greyscale_image.shape[1])
    gradient_matrix = update_current_image(gradient_matrix, seam_update, gradient_matrix.shape[1])
    in_height, in_width = cur_image.shape
    cost_mat, cl, cv, cr = create_cost_matrix_forward(cur_image.shape, cur_image,gradient_matrix)

    best_seam = np.zeros(in_height)  # to keep best match in each row. in the end- list of columns.
    for row in range(in_height - 1, -1, -1):
        if row == in_height - 1:
            best_seam[in_height - 1] = np.argmin(cost_mat[in_height - 1])
        else:
            last_col = int(best_seam[row + 1])
            curr_rounded_value = np.around(cost_mat[row + 1, last_col], 6)
            if curr_rounded_value == np.around(gradient_matrix[row + 1, last_col] +
                                               cost_mat[row, last_col] +
                                               cv[row + 1, last_col], 6):
                min_val = last_col
            elif curr_rounded_value == np.around(gradient_matrix[row + 1, last_col] +
                                                 cost_mat[row, last_col - 1] +
                                                 cl[row + 1, last_col], 6):
                min_val = last_col - 1
            else:
                min_val = last_col + 1
            best_seam[row] = min_val

    return best_seam,cur_image,gradient_matrix


def find_optimal_seam_no_forward(greyscale_image, seam_update,gradient_matrix):
    """ gets the gretscale image and the current mapping, and returns the best seam, without using forward energy"""
    cur_image = update_current_image(greyscale_image, seam_update, greyscale_image.shape[1])
    gradient_matrix = update_current_image(gradient_matrix, seam_update, gradient_matrix.shape[1])
    in_height, in_width = cur_image.shape
    cost_mat = create_cost_matrix_no_forward(cur_image.shape, cur_image,gradient_matrix)
    best_seam = np.zeros(in_height)
    for row in range(in_height - 1, -1, -1):
        if row == in_height - 1:
            best_seam[in_height - 1] = np.argmin(cost_mat[in_height - 1])
        else:
            last_col = int(best_seam[row + 1])
            curr_rounded_value = np.around(cost_mat[row + 1, last_col], 6)
            if curr_rounded_value == np.around(gradient_matrix[row + 1, last_col] + cost_mat[row, last_col], 6):
                min_val = last_col
            elif curr_rounded_value == np.around(gradient_matrix[row + 1, last_col] + cost_mat[row, last_col - 1], 6):
                min_val = last_col - 1
            else:
                min_val = last_col + 1
            best_seam[row] = min_val

    return best_seam, cur_image,gradient_matrix
