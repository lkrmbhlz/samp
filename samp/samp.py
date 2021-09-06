import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt
from tqdm import tqdm


def pca_projections_2d(data_3d):
    projections_2d = []
    for object_3d in data_3d:
        pca = PCA(n_components=2)
        projections_2d.append(pca.fit_transform(object_3d))
    return projections_2d


def varimax_projections_2d(data_3d, get_1st_and_3rd_component=False):
    projections_2d = []
    for object_3d in data_3d:
        transformer = FactorAnalysis(n_components=3, random_state=0, rotation='varimax')
        result = transformer.fit_transform(object_3d)
        if get_1st_and_3rd_component:
            projections_2d.append(result[:, [0, 2]])
        else:
            projections_2d.append(result[:, [0, 1]])
    return projections_2d


def asymmetries_x_axis(projections_2d, title=None, draw=True, stepsize=2):
    asymmetries = []
    asymmetries_single_values = []
    tolerance = stepsize / 2

    if draw:
        fig, axs = plt.subplots(len(projections_2d), 2, figsize=(8, 4 * len(projections_2d)))

    print('Calculating asymmetries for %d objects' % len(projections_2d))

    for i, projection in enumerate(tqdm(projections_2d)):

        x_range_min, x_range_max = (min(projection[:, 0]), max(projection[:, 0]))

        if draw:
            axs[i][0].scatter(projections_2d[i][:, 0], projections_2d[i][:, 1], marker='.', alpha=0.005,
                              c='black')

        asymmetry = 0
        asymmetry_values = []

        for step in np.arange(x_range_min, x_range_max, stepsize):
            points_in_area = np.array(
                [[x, y] for x, y in projections_2d[i] if step - tolerance <= x < step + tolerance])

            if len(points_in_area) != 0:
                number_of_considered_values = 1

                if number_of_considered_values == 0:
                    maximum = max(points_in_area[:, 1])
                else:
                    maximum = np.mean(
                        points_in_area[np.argpartition(points_in_area[:, 1], -number_of_considered_values)[
                                       -number_of_considered_values:]][:, 1])

                if number_of_considered_values == 0:
                    minimum = min(points_in_area[:, 1])
                else:
                    minimum = np.mean(
                        points_in_area[np.argpartition(points_in_area[:, 1], number_of_considered_values)[
                                       :number_of_considered_values]][:, 1])

                if minimum == maximum:
                    asymmetry_value = maximum + minimum
                if np.sign(maximum) == 1 and np.sign(minimum) == -1:
                    asymmetry_value = abs(maximum + minimum)
                if np.sign(maximum) == 1 and np.sign(minimum) == 1:
                    asymmetry_value = maximum + minimum
                if np.sign(maximum) == -1 and np.sign(minimum) == -1:
                    asymmetry_value = abs(maximum) + abs(minimum)
                if minimum > maximum:
                    print('higher min than max --> choose smaller tolerance percentage')
                    asymmetry_value = 10

                asymmetry = asymmetry + asymmetry_value

                if draw:
                    axs[i][0].scatter(step, maximum, marker='.', color='green')
                    axs[i][0].scatter(step, minimum, marker='.', color='red')
                    axs[i][0].set_aspect('equal')

                asymmetry_values.append(asymmetry_value)
        if draw:
            axs[i][1].plot(asymmetry_values, marker='.')
            axs[i][1].set_aspect('auto')
        asymmetries_single_values.append(asymmetry_values)
        asymmetries.append(asymmetry)
    if draw:
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=0.97)
    return asymmetries, asymmetries_single_values


def asymmetries_y_axis(projections_2d, title=None, draw=True, stepsize=2):
    asymmetries_2nd_PC = []
    tolerance = stepsize / 2

    asymmetries_single_values = []

    if draw:
        fig, axs = plt.subplots(len(projections_2d), 2, figsize=(8, 4 * len(projections_2d)))

    print('Calculating asymmetries for %d objects' % len(projections_2d))

    for i, projection in enumerate(tqdm(projections_2d)):

        y_range_min, y_range_max = (min(projection[:, 1]), max(projection[:, 1]))

        if draw:
            axs[i][0].scatter(projections_2d[i][:, 0], projections_2d[i][:, 1], marker='.', alpha=0.01,
                              c='black')

        asymmetry_y = 0
        asymmetry_values_y = []

        for step in np.arange(y_range_min, y_range_max, stepsize):
            points_in_area = np.array(
                [[x, y] for x, y in projections_2d[i] if y > step - tolerance and y < step + tolerance])

            if len(points_in_area) != 0:
                maximum_y = max(points_in_area[:, 0])
                minimum_y = min(points_in_area[:, 0])

            if len(points_in_area) != 0:

                if minimum_y == maximum_y:
                    asymmetry_value_y = maximum_y + minimum_y
                if np.sign(maximum_y) == 1 and np.sign(minimum_y) == -1:
                    asymmetry_value_y = abs(maximum_y + minimum_y)
                if np.sign(maximum_y) == 1 and np.sign(minimum_y) == 1:
                    asymmetry_value_y = maximum_y + minimum_y
                if np.sign(maximum_y) == -1 and np.sign(minimum_y) == -1:
                    asymmetry_value_y = abs(maximum_y) + abs(minimum_y)
                asymmetry_y = asymmetry_y + asymmetry_value_y

                if draw: axs[i][0].scatter(maximum_y, step, marker='.', color='green')
                if draw: axs[i][0].scatter(minimum_y, step, marker='.', color='red')

                asymmetry_values_y.append(asymmetry_value_y)

        if draw and len(asymmetry_values_y) != 0:
            axs[i][1].plot(asymmetry_values_y, marker='.')
            axs[i][1].set_aspect('auto')
        asymmetries_2nd_PC.append(asymmetry_y)
        asymmetries_single_values.append(asymmetry_values_y)

    if draw:
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=0.97)

    return asymmetries_2nd_PC, asymmetries_single_values


def min_max_asymmetries(asymmetries_of_projections_1st_axis, asymmetries_of_projections_2nd_axis):
    min_asymmetry = [min(x, y) for x, y in
                                    zip(asymmetries_of_projections_1st_axis, asymmetries_of_projections_2nd_axis)]
    max_asymmetry = [max(x, y) for x, y in
                                    zip(asymmetries_of_projections_1st_axis, asymmetries_of_projections_2nd_axis)]

    return list(zip(min_asymmetry, max_asymmetry))


def get_min_max_varimax(pointclouds, get_1st_and_3rd_component=False):
    if get_1st_and_3rd_component:
        projections = varimax_projections_2d(pointclouds, get_1st_and_3rd_component=True)
    else:
        projections = varimax_projections_2d(pointclouds)
    asymmetries_x = asymmetries_x_axis(projections)
    asymmetries_y = asymmetries_y_axis(projections)
    return min_max_asymmetries(asymmetries_x, asymmetries_y)


def remove_outliers(data, max_number_std=2):
    mean = np.mean(data)
    std_dev = np.std(data)
    zero_based = abs(data - mean)
    return data[zero_based < max_number_std * std_dev]
