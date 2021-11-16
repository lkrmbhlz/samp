import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import distance
from sklearn import preprocessing


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


def asymmetries_x_axis(projections_2d, title=None, draw=False, n_segments=20,
                       considered_percentage=0.05, outlier_removal=False,
                       number_of_std_for_outlier_removal=3, normalize=False):
    if normalize:
        normalized_projections = []
        min_max_scaler = preprocessing.MinMaxScaler()
        for unnormalized_projection in projections_2d:
            normalized_projections.append(
                min_max_scaler.fit_transform(np.array(unnormalized_projection)))
    else: normalized_projections = projections_2d

    asymmetries = []
    asymmetries_single_values = []

    if draw:
        fig, axs = plt.subplots(len(projections_2d), 2, figsize=(8, 4 * len(projections_2d)))

    print('Calculating asymmetries for %d objects' % len(normalized_projections))

    for i, projection in enumerate(tqdm(normalized_projections)):

        #tolerance = stepsize / 2

        x_range_min, x_range_max = (min(projection[:, 0]), max(projection[:, 0]))

        if draw:
            axs[i][0].scatter(normalized_projections[i][:, 0], normalized_projections[i][:, 1], marker='.', alpha=0.02,
                              c='black')

        asymmetry = 0
        asymmetry_values = []

        segments = np.linspace(min(projection[:, 0]), max(projection[:, 0]), n_segments, endpoint=True)
        #segments = np.append(segments, (segments[-1]+abs(segments[0]-segments[1])))
        tolerance = abs(segments[0]-segments[1])/2

        #for step in np.arange(x_range_min, x_range_max, stepsize):
        for j, step in enumerate(segments[0:-1]):
            min_x_area = step
            max_x_area = segments[j+1]
            points_in_area = np.array(
                [[x, y] for x, y in normalized_projections[i] if min_x_area <= x < max_x_area])
            #points_in_area = np.array(
            #    [[x, y] for x, y in projections_2d[i] if step - tolerance <= x < step + tolerance])
            #axs[i][0].scatter(step - tolerance, 0.02, c='pink',  marker='.')
            #axs[i][0].scatter(step + tolerance, 0, c='brown', marker='.')
            if draw:
                axs[i][0].axvline(step, alpha=0.2)

            number_of_considered_values = int(round(len(points_in_area) * considered_percentage))
            #print('points in area: ', len(points_in_area))
            #print('considered:', number_of_considered_values)
            if len(points_in_area) != 0:
                #number_of_considered_values = 1

                if number_of_considered_values == 0:
                    maximum = max(points_in_area[:, 1])
                else:
                    #maximum = np.mean(
                    #    points_in_area[np.argpartition(points_in_area[:, 1], -number_of_considered_values)[
                    #                   -number_of_considered_values:]][:, 1])
                    maximum = np.median(
                        points_in_area[
                            np.argpartition(points_in_area[:, 1], -number_of_considered_values)[
                            -number_of_considered_values:]][:, 1])
                    #maximum = max(points_in_area[:, 1])

                if number_of_considered_values == 0:
                    minimum = min(points_in_area[:, 1])
                else:
                    try:
                        #minimum = min(points_in_area[:, 1])

                        #minimum = np.mean(
                        #    points_in_area[np.argpartition(points_in_area[:, 1], number_of_considered_values)[
                        #                   :number_of_considered_values]][:, 1])
                        minimum = np.median(
                            points_in_area[
                                np.argpartition(points_in_area[:, 1], number_of_considered_values)[
                                :number_of_considered_values]][:, 1])

                    except ValueError:
                        print('points in area ', points_in_area[:, 1])
                        print('number of considered values', number_of_considered_values)
                        pass

                #if minimum == maximum:
                #    asymmetry_value = maximum + minimum
                #if np.sign(maximum) == 1 and np.sign(minimum) == -1:
                #    asymmetry_value = abs(maximum + minimum)
                #if np.sign(maximum) == 1 and np.sign(minimum) == 1:
                #    asymmetry_value = maximum + minimum
                #if np.sign(maximum) == -1 and np.sign(minimum) == -1:
                #    asymmetry_value = abs(maximum) + abs(minimum)
                #if minimum > maximum:
                #    print('higher min than max --> choose smaller tolerance percentage')
                #    asymmetry_value = 10
                # TODO change this to half
                if normalize:
                    distance_min_to_zero = distance.euclidean(minimum, 0.5)
                    distance_max_to_zero = distance.euclidean(maximum, 0.5)
                else:
                    distance_min_to_zero = distance.euclidean(minimum, 0)
                    distance_max_to_zero = distance.euclidean(maximum, 0)
                asymmetry_value = abs(distance_max_to_zero-distance_min_to_zero)
                asymmetry = asymmetry + asymmetry_value

                if draw:
                    axs[i][0].scatter(step+tolerance, maximum, marker='.', color='green')
                    axs[i][0].scatter(step+tolerance, minimum, marker='.', color='red')
                    axs[i][0].set_aspect('equal')
                    axs[i][0].set_xlabel('x')
                    axs[i][0].set_ylabel('y')

                asymmetry_values.append(asymmetry_value)
        if draw:
            axs[i][0].axvline(segments[-1], alpha=0.2)

            axs[i][1].plot(asymmetry_values, marker='.')
            axs[i][1].set_aspect('auto')
            axs[i][1].set_xlabel('Segment')
            axs[i][1].set_ylabel('Difference')
        if outlier_removal:
            data_mean, data_std = np.mean(asymmetry_values), np.std(asymmetry_values)
            cut_off = data_std * number_of_std_for_outlier_removal
            lower, upper = data_mean - cut_off, data_mean + cut_off
            outliers = [x for x in asymmetry_values if x < lower or x > upper]
            outliers_removed = [x for x in asymmetry_values if x > lower and x < upper]
            asymmetries_single_values.append(outliers_removed)
            asymmetries.append(sum(outliers_removed))
            if draw:
                axs[i][1].axhline(lower, color='black')
                axs[i][1].axhline(upper, color='black')
        else:
            asymmetries_single_values.append(asymmetry_values)
            asymmetries.append(asymmetry)
    if draw:
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=0.97)
    return asymmetries, asymmetries_single_values


def asymmetries_y_axis(projections_2d, title=None, draw=False, n_segments=20, stepsize=2,
                       considered_percentage=0.05, outlier_removal=False, number_of_std_for_outlier_removal=3,
                       normalize=False):
    # normalize the points
    if normalize:
        normalized_projections = []
        min_max_scaler = preprocessing.MinMaxScaler()
        for unnormalized_projection in projections_2d:
            normalized_projections.append(
                min_max_scaler.fit_transform(np.array(unnormalized_projection)))
    else: normalized_projections = projections_2d

    asymmetries_2nd_PC = []
    tolerance = stepsize / 2

    asymmetries_single_values = []

    if draw:
        fig, axs = plt.subplots(len(projections_2d), 2, figsize=(8, 4 * len(projections_2d)))

    print('Calculating asymmetries for %d objects' % len(projections_2d))

    for i, projection in enumerate(tqdm(normalized_projections)):

        y_range_min, y_range_max = (min(projection[:, 1]), max(projection[:, 1]))

        if draw:
            axs[i][0].scatter(normalized_projections[i][:, 0], normalized_projections[i][:, 1], marker='.',
                              alpha=0.02,
                              c='black')

        asymmetry_y = 0
        asymmetry_values_y = []

        segments = np.linspace(min(projection[:, 1]), max(projection[:, 1]), n_segments,
                               endpoint=True)
        # segments = np.append(segments, (segments[-1]+abs(segments[0]-segments[1])))
        tolerance = abs(segments[0] - segments[1]) / 2

        #for step in np.arange(y_range_min, y_range_max, stepsize):

        for j, step in enumerate(segments[0:-1]):
            min_y_area = step
            max_y_area = segments[j + 1]
            points_in_area = np.array(
                [[x, y] for x, y in normalized_projections[i] if min_y_area <= y < max_y_area])
            if draw:
                axs[i][0].axhline(step, alpha=0.2)

            #points_in_area = np.array(
            #    [[x, y] for x, y in projections_2d[i] if y > step - tolerance and y < step + tolerance])

            if len(points_in_area) != 0:
                number_of_considered_values = int(
                    round(len(points_in_area) * considered_percentage))
                maximum_y = np.median(
                    points_in_area[
                        np.argpartition(points_in_area[:, 0], -number_of_considered_values)
                        [-number_of_considered_values:]][:, 0])
                minimum_y = np.median(
                    points_in_area[
                        np.argpartition(points_in_area[:, 0], number_of_considered_values)
                        [:number_of_considered_values]][:, 0])

                #maximum_y = max(points_in_area[:, 0])
                #minimum_y = min(points_in_area[:, 0])
                if normalize:
                    distance_min_to_zero = distance.euclidean(minimum_y, 0.5)
                    distance_max_to_zero = distance.euclidean(maximum_y, 0.5)
                else:
                    distance_min_to_zero = distance.euclidean(minimum_y, 0)
                    distance_max_to_zero = distance.euclidean(maximum_y, 0)
                asymmetry_value_y = abs(distance_max_to_zero-distance_min_to_zero)
                #if minimum_y == maximum_y:
                #    asymmetry_value_y = maximum_y + minimum_y
                #if np.sign(maximum_y) == 1 and np.sign(minimum_y) == -1:
                #    asymmetry_value_y = abs(maximum_y + minimum_y)
                #if np.sign(maximum_y) == 1 and np.sign(minimum_y) == 1:
                #    asymmetry_value_y = maximum_y + minimum_y
                #if np.sign(maximum_y) == -1 and np.sign(minimum_y) == -1:
                #    asymmetry_value_y = abs(maximum_y) + abs(minimum_y)
                #asymmetry_y = asymmetry_y + asymmetry_value_y

                if draw:
                    axs[i][0].scatter(maximum_y, step+tolerance, marker='.', color='green')
                if draw:
                    axs[i][0].scatter(minimum_y, step+tolerance, marker='.', color='red')
                asymmetry_y = asymmetry_y + asymmetry_value_y
                asymmetry_values_y.append(asymmetry_value_y)

        if draw and len(asymmetry_values_y) != 0:
            axs[i][0].axhline(segments[-1], alpha=0.2)
            axs[i][0].set_xlabel('x')
            axs[i][0].set_ylabel('y')

            axs[i][1].plot(asymmetry_values_y, marker='.')
            axs[i][1].set_aspect('auto')
            axs[i][1].set_ylabel('Difference')
            axs[i][1].set_xlabel('Segment')
        if outlier_removal:
            data_mean, data_std = np.mean(asymmetry_values_y), np.std(asymmetry_values_y)
            cut_off = data_std * number_of_std_for_outlier_removal
            lower, upper = data_mean - cut_off, data_mean + cut_off
            outliers = [x for x in asymmetry_values_y if x < lower or x > upper]
            outliers_removed = [x for x in asymmetry_values_y if x > lower and x < upper]
            asymmetries_single_values.append(outliers_removed)
            asymmetries_2nd_PC.append(sum(outliers_removed))
            if draw:
                axs[i][1].axhline(lower, color='black')
                axs[i][1].axhline(upper, color='black')
        else:
            asymmetries_2nd_PC.append(asymmetry_y)
            asymmetries_single_values.append(asymmetry_values_y)

    if draw:
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=0.97)

    return asymmetries_2nd_PC, asymmetries_single_values


def min_max_asymmetries(asymmetries_of_projections_1st_axis, asymmetries_of_projections_2nd_axis):
    #print('asym 1:', asymmetries_of_projections_1st_axis[0])
    #print('asym 2: ', asymmetries_of_projections_2nd_axis[0])

    #print('min ?: ', str(list((zip(asymmetries_of_projections_1st_axis,
    #                                    asymmetries_of_projections_2nd_axis)))))
    min_asymmetries = [min(x, y) for x, y in
                                    zip(asymmetries_of_projections_1st_axis,
                                        asymmetries_of_projections_2nd_axis)]
    #print(min_asymmetries)
    max_asymmetries = [max(x, y) for x, y in
                                    zip(asymmetries_of_projections_1st_axis,
                                        asymmetries_of_projections_2nd_axis)]
    #print(max_asymmetries)

    return list(zip(min_asymmetries, max_asymmetries))


def get_min_max_varimax(pointclouds, stepsize=2, n_segments=20, get_1st_and_3rd_component=False):
    if get_1st_and_3rd_component:
        projections = varimax_projections_2d(pointclouds, get_1st_and_3rd_component=True)
    else:
        projections = varimax_projections_2d(pointclouds)
    asymmetries_x, _ = asymmetries_x_axis(projections, n_segments=n_segments)
    asymmetries_y, _ = asymmetries_y_axis(projections, n_segments=n_segments)
    return min_max_asymmetries(asymmetries_x, asymmetries_y)


def remove_outliers(data, max_number_std=2):
    mean = np.mean(data)
    std_dev = np.std(data)
    zero_based = abs(data - mean)
    return data[zero_based < max_number_std * std_dev]
