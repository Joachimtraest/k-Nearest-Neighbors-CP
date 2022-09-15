from operator import concat
import numpy as np

class ICP_kNN:
    @staticmethod
    def edist(v1, v2): #euclidean distance between two vectors
        return np.sqrt(np.sum((v1 - v2)**2))

    @staticmethod
    def knn_reg(x_train, y_train, tx, k): #k Nearest Neighbor regression method (source: Papadopoulos, Vovk and Gammerman 2011)
        distances = np.zeros(np.shape(x_train)[0])
        for i in range(np.shape(x_train)[0]):
            distances[i] = ICP_kNN.edist(x_train[i],tx)
        inds_sorted = np.argsort(distances)

        y_sorted = y_train[inds_sorted]
        value = np.average(y_sorted[:k])
        return value
    
    @staticmethod
    def get_alpha(x_train, y_train, q, k, epsilon = 0.05): #alpha as explained on p.823 of Papadopoulos (2011)
        #first split the training set in a 'proper training set' and a calibration set 
        x_calib = x_train[0:q][:]
        y_calib = y_train[0:q]
        x_ptrain = x_train[q:][:]
        y_ptrain = y_train[q:]
        calib_non_conf_scores = np.zeros(np.shape(x_calib)[0])
        for i in range(len(y_calib)):
            y_predict = ICP_kNN.knn_reg(x_ptrain, y_ptrain, x_calib[i], k)
            calib_non_conf_scores[i] = ICP_kNN.edist(y_predict, y_calib[i])
        non_conf_sorted = np.flip(np.sort(calib_non_conf_scores))
        #the euclidean distance between y and the predicted y is nonconformity measure 
        s = np.floor(epsilon*(q+1))
        alpha_s = non_conf_sorted[int(s) - 1] #95% percent of the time, the distance between the predicted y and the real y will be less than alpha_s (when this calibration set is used)
        return alpha_s

    @staticmethod
    def get_ICP_knn_prediction_region(x_test, x_train, y_train, q, k, alpha_s): #construct the prediction region as explained on p.823 Papadopoulos (2011)
        x_ptrain = x_train[q:][:]
        y_ptrain = y_train[q:]
        alpha_s = alpha_s
        y_predicted = ICP_kNN.knn_reg(x_ptrain, y_ptrain, x_test, k)
        left_bound = y_predicted - alpha_s
        right_bound = y_predicted + alpha_s

        return (left_bound, right_bound)

    @staticmethod
    def K_fold_validation_ICP_knn(x,y, folds, k, q, epsilon = 0.05): #K fold cross validation 
        #the algorithm is run K times, each time with another test set, the examples which are not in the test set are in the training set
        #this function counts how many times, y is not in the prediction interval, this should approximately be equal to epsilon
        perm = np.random.permutation(np.shape(x)[0])
        x_random = x[perm]
        y_random = y[perm]
        n = len(y)%folds
        last_part_x = x[-n:]
        last_part_y = y[-n:]
        x_split = np.split(x[:-n], folds)
        y_split = np.split(y[:-n],folds)
        x_split[-1] = np.concatenate((x_split[-1], last_part_x))
        y_split[-1] = np.concatenate((y_split[-1], last_part_y))
        hits = 0
        errors = 1
        count = 0
        for i in range(len(x_split)):
            test_x = x_split[i]
            test_y = y_split[i]
            if i == 0:
                train_x = np.concatenate(x_split[i+1:], axis = 0)
                train_y = np.concatenate(y_split[i+1:], axis = None)
            elif i == len(x_split) - 1:
                train_x = np.concatenate(x_split[:i], axis = 0)
                train_y = np.concatenate(y_split[:i], axis = None)
            else:
                train_x = np.concatenate( (np.concatenate((x_split[:i]), axis = 0),np.concatenate((x_split[i+1:]), axis = 0)  ))
                train_y = np.concatenate( (np.concatenate((y_split[:i]), axis = None),np.concatenate((y_split[i+1:]), axis = None)  ))
            
            alpha_s = ICP_kNN.get_alpha(train_x, train_y, q, k, epsilon = epsilon)
            for j in range(len(test_x)):
                count += 1
                (left_bound, right_bound) = ICP_kNN.get_ICP_knn_prediction_region(test_x[j], train_x, train_y, q, k, alpha_s)
                if test_y[j] >= left_bound and test_y[j] <= right_bound:
                    hits += 1
                else:
                    errors += 1
                print('progression: '+ str(count) + '/' + str(len(x)))
        error_fraction = errors/len(y)
        error_procent = error_fraction*100
        return str(error_procent) + '%' + ' of the predictions were NOT inside the '  + str(100 - epsilon*100) + '% prediction interval'



