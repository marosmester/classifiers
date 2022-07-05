import argparse             # parsing of arguments from command line
import os                   # in order to search and load data from a given directory
from PIL import Image      # for reading images 
import numpy as np
 
FILENAME = "truth.dsv"
ARTIF_TEST_FILES = 0
INTENSITY = 256
GOOD_ENOUGH_FOR_ME = 0.82
TRAINING_PERCENTAGE = 0.85


# helper functions----------------------------------------------------------------------------------------------------
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument('-k', type=int, 
                             help='run k-NN classifier (if k is 0 the code may decide about proper K by itself')
    mutex_group.add_argument("-b", 
                             help="run Naive Bayes classifier", action="store_true")
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser

def read_truth_dsv(filepath):
    """
    :returns: dict filled with correct classifications,
              dict of all possible classes with total number of their occurences,
    """
    data_dict = {}
    all_classes = {}
    nbr_of_samples = 0
    with open(filepath, "r") as f:
        for rawline in f:
            nbr_of_samples += 1
            line = rawline.strip()
            separated = line.split(":")
            if separated[1] in all_classes:
                all_classes[separated[1]] += 1
            else:
                all_classes[separated[1]] = 1
            data_dict[separated[0]] = separated[1]
    return data_dict, all_classes, nbr_of_samples

def evaluate_classifier(truth_dict, result_dict):
    correct = 0
    total = 0
    for image in list(result_dict.keys()):
        if truth_dict[image] == result_dict[image]:
            correct += 1
        total += 1

    return correct/total
#----------------------------------------------------------------------------------------------------------------------

# Python classes-------------------------------------------------------------------------------------------------------
class NaiveBayes():
    def __init__(self, all_classes_dict, truth_data_dict, nbr_of_samples, train_path, test_path, output_path):
        #from input:
        self.all_classes_dict = all_classes_dict
        self.all_classes_keys = list(all_classes_dict.keys())
        self.truth_data= truth_data_dict                            #KEY=image_name, VALUE=its class
        self.nbr_of_samples = nbr_of_samples - ARTIF_TEST_FILES
        self.train_path= train_path
        self.test_path= test_path
        self.output_path = output_path
        #other:
        self.img_size = self.get_image_size()
        self.intensities_dict = self.create_intensities_dict()   # dictionary of dictionaries
        self.class_probabilities = {}
        self.nbr_of_training = int(TRAINING_PERCENTAGE * self.nbr_of_samples)
        self.edited_scale = self.compute_scale()   # should be equal to INTESITY (=256) if one wants default settings
        print(self.edited_scale)
        self.rsd_coeff = round(256/self.edited_scale) # resulution scale down
    
    def compute_scale(self):
        min_occurence = min(list(self.all_classes_dict.values()))
        if min_occurence < 12:
            scale = 4
        elif min_occurence < 50:
            scale = 6
        else:
            scale = 256


        return scale

    def get_image_size(self):
        ret = 0
        img_name = ""
        for fname in os.listdir(self.train_path):
            if fname[-4:] == ".png":
                img_name = fname
                break
        
        if img_name != "":
            img_path = self.train_path + "/" + img_name
            image_vector = np.array(Image.open(img_path)).flatten()
            ret = len(image_vector)

        return ret

    def create_intensities_dict(self):
        ret = {}
        for class_ in self.all_classes_keys:
            ret[class_] = {}
            for i in range(0,self.img_size):
                ret[class_][i] = []
        
        return ret      # dictionary of dictionaries

    def read_data(self):
        cnt = 0 
        for fname in os.listdir(self.train_path):
            if cnt >= self.nbr_of_training:
                break
            if fname[-4:] == ".png":
                current_class = self.truth_data[fname]
                #create 1D vector from the image:
                img_path = self.train_path + "/" + fname
                img_vector = np.array(Image.open(img_path)).flatten()
                #write down the intensities:
                size = len(img_vector)
                for i in range(size):
                    self.intensities_dict[current_class][i].append(img_vector[i])
            cnt += 1

    def train(self, hyperparam):
        #1: read the intensities from images
        self.read_data()
        print("using hyperparam=", hyperparam)
        
        #2: calculate 'prior' class probabilities
        total = sum(self.all_classes_dict.values())
        for class_ in self.all_classes_keys:
            self.class_probabilities[class_] = self.all_classes_dict[class_]/total
        
        #3: create np.arrays of intensity probabilities for each pixel in each class using Laplace smoothing
        trained_probabilities = {}
        for class_ in self.all_classes_keys:
            arr = np.empty((self.img_size, self.edited_scale))
            #class_occurence = self.all_classes_dict[class_]         # is also the total number of values measured, N
            for pixel in range(self.img_size):                      
                intensities = self.intensities_dict[class_][pixel]
                for i in range(len(intensities)):
                    intensities[i] = intensities[i]//self.rsd_coeff
                #Laplace (maybe class_occurence here instead of len(intensities)... but should be equal ) :                 
                zero_value = hyperparam /( len(intensities) + hyperparam*self.edited_scale)  
                row = np.full((1,self.edited_scale), zero_value)
                for intens in list(set(intensities)):                              
                    row[0][intens] = ( intensities.count(intens) + hyperparam )/( len(intensities) + hyperparam*self.edited_scale) #LAPLACE
                
                arr[pixel,:] = row
                    
            trained_probabilities[class_] = arr

        return trained_probabilities

    def train_main(self):               # sadly, this has very high computation time
        k_list = [1e-11, 1e-8, 1e-5, 1e-4, 1e-2,]
        best_success = 0
        final_learned = 0
        k_picked = 0
        for k in k_list:
            if best_success > GOOD_ENOUGH_FOR_ME:
                break
            learned = self.train(k)
            classified_data = self.try_it_on_validation_data(learned)
            success = evaluate_classifier(self.truth_data, classified_data)
            print("k=",k,"success rate=", success)
            if success > best_success:
                final_learned = learned
                best_success = success
                k_picked = k
        
        print("PICKED HYPERPARAMETER=", k_picked)
        return final_learned

    def test(self, learned, validation = False):
        """
        :param learned: dictionary of numpy arrays, each dict item corresponds to one class 
        """
        if validation:
            path = self.train_path
        else:
            path= self.test_path

        classified = {}
        cnt = 0
        for fname in os.listdir(path):
            """for every image"""
            if validation and (cnt < self.nbr_of_training):
                cnt += 1
                continue
            if fname[-4:] == ".png":
                #create 1D vector from the image:
                img_path = path + "/" + fname
                img_vector = np.array(Image.open(img_path)).flatten()
                for i in range(len(img_vector)):
                    img_vector[i] = img_vector[i]//self.rsd_coeff

                all_p_d_given_x = {}
                for class_ in self.all_classes_keys:
                    """for every possible class"""
                    log_p_x_given_s = 0
                    for i in range(self.img_size):
                        """for every pixel in the new image"""
                        log_p_x_given_s += np.log10(learned[class_][i][img_vector[i]])
                    
                    all_p_d_given_x[class_] = log_p_x_given_s + np.log10(self.class_probabilities[class_])

                delta_star = max(list(all_p_d_given_x.items()), key=lambda x: x[1])[0]
                classified[fname] = delta_star
            
        if not validation:
            with open(self.output_path, "w") as out:
                for item in list(classified.keys()):
                    out.write(item+":"+str(classified[item])+"\n")

        return classified

    def try_it_on_validation_data(self, learned):
        return self.test(learned, True)

class kNN_classifier():
    def __init__(self, truth_data_dict, nbr_of_samples, train_path, test_path, output_path):
        #from input:
        self.truth_data = truth_data_dict           #KEY=image_name, VALUE=its class
        self.train_path= train_path
        self.test_path= test_path
        self.output_path = output_path
        #other:

    def train(self):
        trained_data = []
        for fname in os.listdir(self.train_path):
            if fname[-4:] != ".dsv":
                img_path = self.train_path + "/" + fname
                img_vector = np.array(Image.open(img_path)).astype(int).flatten()
                trained_data.append( (self.truth_data[fname], img_vector) )

        return trained_data

    def test(self, trained_data, k):
        classified ={}
        for fname in os.listdir(self.test_path):
            img_path = self.test_path + "/" + fname
            x = np.array(Image.open(img_path)).astype(int).flatten()

            dist_table = np.empty( (len(trained_data), 2) )
            row_cnt = 0
            row_to_class = []
            for pair in trained_data:
                class_name = pair[0]
                vector = pair[1]
                diff = vector - x
                d = np.linalg.norm(diff)

                row_to_class.append(class_name)
                row = np.array([row_cnt , d], dtype=float)
                dist_table[row_cnt,:] = row

                row_cnt += 1

            if k==0:
                k = 3       

            if k == 1:
                dist_column = dist_table[:,1]
                min_idx = np.argmin(dist_column)
                classified[fname] = row_to_class[ int(dist_table[min_idx][0]) ]     
            
            elif k > 1:
                neighbours = []

                #find nearest neighbours, put them (their class names) into neighbours
                for i in range(k):
                    dist_column = dist_table[:,1]
                    min_idx = np.argmin(dist_column)
                    neighbours.append(row_to_class[ int(dist_table[min_idx][0]) ])
                    dist_table[min_idx][1] = np.inf

                # find the most frequent element in a list:
                decision = max(set(neighbours), key = neighbours.count)

                classified[fname] = decision

        return classified

    def create_output_file(self, classified):
        with open(self.output_path, "w") as out:
            for item in list(classified.keys()):
                out.write(item+":"+str(classified[item])+"\n")


#----------------------------------------------------------------------------------------------------------------------

def main():
    #-- Parsing start --
    parser = setup_arg_parser()
    args = parser.parse_args()
    #-- Parsing end--
    
    truth_data, all_classes, nbr_of_samples = read_truth_dsv(args.train_path + "/" + FILENAME) 

    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    if args.k is not None:
        print(f"Running k-NN classifier with k={args.k}")
        knn = kNN_classifier(truth_data, nbr_of_samples, args.train_path, args.test_path, args.o)
        print("Training started")
        trained_data = knn.train()
        print("Training ended")
        
        print("Testing started.")
        result = knn.test(trained_data, args.k)
        knn.create_output_file(result)
        print("Testing ended")

        #print("Portion of successful classsification=", evaluate_classifier(truth_data, result) )

    elif args.b:
        print("Running Naive Bayes classifier")
        print("all_classes", all_classes)
        nb = NaiveBayes(all_classes, truth_data, nbr_of_samples, args.train_path, args.test_path, args.o)
        print("TOTAL SAMPLES=",nb.nbr_of_samples)
        print("FOR TRAINING=",nb.nbr_of_training)

        print("Training started")
        trained = nb.train_main()
        print("Training ended")
        
        print("Testing started.")
        print(list(trained.keys()))
        result = nb.test(trained)
        print("Testing ended")

        #print("Portion of successful classsification=", evaluate_classifier(truth_data, result) )
    

if __name__ == "__main__":
    main()
    
