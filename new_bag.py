import cv2
import numpy as np 
from glob import glob 
import argparse
# from hashcode import HashTable, Hashing
from helper import *
from matplotlib import pyplot as plt 


class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def trainModel(self):
        """
        This method contains the entire module 
        required for training the bag of visual words model
        Use of helper functions will be extensive.
        """

        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
        # extract SIFT Features from each image
        label_count = 0 
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print ("Computing Features for ", word)
            for im in imlist:
                # cv2.imshow("im", im)
                # cv2.waitKey()
                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.im_helper.features(im)
                self.descriptor_list.append(des)

            label_count += 1


        # perform clustering
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        self.bov_helper.plotHist()
 

        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)


    def recognize(self,test_img, test_image_path=None):

        """ 
        This method recognizes a single image 
        It can be utilized individually as well.
        """

        kp, des = self.im_helper.features(test_img)
        # print kp
        print (des.shape)

        # generate vocab for test image
        vocab = np.array( [[ 0 for i in range(self.no_clusters)]])
        # locate nearest clusters for each of 
        # the visual word (feature) present in the image
        
        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        # print test_ret

        # print vocab
        for each in test_ret:
            vocab[0][each] += 1

        print (vocab)
        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)

        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb

    predictions = []

    def testModel(self):
        """ 
        This method is to test the trained classifier
        read all images from testing path 
        use BOVHelpers.predict() function to obtain classes of each image
        """

        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

        predictions = []

        for word, imlist in self.testImages.items():
            print ("processing " ,word)
            for im in imlist:
                # print imlist[0].shape, imlist[1].shape
                print (im.shape)
                cl = self.recognize(im)
                print (cl)
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))]
                    })

        print (predictions)
        for each in predictions:
            # cv2.imshow(each['object_name'], each['image'])
            # cv2.waitKey()
            # cv2.destroyWindow(each['object_name'])
            # 
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()


    def print_vars(self):
        pass

    empty=[3,4,5,6,7,8,9,10,11]
    HashTable = [i for i in empty]

    def display_hash(hashTable):
	
	    for i in range(len(hashTable)):
		    print(i, end = " ")
		
		    for j in hashTable[i]:
			    print("-->", end = " ")
			    print(j, end = " ")
			
		    print()

    # Creating Hashtable as
    # a nested list.

    # Hashing Function to return
    # key for every value.
    def Hashing(keyvalue):
	    return keyvalue % len(HashTable)


    # Insert Function to add
    # values to the hash table
    def insert(Hashtable, keyvalue, value):
	
	    hash_key = Hashing(keyvalue)
	    Hashtable[hash_key].append(value)

    # Driver Code
    insert(HashTable, 10, 'Allahabad')
    insert(HashTable, 25, 'Mumbai')
    insert(HashTable, 20, 'Mathura')
    insert(HashTable, 9, 'Delhi')
    insert(HashTable, 21, 'Punjab')
    insert(HashTable, 23, 'jammu')
    insert(HashTable, 16, 'rae bareily')
    insert(HashTable, 18, 'srinagar')
    insert(HashTable, 29, 'agra')
    insert(HashTable, 9, 'vellore')
    insert(HashTable, 11, 'chennai')

    # display_hash (HashTable)


if __name__ == '__main__':

    # parse cmd args
    parser = argparse.ArgumentParser(
            description=" Bag of visual words example"
        )
    parser.add_argument('--train_path', action="store", dest="train_path", required=True)
    parser.add_argument('--test_path', action="store", dest="test_path", required=True)

    args =  vars(parser.parse_args())
    print (args)

    
    bov = BOV(no_clusters=100)

    # set training paths
    bov.train_path = args['train_path'] 
    # set testing paths
    bov.test_path = args['test_path'] 
    # train the model
    bov.display_hash (HashTable)

    bov.trainModel()
    # test model
    bov.testModel()