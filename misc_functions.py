import numpy as np
from collections import Counter

# function to get unique values 
def unique(list1): 
    '''https://www.geeksforgeeks.org/python-get-unique-values-list/#:~:text=Using%20set()%20property%20of,a%20list%20to%20print%20it.'''
    x = np.array(list1) 
    print(list1)
    print(np.unique(x)) 

# define methods for class-count visualization


def class_count(dataset):
    count = dict(Counter(dataset.targets))
    count = dict(zip(dataset.classes[::-1], list(count.values())))      # changing keys of dictionary 
    return count

def plot_class_count(dataset, name='Dataset Labels Count'):
    count = class_count(dataset)
    pd.DataFrame(count, index=['Labels']).plot(kind='bar', title=name).show()

# show batches
def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))
    images = make_grid(images, nrow=4, padding=15)
    imshow(images, title=["NORMAL" if x==0  else "PNEUMONIA" for x in labels])