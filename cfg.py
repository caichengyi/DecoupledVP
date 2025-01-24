DOWNSTREAM_PATH = ""

dataset_clusters = {'caltech101':3 ,'dtd':3 ,'eurosat':1 ,'fgvc':3 ,'food101':2, 'oxford_flowers':3 ,'oxford_pets':3,'stanford_cars':3 ,'sun397':3 ,'ucf101':2, 'resisc45':3}
param = {
    'dvp-cse': {'epoch': 200, 'lr': 40, 'input_size': 192, 'shot':16, 'causes': 3, 'cause_no': '1,2,3', 'k':3, 'm': 20},
    'dvp-cls': {'epoch': 200, 'lr': 40, 'input_size': 192, 'shot':16, 'clusters': dataset_clusters, 'k':3, 'm': 20}
}


