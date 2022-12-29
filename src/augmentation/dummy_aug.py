
class Augmentation:
    def __init__(self, __name__ = "DEFAULT", dataset=None):
        self.__name__ = __name__
        self.dataset = dataset
        
    def add_name_col(self):
        ## to be updated
        return self.dataset
    
    def augment(self):
        ## write your custom augmentation script
        raise NotImplementedError
    
    def get_dataset(self):
        self.augment()
        self.add_name_col()
        return self.dataset

class DummyAug(Augmentation):
    def __init__(self, __name__ = "DEFAULT", dataset=None):
        super().__init__(__name__=__name__, dataset=dataset)
        
    def augment(self):
        pass
        