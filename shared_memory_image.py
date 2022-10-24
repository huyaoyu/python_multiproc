from multiprocessing import shared_memory
import numpy as np

def shm_size_from_img_shape( img_shape, n_img ):
    return int( np.prod( img_shape ) * n_img )

class SharedMemoryImage(object):
    def __init__(self, name: str, img_shape: tuple):
        '''
        This is a wrapper class for using shared memory to store images across
        mulitple processes. The shared memory object is created by the user and
        refered by the name.

        The user specifies the shape of the image. And the total number of images
        in the shared memory is then determined by the size of the shared memory. 
        
        Note that the user is supposed to call initialize() and finalize() to properly
        handle the shared resources.

        name: The name of the shared memory object.
        img_shape: The shape of the image, including channel number.
        '''
        super().__init__()

        self.name = name
        self.img_shape = img_shape
        self.img_capacity = np.prod( self.img_shape )

        self.shm = None # User must call initialize() to assign a valid value.
        self._n_img = 0 # This should not be assigned buy the user.

    @property
    def n_img(self):
        return self._n_img

    def initialize(self):
        if ( self.shm is not None ):
            raise Exception('Already initialized. ')

        self.shm = shared_memory.SharedMemory(name=self.name)

        # Check the size of the shared memory.
        assert ( self.shm.size > 0 and self.shm.size % self.img_capacity == 0 ), \
            'Incompatible shared memory size. self.shm.size = {}, self.img_capacity = {}'.format( 
                self.shm.size, self.img_capacity )
        self._n_img = self.shm.size // self.img_capacity

    def finalize(self):
        if ( self.shm is not None ):
            self.shm.close()

    def __del__(self):
        self.finalize()

    def seek(self, idx):
        assert ( idx < self.n_img and idx >= 0 ), 'self.n_img = {}, idx = {}'.format( self.n_img, idx )
        return self.shm.buf[ idx * self.img_capacity : (idx+1) * self.img_capacity ]

    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        return np.ndarray( 
            ( self.img_capacity, ), 
            dtype=np.uint8, 
            buffer=self.seek(idx) 
            ).reshape( self.img_shape )