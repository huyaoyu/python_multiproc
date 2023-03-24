from multiprocessing import shared_memory
import numpy as np

def shm_size_from_img_shape( img_shape, channel_depth, n_grp=1 ):
    return int( np.prod( img_shape ) * channel_depth * n_grp )

# ==================== Processor Functions ====================
def pass_forward(img):
    return img

def enforce_group_dim(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
        
    if img.ndim == 3:
        return np.expand_dims(img, axis=0)
    elif img.ndim == 4:
        return img
    else:
        raise Exception(f'img.shape = {img.shape}. Expecting image.ndim in [2, 3, 4]. ')

def c4_uint8_as_float(img):
    img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
    assert img.ndim == 3 and img.shape[-1] == 4, \
        f'img.shape = {img.shape}. Expecting image.shape[-1] == 4. '
    
    # Check if the input array is contiguous.
    if ( not img.flags['C_CONTIGUOUS'] ):
        img = np.ascontiguousarray(img)

    return np.expand_dims( img.view('<f4'), axis=0)

def c4_uint8_as_float_arr(img):
    img = np.expand_dims(img, axis=-1) if img.ndim == 3 else img
    assert img.ndim == 4 and img.shape[-1] == 4, \
        f'img.shape = {img.shape}. Expecting image.shape[-1] == 4. '
    
    # Check if the input array is contiguous.
    if ( not img.flags['C_CONTIGUOUS'] ):
        img = np.ascontiguousarray(img)

    return img.view('<f4')

def float_as_c4_uint8(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    elif img.ndim == 3 and img.shape[-1] != 1:
        raise Exception(f'img.shape = {img.shape}. Expecting image.shape[-1] == 1. ')
    
    # Check if the input array is contiguous.
    if ( not img.flags['C_CONTIGUOUS'] ):
        img = np.ascontiguousarray(img)

    return np.expand_dims( img.view('<u1'), axis=0 )

def float_as_c4_uint8_arr(img):  
    if img.ndim == 3:
        img = np.expand_dims(img, axis=-1)
    elif img.ndim == 4 and img.shape[-1] != 1:
        raise Exception(f'img.shape = {img.shape}. Expecting image.shape[-1] == 1. ')
    
    # Check if the input array is contiguous.
    if ( not img.flags['C_CONTIGUOUS'] ):
        img = np.ascontiguousarray(img)

    return img.view('<u1')

# ==================== End of Processor Functions ====================

class SharedMemoryImage(object):
    def __init__(self, name: str, img_shape: tuple, channel_depth: int=1, grp_sz: int=1, 
                 n_grp: int=20, processor_in=None, processor_out=None):
        '''
        This is a wrapper class for using shared memory to store images across
        mulitple processes. The shared memory object is created by the user and
        refered internally by the name argument.

        The user specifies the shape (H, W, C) of the image, the channel depth (1 for uint8 type
        and 4 for float type, etc), the group size, and the toal number of groups (n_grp). Then the 
        total byte size of the shared memory is >= (H * W * C * channel_depth * grp_sz * n_grp), 
        depdending on the architecture of the operating system.
        
        Note that the user is supposed to call initialize() and finalize() to properly
        handle the shared resources.

        Note that internally, this class treats all image as uint8, thus byte. The user must properly
        set the channel_depth value and the associated processor_in and processor_out functions. 
        porcessor_in is used to convert the input image to uint8 type with channel_depth. 
        Inversely, processor_out is used to convert the image from uint8 type to the original type.
        Use None for processor_in and processor_out if no conversions are needed.
        
        Note that internally, all images are stored in groups. Some even for a single image, the 
        shape is [1, H, W, C]. If the user wants to strip the first dimension, then a dedicated
        output processor function must be used.

        name: The name of the shared memory object.
        img_shape: The shape of the image, [H, W] or [H, W, C] including channel number.
        channel_depth: The number of bytes per channel. Default is 1. For single-precision
            floating point values, use 4.
        grp_sz: The number of images in a group. Default is 1.
        n_grp: The total number of groups. Default is 20.
        processor_in: A callable object that performs the conversion to uint8.
        processor_out: A callable object that performs the conversion from uint8.
        '''
        super().__init__()

        self.name = name # The name of the shared memory object.
        
        #print(f"SharedMemoryImage ~~ Group_Sz: {grp_sz}, Image_Shape: {img_shape}")
        self.grouped_img_shape = (grp_sz, *img_shape[:2], img_shape[2] * channel_depth) \
            if len(img_shape) == 3 else (grp_sz, *img_shape[:2], channel_depth )
        #print(f"Resulting Img Shape: {self.grouped_img_shape}")

        self.grp_capacity = np.prod( self.grouped_img_shape )

        self.processor_in  = processor_in if processor_in is not None else pass_forward
        self.processor_out = processor_out if processor_out is not None else pass_forward

        self.shm = None # User must call initialize() to assign a valid value.
        self.n_grp = n_grp
        self.grp_sz = grp_sz
        
    @property
    def n_img(self):
        return self.n_grp * self.grp_sz
    
    @property
    def img_shape(self):
        '''
        Note that internally images are always represented as bytes.
        '''
        return self.grouped_img_shape[1:]

    def initialize(self):
        if ( self.shm is not None ):
            raise Exception('Already initialized. ')

        # print(f"Inside Initialization: {self.name}")
        self.shm = shared_memory.SharedMemory(name=self.name)

        # Check the size of the shared memory.
        assert ( self.shm.size > 0 and self.shm.size // self.grp_capacity == self.n_grp ), \
            ( f'Incompatible shared memory size. self.shm.size = {self.shm.size}, '
              f'self.grp_capacity = {self.grp_capacity}, '
              f'self.grouped_img_shape = {self.grouped_img_shape}. ' )

    def finalize(self):
        if ( self.shm is not None ):
            self.shm.close()

    def __del__(self):
        self.finalize()

    def seek(self, idx):
        assert ( idx < self.n_grp and idx >= 0 ), f'self.n_grp = {self.n_grp}, idx = {idx}. '
        return self.shm.buf[ idx * self.grp_capacity : (idx+1) * self.grp_capacity ]

    def __len__(self):
        return self.n_grp

    def as_array_at_idx(self, idx):
        return np.ndarray( 
            ( self.grp_capacity, ), 
            dtype=np.uint8, 
            buffer=self.seek(idx) 
            ).reshape( self.grouped_img_shape )

    def __getitem__(self, idx):
        array = self.as_array_at_idx(idx)
        return self.processor_out(array)

    def __setitem__(self, idx, img):
        '''
        Note that the preprocessor must know how to handle the image dimension. img could ba a
        single image with shape [H, W, C] or [H, W]. Or it could be a group of images with shape
        [G, H, W, C] or [G, H, W]. There is an ambiguity between single 3-channel image and a group
        of 1-channel images. Thus, the preprocessor must know how to handle this ambiguity.
        
        img: NumPy array of shape (G, H, W, C) or (G, H, W) with G being the group size.
        '''
        img = self.processor_in(img)
        array = self.as_array_at_idx(idx)
        #     G, H, W, C
        array[:, :, :, ...] = img
        