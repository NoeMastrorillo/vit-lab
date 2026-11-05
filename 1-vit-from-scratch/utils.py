
def images_to_patches(imgs, patch_size):
    """ Input: (batch_size, n_channels, height, width)
        Output: (batch_size, n_patches, n_channels*patch_size*patch_size)
    """
    batch_size, n_channels, height, width = imgs.shape
    assert height % patch_size == 0 and width % patch_size == 0, "Image dimensions must be a multiple of the patch size"

    # n_patches = (height*width) // (patch_size*patch_size)
    imgs = imgs.view(batch_size, n_channels, height//patch_size, patch_size, width//patch_size, patch_size)
    imgs = imgs.permute(0, 2, 4, 1, 3, 5)    # (batch_size, height//patch_size, width//patch_size, n_channels, patch_size, patch_size)
    return imgs.reshape(batch_size, -1, n_channels*patch_size*patch_size)
