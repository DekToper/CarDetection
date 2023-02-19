import os
import imghdr
import sys


def clean_dataset(data_dir, img_exts):
    print('Start preprocessing dir ' + data_dir)
    for image_class in os.listdir(data_dir):
        k = 1
        print('Start checking ' + image_class + ' dir...')
        images = os.listdir(os.path.join(data_dir, image_class))
        for image in images:
            sys.stdout.write("\rCheck №%d" % k + " / " + str(len(images)))
            sys.stdout.flush()
            image_path = os.path.join(data_dir, image_class, image)
            try:
                tip = imghdr.what(image_path)
                if tip not in img_exts:
                    print('Image not in ext list')
                    os.remove(image_path)
            except Exception as e:
                print(e)
                print('Issue with image {}'.format(image_path))
            k += 1
        print('')