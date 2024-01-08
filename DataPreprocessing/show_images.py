from matplotlib import pyplot as plt


def show_images(n, data, cfg):
    if cfg['mode'] == 'dev':
        batch = data.next()
        fig, ax = plt.subplots(ncols=n, figsize=(20, 20))
        for idx, img in enumerate(batch[0][:n]):
            ax[idx].imshow(img.astype(int))
            ax[idx].title.set_text(('car', 'motorbike')[batch[1][idx]])
        plt.show()
    elif cfg['mode'] == 'prod':
        print('ModeTypeError: For image visualization, you need to switch the mode in the config file to dev mode.')


def show_predict(img, yhat):
    fig, ax = plt.subplots(1, figsize=(20, 20))
    ax[0].imshow(img)
    if yhat[0] > yhat[1]:
        ax[0].title.set_text('car')
    else:
        ax[0].title.set_text('motorbike')
    plt.show()