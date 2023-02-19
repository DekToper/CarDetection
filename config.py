def load_config():
    config = dict(
        dataset_dir='/DataSet',
        img_exts=['jpg', 'jpeg'],
        is_clean=False,
        mode='dev',  # for production mode need to equal 'prod'
        logs_dir='/DeepLearning/logs'
    )

    return config
