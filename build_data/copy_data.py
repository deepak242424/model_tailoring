import shutil, errno

src_dir = '/imagenet/preprocessed/raw-data/train/'
dst_dir = '/imagenet/model_tailoring/raw_data/imagenet_subset/'


items = ['n04152593', 'n02917067', 'n04344873','n03991062', 'n09835506', 'n10148035', 'n10565667', 'n03785016', 'n02389026', 'n03982430', 'n03179701', 'n02791124', 'n04099969', 'n04429376', 'n03376595','n02701002', 'n03670208', 'n03594945', 'n03777568', 'n02930766', 'n03770679', 'n03100240', 'n04037443', 'n02814533', 'n04285008','n02124075', 'n02123394', 'n02123045', 'n02123597', 'n02123159','n01514668', 'n01514859','n02110341', 'n02113978', 'n02110958', 'n02111277', 'n02111129', 'n02110806', 'n02111500','n04606251', 'n03947888', 'n03447447', 'n03344393', 'n04147183']

for item in  items:
    shutil.copytree(src_dir + item, dst_dir + item)
