import torch
import os
from torchvision import models, transforms
from PIL import Image
import fnmatch
import pickle


for run in range(5):
    modelname = 'resnet512_five_aug_{}'.format(run)
    basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/'
    trainsite = 'duck'
    modelpath= '{}/resnet_models/train_on_{}/{}.pth'.format(basedir, trainsite, modelname)
    imgdir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/match_nbn/'

    res_height = 512 #height
    res_width = 512 #width

    ##load model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
    nb_classes = len(classes)



    def preprocess(image_path, res_height, res_width):
        transform = transforms.Compose([transforms.Resize((res_height,res_width)), transforms.ToTensor()])
                                           #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
            raw_image = transform(image)

        return raw_image, raw_image


    def load_images(test_IDs, res_height, res_width):
        images = []
        raw_images = []

        for ID in test_IDs:
            image, raw_image = preprocess(ID, res_height, res_width)
            images.append(image)
            raw_images.append(raw_image)

        return images, raw_images

    augmentations = ['flips', 'gamma', 'rot', 'erase', 'translate']
    years = ['1986', '1987', '1988']
    #Find the appropriate images
    all_imgs = os.listdir(imgdir)
    test_imgs = []
    for aa in all_imgs:
        year = aa.split('.')[5]
        if any([year in aa for year in years]):
            if any([sub in aa for sub in augmentations]):
                continue
            else:
                test_imgs.append(aa)


    #filter out augmented images:

    #filter out trainfiles
    # with open('../ResNet/labels/duck_daytimex_trainfiles.no_aug.pickle', 'rb') as f:
    #     trainfiles = pickle.load(f)
    #
    # test_IDs = [imgdir + tt for tt in test_imgs if tt not in trainfiles]
    test_IDs = [imgdir + tt for tt in test_imgs]



    images, raw_images = load_images(test_IDs, res_height, res_width)
    images = torch.stack(images).to(device)

    if 'resnet' in modelname:
        model_conv = models.resnet50()
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = torch.nn.Linear(num_ftrs, nb_classes)


    model_conv.load_state_dict(torch.load(modelpath))
    model_conv = model_conv.to(device)
    model_conv.eval()

    predictionary = {}
    for ii, (image, test_ID) in enumerate(zip(images, test_IDs)):
        image = image.unsqueeze(dim = 0)
        logits = model_conv(image)
        probs = torch.nn.functional.softmax(logits)
        _, prediction= torch.max(logits,1)

        state = classes[prediction.item()]
        label = {test_IDs[ii]:state}
        predictionary.update(label)


    with open('predictions_{}.pickle'.format(modelname), 'wb') as f:
        pickle.dump(predictionary, f, protocol = 2)

    print('Finished predictions for run {}'.format(run))
