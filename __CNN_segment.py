import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage import segmentation
import torch.nn.init
import imageio

from .__utils import *
import cv2
########################################################################################################################


def set_global_Cnn_variables(bands=11, convs=2):
    """
    CNN needs some global variables
    :param bands: number of bands to be used for CNN
    :param convs: number of convolutions
    :return:
    """
    # CNN model
    global n_band
    global nConv
    global padd
    global stride_
    n_band = bands
    nConv = convs
    padd = np.int((n_band - 1) / 2)
    stride_ = 1


class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, n_band, kernel_size=n_band, stride=stride_, padding=padd)

        self.bn1 = nn.BatchNorm2d(n_band)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv - 1):  # n_conv
            self.conv2.append(nn.Conv2d(n_band, n_band, kernel_size=n_band, stride=stride_, padding=padd))
            self.bn2.append(nn.BatchNorm2d(n_band))
        self.conv3 = nn.Conv2d(n_band, n_band, kernel_size=1, stride=stride_, padding=0)
        self.bn3 = nn.BatchNorm2d(n_band)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        """
        #### experimenting
        x_ = x.detach().numpy()
        x_ = x_.squeeze()
        x_ = np.moveaxis(x_, 0, 2)
        plt.imshow(x_)
        plt.show()
        #### experimenting
        """
        for i in range(nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


def OpenRasterToMemory(path):
    drvMemR = gdal.GetDriverByName('MEM')
    ds = gdal.Open(path)
    dsMem = drvMemR.CreateCopy('', ds)
    return dsMem


def minmax_transfor(X):
    shape = X.shape
    i = shape[0]
    j = shape[1]
    bands = shape[2]
    result = 255 * (img1[i, j, bands] - np.nanmin(img1[:, :, bands])) / (
            np.nanmax(img1[:, :, bands]) - np.nanmin(img1[:, :, bands]))
    return result


class Segmentation():
    def __init__(self, string_to_raster, vector_geom, custom_subsetter, MMU, bands=11, convs=2, ):
        set_global_Cnn_variables(bands, convs)
        three_d_image, two_d_im_pca, mask_local, gt_gdal = prepare_data(string_to_raster, vector_geom, custom_subsetter,
                                                                        n_band, MMU=MMU, PCA=True)


def segment_cnn(string_to_raster, vector_geom, indexo=np.random.randint(0, 100000),
              data_path_output=None, n_band=50, into_pca=50, lr_var=0.1, convs=2,
              custom_subsetter=range(0,80),  MMU=0.05, PCA=True):
    """
    The CNN unsupervised segmentation is based on a paper by Asako Kanezaki;
    "Unsupervised Image Segmentation by Backpropagation"
    on Github: https://github.com/kanezaki/pytorch-unsupervised-segmentation
    :param string_to_raster:
    :param vector_geom:
    :param indexo:
    :param data_path_output:
    :param n_band:
    :param into_pca:
    :param lr_var:
    :param custom_subsetter:
    :param MMU:
    :param PCA:
    :return:
    """
    set_global_Cnn_variables(bands=n_band, convs=convs)
    print(nConv)
    if os.path.exists(data_path_output + 'output'):
        print('output directory already exists')
        # os.rmdir(data_path_output + 'output')
    else:
        os.mkdir(data_path_output + 'output')
    data_path = data_path_output + 'output'

    field_counter = "{}{}{}{}{}{}{}".format(str(n_band), '_', str(nConv), '_', str(lr_var), '_', str(indexo))

    ###########
    # prepare data function does the magic here
    three_d_image, two_d_im_pca, mask_local, gt_gdal, MMU_fail = prepare_data(string_to_raster, vector_geom, custom_subsetter,
                                                                    n_band, MMU=MMU, PCA=PCA, into_pca=into_pca)
    # in case grassland area is too small
    if MMU_fail:
        flat_array = np.zeros(three_d_image.shape[0], three_d_image.shape[1])
    elif three_d_image is None:
        return

    else:

        #labels = segmentation.felzenszwalb(three_d_image, scale=0.1)  #
        labels = segmentation.slic(three_d_image, n_segments=500, compactness=20)

        im = three_d_image
        file_str = "{}{}{}".format(data_path + "/output/out_labels", str(field_counter), "_")

        labels_img = np.reshape(labels, (three_d_image.shape[0], three_d_image.shape[1]))
        # labels__img = np.reshape(labels_, (scaled_shaped.shape[0], scaled_shaped.shape[1]))
        labels_img += 1
        labels_img[mask_local] = 0
        labels = labels_img.reshape(im.shape[0] * im.shape[1])
        print(im.shape, labels.shape)
        plt.imshow(labels_img)
        #plt.show()

        # WriteArrayToDisk(labels_img, file_str, gt_gdal, polygonite=True)
        ################################ cnn stuff
        data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))

        visualize = True
        data = Variable(data)
        u_labels = np.unique(labels)

        l_inds = []
        for i in range(len(u_labels)):
            l_inds.append(np.where(labels == u_labels[i])[0])

        # train
        model = MyNet(data.size(1))

        # if os.path.exists(data_path + 'cnn_model'):
        # model = torch.load(data_path + 'cnn_model')
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr_var, momentum=0.8)
        label_colours = np.random.randint(255, size=(100, n_band))
        # to create a gif
        images = []
        for batch_idx in range(100):
            # forwarding
            optimizer.zero_grad()
            output = model(data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, n_band)
            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))

            if visualize:
                im_target_rgb = np.array([label_colours[c % 100] for c in im_target])

                im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
                shp_1 = (np.sqrt(im_target_rgb.shape[0])).astype(np.int)

                # im_target_rgb = im_target_rgb.reshape((shp_1, shp_1, 10)).astype(np.uint8)

                images.append((im_target_rgb[:, :, 0]))

                cv2.imshow("output", im_target_rgb[:, :, [0, 1, 2]])
                cv2.waitKey(n_band)

            # superpixel refinement
            # TODO: use Torch Variable instead of numpy for faster calculation
            for i in range(len(l_inds)):
                labels_per_sp = im_target[l_inds[i]]
                u_labels_per_sp = np.unique(labels_per_sp)
                hist = np.zeros(len(u_labels_per_sp))
                for j in range(len(hist)):
                    hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
                im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
            target = torch.from_numpy(im_target)

            target = Variable(target)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            # print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
            print(batch_idx, '/', 100, ':', nLabels, loss.item())

            if nLabels < 2:
                print("nLabels", nLabels, "reached minLabels", 2, ".")
                break
        # torch.save(model, data_path + 'cnn_model')
        # save output image
        if not visualize:
            output = model(data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, n_band)
            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        imageio.mimsave(data_path + 'cnn.gif', images)
        print(im_target_rgb.shape)

        flat_array = im_target_rgb[:, :, 0].squeeze()
        flat_array += 1
        flat_array[mask_local] = 0
        print(type(flat_array))

        plt.figure(figsize=(15, 8))
        plt.imshow(flat_array)
        # plt.show()

    file_str = "{}{}".format(data_path + "/CNN_", str(field_counter))
    WriteArrayToDisk(flat_array, file_str, gt_gdal, polygonite=True, fieldo=field_counter)


