import rasterio.mask
from affine import Affine
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage import segmentation
import torch.nn.init
from sklearn.preprocessing import MinMaxScaler
import imageio
from sklearn import decomposition
from .__utils import *
#import cv2
########################################################################################################################


def set_global_Cnn_variables(bands=11, convs=2,):
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


def segment_cnn(raster_l, vector_geom, data_path, custom_subsetter=range(5,65),n_band=11, nConv = 1, lr_var=0.15):

    field_counter = "{}{}{}{}{}".format(str(n_band), '_', str(nConv), '_', str(lr_var))
    aggregated_results = pd.DataFrame()
    i = 1
    subsetter_tss = custom_subsetter
    subsetter_tsi = custom_subsetter

    for shp in vector_geom:
        with rasterio.open(raster_l) as src:
            out_image, out_transform = rasterio.mask.mask(src, [shp], crop=True)

            create_mask = True
            if create_mask:
                mask = create_mask_from_ndim(out_image)

        with rasterio.open(raster_l) as src:
            out_image_agg, out_mask_agg = rasterio.mask.mask(src, [shp], crop=True)
            out_image_agg = out_image_agg.copy() / 10000
            out_image_agg = out_image_agg[subsetter_tsi, :, :]
            shape_out_tsi = out_image_agg.shape

            gt_gdal = Affine.to_gdal(out_transform)
            #################################

            out_meta = src.meta

            out_image = out_image.copy() / 10000
            out_image = out_image[subsetter_tsi, :,:]
            shape_out = out_image.shape
            max_valid_pixel = (sum(np.reshape(mask[:, :], (shape_out[1] * shape_out[2])) > 0))
            print('Parcel Area:', max_valid_pixel * 100 / 1000000, ' km²')
            if max_valid_pixel * 100 / 1000000 < 0.05:
                print('pass')
                pass
            else:
                w = np.where(out_image < 0)

                out_sub = mask[:, :]
                mask_local = np.where(out_sub <= 0)
                out_image[w] = 0
                out_image_nan = out_image.copy().astype(dtype=np.float)
                out_image_nan[w] = np.nan
                std_glob = np.nanstd(out_image_nan, axis=(1, 2))
                print('global:', sum(std_glob))

                x, y, z = out_image.shape
                image_2d = np.reshape(out_image, (x, y * z))
                image_2d = np.moveaxis(image_2d.copy(), 0, 1)

                # load image

                three_band_img = out_image_nan

                three_band_img = np.moveaxis(three_band_img, 0, 2)

                img1 = three_band_img
                print(img1.shape)

                re = np.reshape(img1, (img1.shape[0] * img1.shape[1], img1.shape[2]))
                # re_scale = RobustScaler(quantile_range=(0.8, 1)).fit_transform(re)
                scaled = (MinMaxScaler(feature_range=(0, 255)).fit_transform(re))
                scaled_shaped = np.reshape(scaled, (img1.shape))
                #scaled_shaped = np.square(img1+10)
                wh_nan = np.where(np.isnan(scaled_shaped))
                scaled_shaped[wh_nan] = 0

                # file_str = "{}{}{}".format(data_path + "out_img", str(i), ".png")
                # print(file_str)
                # cv2.imwrite(file_str, scaled_shaped[:,:,1])


                # slic # args.compactness
                # argmax bands as input for superpixel segmentation
                arg = (-np.nanstd(out_image_nan, axis=(1, 2))).argsort()[:3]
                arg_50 = (-np.nanstd(out_image_nan, axis=(1, 2))).argsort()[:500]
                arg_10 = []
                for args in arg_50:
                    valid_pixel = (sum(np.reshape(out_image[args, :,:], (shape_out[1]*shape_out[2]))>0))
                    #if np.any(out_image_nan[args, :,:] == 0):
                        #print('scheisse', args)
                    if valid_pixel < max_valid_pixel:
                        print('only:', valid_pixel, 'of:', max_valid_pixel)
                    elif len(arg_10) == 500:
                        break
                    else:
                        arg_10.append(int(args))

                        #file_str = "{}{}{}{}".format(data_path + "/output/out_field", str(args), "_", str(shp_counter))
                        #out_img = (out_image[args,:,:])*255
                        #cv2.imwrite(file_str + '.png', out_img)

                    # count maximum valid pixels
                    # if smaller than maximum valid pixels reject

                print(arg_10)
                #print(arg_10[:3])

                #labels = segmentation.felzenszwalb(scaled_shaped[:,:,arg_10[:3]], scale=10)
                #labels = segmentation.slic(scaled_shaped[:,:,arg], n_segments=6, compactness=6)
                #labels = segmentation.slic(scaled_shaped[:, :, arg], n_segments=90, compactness=15)


                im = scaled_shaped[:, :, arg_10]
                im[im==0] = np.nan


                scaled_arg_2d = np.reshape(im, (im.shape[0] * im.shape[1], len(arg_10)))
                #plt.hist(scaled_arg_2d[:,0], bins = 50)
                #plt.show()

                im[np.isnan(im)] = 0
                adjust = 0

                n_comp = 4 - adjust

                scaled_arg_2d[np.isnan(scaled_arg_2d)] = 0
                print(scaled_arg_2d)
                #################
                # PCA
                n_comps = n_band
                pca = decomposition.PCA(n_components=n_comps)
                im_pca = pca.fit_transform(scaled_arg_2d)
                print(pca.explained_variance_ratio_)
                image_pca = np.reshape(im_pca, (im.shape[0], im.shape[1], n_comps))
                im_pca[im_pca == 0] = np.nan
                print('IMAGE PCA', image_pca.shape)

                #labels = KMeans(n_clusters=5).fit_predict(scaled_arg_2d)
                #labels = segmentation.felzenszwalb(scaled_shaped[:, :, arg_10[:3]], scale=2)  #
                labels = segmentation.slic(image_pca, n_segments=100, compactness=10)

                im = image_pca
                file_str = "{}{}{}".format(data_path + "/output/out_labels", str(field_counter), "_")

                labels_img = np.reshape(labels, (image_pca.shape[0], image_pca.shape[1]))
                # labels__img = np.reshape(labels_, (scaled_shaped.shape[0], scaled_shaped.shape[1]))
                labels_img += 1
                labels_img[mask_local] = 0
                labels = labels_img.reshape(im.shape[0] * im.shape[1])
                print(im.shape, labels.shape)
                plt.imshow(labels_img)
                plt.show()

                #WriteArrayToDisk(labels_img, file_str, gt_gdal, polygonite=True)
                ################################ cnn stuff
                cnn = True
                if cnn:
                    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))


                    visualize = True
                    data = Variable(data)
                    u_labels = np.unique(labels)

                    l_inds = []
                    for i in range(len(u_labels)):
                        l_inds.append(np.where(labels == u_labels[i])[0])

                    # train
                    model = MyNet(data.size(1))

                    #if os.path.exists(data_path + 'cnn_model'):
                        #model = torch.load(data_path + 'cnn_model')
                    model.train()
                    loss_fn = torch.nn.CrossEntropyLoss()
                    optimizer = optim.SGD(model.parameters(), lr=lr_var, momentum=0.8)
                    label_colours = np.random.randint(255, size=(100, n_band))
                    # to create a gif
                    images = []
                    for batch_idx in range(500):
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

                            #im_target_rgb = im_target_rgb.reshape((shp_1, shp_1, 10)).astype(np.uint8)

                            images.append((im_target_rgb[:, :, 0]))


                            #cv2.imshow("output", im_target_rgb[:, :, [0, 1, 2]])
                            #cv2.waitKey(n_band)

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
                        print(output)
                        loss.backward()
                        optimizer.step()

                        # print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
                        print(batch_idx, '/', 500, ':', nLabels, loss.item())

                        if nLabels < 2:
                            print("nLabels", nLabels, "reached minLabels", 2, ".")
                            break
                    torch.save(model, data_path + 'cnn_model')
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
                    #plt.show()

                    file_str = "{}{}{}".format(data_path + "/output/out", str(field_counter), "_")
                    WriteArrayToDisk(flat_array, file_str, gt_gdal, polygonite=True, fieldo=field_counter)

                    # cv2.imwrite(data_path + "/output.png", im_target_rgb)
                    #################################
                    # write raster file
                    # 0 to nan

                    ############################################################
                    # aggregation for the parcels found
                    aggregate = True
                    labels_img = flat_array
                    if aggregate:
                        for classes in np.unique(labels_img):
                            if classes == 0:
                                continue
                            mask = np.array([labels_img == classes])

                            mask_re = np.repeat(mask, shape_out_tsi[0], axis=0)
                            extract = out_image_agg * mask_re
                            extract = extract.astype(dtype=np.float)
                            extract[extract == 0] = np.nan
                            # different metrics
                            std = np.nanstd(extract, axis=(1, 2))
                            # print(field_counter, ':', sum(std))
                            row_mean = (np.nanmean(extract, axis=(1, 2)))
                            # row_median = (np.nanmedian(extract, axis=(1, 2)))
                            # row_percentile = (np.nanpercentile(extract, (10, 30, 70, 90), axis=(1, 2)))
                            # row_min = (np.nanmin(extract, axis=(1, 2)))
                            # row_max = (np.nanmax(extract, axis=(1, 2)))

                            ident = pd.Series(field_counter, name='field')
                            cluster = pd.Series(classes, name='cluster')
                            # ab = itertools.chain(row_mean, row_median, row_min, row_max, ident, cluster)
                            ab = itertools.chain(row_mean, ident, cluster)
                            Series = pd.Series(list(ab), name=i)
                            i += 1
                            aggregated_results = aggregated_results.append(Series, ignore_index=True)
                            print(aggregated_results)
                    #field_counter += 1
    aggregated_results.to_csv(data_path + '/cnn_results.csv')
"""
# global variables
########################################################################################################################
# raster_string = '/PCA_Berlin_Predict.tif'
#vector_string = '/Invekos_clip_3035.shp'
vector_string = 'Vector/Invekos_Grassland_3035.shp'
data_path = 'H:/MA/'
file_path_raster = []

# find all files in the given directory containing either MODIS or bsq
for root, dirs, files in os.walk(data_path + 'output_folder'):
    for file in files:
        if re.match(".*[t][i][f]{1,2}$", file):
            file_path_raster.append(str(root + '/' + file))
        else:
            continue


print(file_path_raster)
# raster_string = file_path_raster[0]

raster_string ='Z:/BB_vrt_stack/BRB_2.vrt'
raster_for_aggregation = 'Z:/BB_vrt_stack/BRB_2.vrt'

#raster_string = file_path_raster[0]
# tsi 2018 = 146, 218 (-1 already)
# tss 2018 = 54, 122
#subsetter = range(54, 122)
subsetter_tsi = range(146, 218)
subsetter = range(0, 980)

raster = gdal.Open(raster_string)
meta = (raster.GetMetadata())
print(meta)

# raster_arr = raster.ReadAsArray()

field = ogr.Open(data_path + vector_string)
field_lyr = field.GetLayer()

jobs = 3

# desoising the entire image?
n_cluster = 4
add_to_cluster = 1


# when smaller than median = more likely that only one land use in parcel
########################################################################################################################
vector_string = 'O:/Student_Data/Wesemeyer/Projekt_/Shapes/Dissolved_3035.gpkg'
with fiona.open(vector_string) as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]
# a mask for the entire image should be generated
# create array where every field gets a number and append to the pca array; kind of a proximity layer;
# could also add more features e.g. field std
raster_string = 'Z:/lower_saxony_sentinel2_TSA_coreg/X0061_Y0046/stack.tif'


"""