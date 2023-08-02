# Copyright (c) Microsoft Corporation.
# Copyright (c) University of Florida Research Foundation, Inc.
# Licensed under the MIT License.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# IN THE SOFTWARE.
#
# demo_colmap.py
# Demo script for running pre-trained models on data loaded directly from colmap sparse reconstruction files
# Author: Francesco Pittaluga

import os
import sys

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import colmap.database as database
import colmap.read_model as read_model

import utils as ut
import load_data as ld
from models import VisibNet
from models import CoarseNet
from models import RefineNet

def create_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


################################################################################
def load_points_colmap(database_fp,points3D_fp):
    
    db = database.COLMAPDatabase.connect(database_fp)
    descriptors = dict(
        (image_id, database.blob_to_array(data, np.float32, (rows, cols)))
        for image_id,data,rows,cols in db.execute("SELECT image_id, data, rows, cols FROM descriptors"))
    #print(list(descriptors.keys()))

    points3D = read_model.read_points3d_binary(points3D_fp)    
    keys = list(points3D.keys())
    #print("keys")
    #print(keys[:10])
    #exit(1)
    
    pcl_xyz = []
    pcl_rgb = []
    pcl_sift = []
    for pt3D in points3D.values():
        pcl_xyz.append(pt3D.xyz)
        pcl_rgb.append(pt3D.rgb)
        i = np.random.randint(len(pt3D.image_ids),size=1)[0]
        #print(descriptors[pt3D.image_ids[i]].shape)
        #print(pt3D.point2D_idxs[i])
        desc_float = descriptors[pt3D.image_ids[i]][pt3D.point2D_idxs[i]]
        
        ## KO [0,1] float -> [0,255] uint8
        #old_min, old_max = np.min(desc_float), np.max(desc_float)
        #new_min, new_max = 0, 255
        #desc = new_min + (desc_float - old_min) * (new_max -
        #        new_min) / (old_max - old_min)
        #desc = desc.astype(np.uint8)
        #pcl_sift.append(desc)


        # Emulating  FeatureDescriptorsToUnsignedByte
        # from colmap/src/colmap/feature/utils.cc
        # because Hloc keeps features between [0,1]
        desc = np.round(512.0 * desc_float)
        # emulate TruncateCast
        # T1: float
        # T2: uint8
        #static_cast<T1>(std::numeric_limits<T2>::max()) = 255.0
        # static_cast<T1>(std::numeric_limits<T2>::min()) = 0
        desc = np.minimum(255.0, np.maximum(0.0, desc))
        desc = desc.astype(np.uint8)
        pcl_sift.append(desc)

        #pcl_sift.append(desc_float)
        
    pcl_xyz = np.vstack(pcl_xyz).astype(np.float32)
    pcl_rgb = np.vstack(pcl_rgb).astype(np.uint8)
    pcl_sift = np.vstack(pcl_sift).astype(np.uint8)
    
    #images_d = {}
    #images = db.execute("SELECT * FROM images")
    #for image in images:
    #    image_id, image_nm, camera_id = image[0 : 3]
    #    print(image_id, image_nm) # 1, seq2/frame00008.png
    #    images_d[image_id] = image_nm

    return pcl_xyz, pcl_rgb, pcl_sift


def load_points_colmap_only(database_fp,points3D_fp):
    #points3D = read_model.read_points3d_binary(points3D_fp)    
    points3D = read_model.read_points3D_text(points3D_fp.replace(".bin",
        ".txt")) 
    keys = list(points3D.keys())
    
    pcl_xyz = []
    pcl_rgb = []
    pcl_sift = []
    for pt3D in points3D.values():
        pcl_xyz.append(pt3D.xyz)
        pcl_rgb.append(pt3D.rgb)
        
    pcl_xyz = np.vstack(pcl_xyz).astype(np.float32)
    pcl_rgb = np.vstack(pcl_rgb).astype(np.uint8)

    return pcl_xyz, pcl_rgb


def main(prm):
    # set paths for model wts
    vnet_wts_fp = 'wts/pretrained/{}/visibnet.model.npz'.format(prm.input_attr)
    cnet_wts_fp = 'wts/pretrained/{}/coarsenet.model.npz'.format(prm.input_attr)
    rnet_wts_fp = 'wts/pretrained/{}/refinenet.model.npz'.format(prm.input_attr)
    
    ## set paths for colmap files
    #scene = 'nyu_bedroom_0041' if prm.dataset == 'nyu' else 'megadepth_0117_dense0'
    #cmap_database_fp = 'data/demo_colmap_outputs/{}/database.db'.format(scene)
    #cmap_points3D_fp = 'data/demo_colmap_outputs/{}/points3D.bin'.format(scene)
    #cmap_cameras_fp = 'data/demo_colmap_outputs/{}/cameras.bin'.format(scene)
    #cmap_images_fp = 'data/demo_colmap_outputs/{}/images.bin'.format(scene)
    
    cmap_database_fp = '%s/database_with_desc.db'%(prm.colmap_model_path)
    cmap_points3D_fp = '%s/points3D.bin'%(prm.colmap_model_path)
    cmap_cameras_fp = '%s/cameras.bin'%(prm.colmap_model_path)
    cmap_images_fp = '%s/images.bin'%(prm.colmap_model_path)
 

    rec_cmap_database_fp = '%s/database_with_desc.db'%(prm.rec_colmap_model_path)
    rec_cmap_points3D_fp = '%s/points3D.bin'%(prm.rec_colmap_model_path)
    rec_cmap_cameras_fp = '%s/cameras.bin'%(prm.rec_colmap_model_path)
    rec_cmap_images_fp = '%s/images.bin'%(prm.rec_colmap_model_path)

    ################################################################################
    
    # Load point cloud with per-point sift descriptors and rgb features from
    # colmap database and points3D.bin file from colmap sparse reconstruction
    print('Loading point cloud...')
    #pcl_xyz, pcl_rgb, pcl_sift = ld.load_points_colmap(cmap_database_fp, cmap_points3D_fp)

    # use original rec
    if prm.mode == "original":
        pcl_xyz, pcl_rgb, pcl_sift = load_points_colmap(cmap_database_fp, cmap_points3D_fp)
    else:
        # use original descriptors and reconstructed point cloud 
        _, _, pcl_sift = load_points_colmap(cmap_database_fp, cmap_points3D_fp)
        pcl_xyz, pcl_rgb = load_points_colmap_only(rec_cmap_database_fp, rec_cmap_points3D_fp)
        print('Done!')

    #return
    
    # Load camera matrices and from images.bin and cameras.bin files from
    # colmap sparse reconstruction
    print('Loading cameras...')
    K,R,T,h,w,image_names = ld.load_cameras_colmap(cmap_images_fp, cmap_cameras_fp)
    print('Done!')
    
    # Generate projections
    proj_depth = []
    proj_sift = [] 
    proj_rgb = []
    proj_names = []
    indices = list(range(len(K))[::(len(K)//prm.num_samples)])
    print(indices)

    for i in indices:
        print("%d / %d"%(i, len(K)))
        proj_mat = K[i].dot(np.hstack((R[i],T[i])))
        pdepth, prgb, psift = ld.project_points(pcl_xyz, pcl_rgb, pcl_sift,
                                                proj_mat, h[i], w[i], prm.scale_size, prm.crop_size)    
        proj_depth.append((pdepth)[None,...])
        proj_sift.append((psift)[None,...])
        proj_rgb.append((prgb)[None,...])

        proj_names.append(image_names[i])
        print(image_names[i])
        
    proj_depth = np.vstack(proj_depth)
    proj_sift = np.vstack(proj_sift)
    proj_rgb = np.vstack(proj_rgb)
    
    print("Data preparation OK")
    
    ################################################################################
    
    # Build Graph
    proj_depth_p = tf.placeholder(tf.float32,shape=[1,prm.crop_size,prm.crop_size,1])
    proj_rgb_p = tf.placeholder(tf.uint8,shape=[1,prm.crop_size,prm.crop_size,3])
    proj_sift_p = tf.placeholder(tf.uint8,shape=[1,prm.crop_size,prm.crop_size,128])
    
    pdepth = proj_depth_p
    prgb = tf.to_float(proj_rgb_p)
    psift = tf.to_float(proj_sift_p)
    
    keep = prm.pct_3D_points/100.
    pdepth = tf.nn.dropout(pdepth,keep,noise_shape=[1,prm.crop_size,prm.crop_size,1],seed=0)*keep
    prgb = tf.nn.dropout(prgb,keep,noise_shape=[1,prm.crop_size,prm.crop_size,1],seed=0)*keep
    psift = tf.nn.dropout(psift,keep,noise_shape=[1,prm.crop_size,prm.crop_size,1],seed=0)*keep
    valid = tf.greater(pdepth,0.)
    
    # set up visibnet
    if prm.input_attr=='depth':
        vinp = pdepth
    elif prm.input_attr=='depth_rgb':
        vinp = tf.concat((pdepth, prgb/127.5-1.),axis=3)
    elif prm.input_attr=='depth_sift':
        vinp = tf.concat((pdepth, psift/127.5-1.),axis=3)
    elif prm.input_attr=='depth_sift_rgb':
        vinp = tf.concat((pdepth, psift/127.5-1., prgb/127.5-1.),axis=3)
    vnet = VisibNet(vinp,bn='test')
    vpred = tf.logical_and(tf.greater(vnet.pred,.5),valid)
    vpredf = tf.to_float(vpred)*0.+1.
    
    # set up coarsenet 
    if prm.input_attr=='depth':
        cinp = pdepth*vpredf
    elif prm.input_attr=='depth_rgb':
        cinp = tf.concat((pdepth*vpredf, prgb*vpredf/127.5-1.),axis=3)
    elif prm.input_attr=='depth_sift':
        cinp = tf.concat((pdepth*vpredf, psift*vpredf/127.5-1.),axis=3)
    elif prm.input_attr=='depth_sift_rgb':
        cinp = tf.concat((pdepth*vpredf, psift*vpredf/127.5-1., prgb*vpredf/127.5-1.),axis=3)
    cnet = CoarseNet(cinp,bn='test')
    cpred = cnet.pred
    
    # set up refinenet
    rinp = tf.concat((cpred,cinp),axis=3)
    rnet = RefineNet(rinp,bn='train')
    rpred = rnet.pred
    
    # scale outputs
    cpred = (cpred+1.)*127.5
    rpred = (rpred+1.)*127.5
    
    ################################################################################
    
    # Run Graph
    sess=tf.Session()
    try: init_all_vars = tf.global_variables_initializer()
    except: init_all_vars = tf.initialize_all_variables()
    
    # Load net wts
    vnet.load(sess,vnet_wts_fp)
    cnet.load(sess,cnet_wts_fp)
    rnet.load(sess,rnet_wts_fp)
    sess.run([vnet.unset_ifdo,
              cnet.unset_ifdo,
              rnet.unset_ifdo])
    
    # Run cnet
    for i in range(prm.num_samples):
        print("Run %d / %d"%(i, prm.num_samples))
        #fp = 'viz/cambridge/%s/%s_%s'%(
        fp = '%s/%s_%s'%(
                prm.output_path,
                #prm.scene,
                prm.input_attr,
                proj_names[i].replace("/","_"))
        print("Save to %s"%fp)
        #print(proj_names[i])

        vpred_img = []
        cpred_img = []
        rpred_img = []
        valid_img = []

        fd = {proj_depth_p:proj_depth[i:i+1],
              proj_rgb_p:proj_rgb[i:i+1],
              proj_sift_p:proj_sift[i:i+1]}
        out = sess.run([vpred,cpred,rpred,valid],feed_dict=fd)
        vpred_img.append(out[0])
        cpred_img.append(out[1])
        rpred_img.append(out[2])
        valid_img.append(out[3])

        vpred_img = np.vstack(vpred_img)
        cpred_img = np.vstack(cpred_img)
        rpred_img = np.vstack(rpred_img)
        valid_img = np.vstack(valid_img)

        img_path = "%s/%s"%(prm.image_path, proj_names[i])
        print("img_path: %s"%img_path)
        original_img = Image.open(img_path)
        # resize
        new_h, new_w = out[0].shape[1:3]
        print(out[0].shape)
        original_img = original_img.resize((new_w, new_h))
        original_img = np.array(original_img)

        ################################################################################
        
        # Generate visibnet visualization
        vpred_np = np.vstack(vpred_img)
        valid_np = np.vstack(valid_img)
        zero = np.zeros(valid_np.shape,dtype=bool)
        vpred_img = np.ones([vpred_np.shape[0],prm.crop_size,3])*255.
        vpred_img[np.dstack((valid_np,valid_np,valid_np))] = 0.
        vpred_img[np.dstack((np.logical_and(valid_np,np.logical_not(vpred_np)),zero,zero))] = 255.
        vpred_img[np.dstack((zero,zero,np.logical_and(valid_np,vpred_np)))] = 255.
        
        # Build results montage
        header_size = 60
        mntg = np.hstack((vpred_img.astype(np.uint8),
                          np.vstack(cpred_img).astype(np.uint8),
                          np.vstack(rpred_img).astype(np.uint8),
                          original_img.astype(np.uint8),
                          ))
        #header_bot = np.ones((header_size,prm.crop_size*3,3))*127.
        #header_top = np.zeros((header_size,prm.crop_size*3,3))
        header_bot = np.ones((header_size,prm.crop_size*4,3))*127.
        header_top = np.zeros((header_size,prm.crop_size*4,3))

        mntg = np.vstack((header_top,header_bot,mntg))
        
        # Add titles to montage header
        mntg = Image.fromarray(mntg.astype(np.uint8))
        im_draw = ImageDraw.Draw(mntg)
        font = ImageFont.truetype("FreeMonoBold.ttf", 36)
        column_titles = ['VisibNet Prediction','CoarseNet Prediction','RefineNet Prediction']
        figure_title = 'Input Attributes: ' + prm.input_attr.replace('_',', ')
        for i in range(len(column_titles)):
            xpos = prm.crop_size*i + prm.crop_size/2 - font.getsize(column_titles[i])[0]/2
            im_draw.text((xpos,70), column_titles[i], font=font, fill=(255,255,255))
        xpos = header_top.shape[1]/2-font.getsize(figure_title)[0]/2
        im_draw.text((xpos,10), figure_title, font=font, fill=(255,255,255))
        
        # Save montage
        #print(i)
        #print(len(proj_names))
        print(fp)
        create_parent_dir(fp)
        print('Saving visualization to {}...'.format(fp))
        mntg.save(fp)
        print('Done!')


if __name__=="__main__":
    parser = ut.MyParser(description='Configure')
    parser.add_argument("--colmap_model_path", type=str)

    parser.add_argument("--input_attr", type=str, default='depth_sift_rgb',
                        choices=['depth','depth_sift','depth_rgb','depth_sift_rgb'],
                        help="%(type)s: Per-point attributes to inlcude in input tensor (default: %(default)s)")
    parser.add_argument("--pct_3D_points", type=float, default=100., choices=[20,60,100],
                        help="%(type)s: Percent of available 3D points to include in input tensor (default: %(default)s)")
    parser.add_argument("--dataset", type=str, default='nyu', choices=['nyu','medadepth'],
                        help="%(type)s: Dataset to use for demo (default: %(default)s)")
    parser.add_argument("--crop_size", type=int, default=512, choices=[256,512],
                        help="%(type)s: Size to crop images to (default: %(default)s)")
    parser.add_argument("--scale_size", type=int, default=512, choices=[256,394,512],
                        help="%(type)s: Size to scale images to before crop (default: %(default)s)")
    parser.add_argument("--num_samples", type=int, default=32,
                        help="%(type)s: Number of samples to process/visualize (default: %(default)s)")
    
    parser.add_argument("--scene", type=str, default="ShopFacade")
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--rec_colmap_model_path", type=str)
    parser.add_argument("--mode", type=str)
    prm = parser.parse_args()
    
    if prm.scale_size < prm.crop_size: parser.error("SCALE_SIZE must be >= CROP_SIZE")
    if prm.num_samples <= 0: parser.error("NUM_SAMPLES must be > 0")
    
    prm_str = 'Parameters:\n'+'\n'.join(['{} {}'.format(k.upper(),v) for k,v in vars(prm).items()])
    print(prm_str+'\n')

    main(prm)
