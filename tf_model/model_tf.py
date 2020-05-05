""" Estimate 3D Hand Pose through binary Mask
Author: Wayne Lee
"""
import os
from importlib import import_module
import numpy as np
from utils.iso_boxes import iso_cube
import tensorflow as tf
from collections import namedtuple
from model.mv_silhouette_net import mv_silhouette_net
import matplotlib.pyplot as mpplot

class mv_fpn_sn_2(mv_silhouette_net):
    """ 
    End-to-end 3D hand pose estimation from a single binary mask
    This class use clean_depth (128, 128, 1), clean_binary(128, 128, 1)
    Plus Multiview data (128, 128, 3)
    'MV' stands for Multi-View
    """
 
    def __init__(self, args):
        super(mv_fpn_sn_2, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_ortho3b'
        )
        
        self.crop_size=128
  
    def fetch_batch(self, mode='train', fetch_size=None):
        if fetch_size is None:
            fetch_size = self.batch_size
        batch_end = self.batch_beg + fetch_size
        if batch_end >= self.split_end:
            return None
        store_handle = self.store_handle[mode]
        
        self.batch_data['batch_frame'] = \
            store_handle['ortho3b'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_depth'] = \
            store_handle['ortho3'][self.batch_beg:batch_end, ...]  
        self.batch_data['batch_poses'] = \
            store_handle['pose_c'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_index'] = \
            store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(mv_silhouette_net, self).receive_data(thedata, args)
        self.store_name = {
            'index': thedata.annotation,
            'poses': thedata.annotation,
            'resce': thedata.annotation,
            'pose_c': 'pose_c',
            'ortho3': 'ortho3_{}'.format(self.crop_size),
            'ortho3b': 'ortho3b_{}'.format(self.crop_size),
        }
        
        self.frame_type = 'ortho3b'

    def draw_random(self, thedata, args):
        
        ## TODO: this should be fixed
        
        # mode = 'train'
        mode = 'test'
        store_handle = self.store_handle[mode]
        index_h5 = store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        frame_id = 598
        frame_id = 239
        img_id = index_h5[frame_id, ...]
        frame_h5 = store_handle['ortho3b'][frame_id, ...]
        poses_h5 = store_handle['pose_c'][frame_id, ...].reshape(-1, 3)
        resce_h5 = store_handle['resce'][frame_id, ...]
        d_h5 = store_handle['ortho3'][frame_id, ...]
        #ov3edt2_h5 = store_handle['ov3edt2'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        fig, _ = mpplot.subplots(nrows=3, ncols=4, figsize=(4 * 5, 3 * 5))
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)

        ax = mpplot.subplot(3, 4, 1)
        mpplot.gca().set_title('test image - {:d}'.format(img_id))
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(self.data_inst.images_join(img_name, mode))
        

        
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        pose_raw = self.yanker(poses_h5, resce_h5, self.caminfo)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])

        for spi in range(3):
            ax = mpplot.subplot(3, 4, spi + 2)
            img = frame_h5[..., spi]
            ax.imshow(self.to_binary_fill(img), cmap=mpplot.cm.binary)#, cmap=mpplot.cm.bone_r)
            d = d_h5[..., spi]
            import scipy.misc
            scipy.misc.imsave('results_images/bb'+str(spi)+'.jpg', self.to_binary_fill(img))
            scipy.misc.imsave('results_images/dd'+str(spi)+'.jpg', d)
            # pose3d = poses_h5
            pose3d = cube.trans_scale_to(poses_h5)
            pose2d, _ = cube.project_ortho(pose3d, roll=spi, sort=False)
            pose2d *= self.crop_size
            args.data_draw.draw_pose2d(
                ax, thedata,
                pose2d,
            )

#         from utils.image_ops import draw_edt2
#         joint_id = self.join_num - 1
#         for spi in range(3):
#             ax = mpplot.subplot(3, 4, spi + 6)
#             edt2 = ov3edt2_h5[..., spi * self.join_num + joint_id]
#             draw_edt2(fig, ax, edt2)

#         joint_id = self.join_num - 1 - 9
#         for spi in range(3):
#             ax = mpplot.subplot(3, 4, spi + 10)
#             edt2 = ov3edt2_h5[..., spi * self.join_num + joint_id]
#             draw_edt2(fig, ax, edt2)

        fig.tight_layout()
        mpplot.savefig(os.path.join(
            self.predict_dir,
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))
        
    def to_binary_fill(self, img_ndarray):
 
        data = img_ndarray    
        stack = set(((0, 0),))
        idx = 0
        while stack:
            x, y = stack.pop()
            if data[x][y] == 1.0:
                data[x][y] = 0.5
                if x > 0:
                    stack.add((x - 1, y))
                if x < (128 - 1):
                    stack.add((x + 1, y))
                if y > 0:
                    stack.add((x, y - 1))
                if y < (128 - 1):
                    stack.add((x, y + 1))
      
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i][j] == 0.5:
                    data[i][j] = 1.0
                else :
                    data[i][j] = 0.0
        
        img_ndarray_binary_fill = data

        return img_ndarray_binary_fill

    def get_model( self, input_tensor, is_training, bn_decay, regu_scale, scope=None, hg_repeat=2):
        """ input_tensor: BxHxWxC
             out_dim: Bx(Jx3), where J is number of joints
        """
        end_points = {}
        self.end_point_list = []
        
        ### Size of heatmap ###
        #num_feature = 32
        num_feature = 64
        num_joint = self.join_num
        
        final_endpoint='stage_out'
        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        from tensorflow.contrib import slim
        from model.incept_resnet import incept_resnet
        from model.style_content_mv import style_content_mv
        from model.hourglass import hourglass

        with tf.variable_scope( scope, self.name_desc, [input_tensor] ):

            bn_epsilon = 0.001

            with \
                slim.arg_scope( [slim.batch_norm],
                                       is_training=is_training,
                                       epsilon=bn_epsilon,
                                       decay=bn_decay), \
                slim.arg_scope( [slim.dropout],
                                       is_training=is_training), \
                slim.arg_scope( [slim.fully_connected],
                                       weights_regularizer=slim.l2_regularizer(regu_scale),
                                       biases_regularizer=slim.l2_regularizer(regu_scale),
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=slim.batch_norm), \
                slim.arg_scope( [slim.max_pool2d, slim.avg_pool2d],
                                       stride=2, padding='SAME'), \
                slim.arg_scope( [slim.conv2d_transpose],
                                       stride=2, padding='SAME',
                                       weights_regularizer=slim.l2_regularizer(regu_scale),
                                       biases_regularizer=slim.l2_regularizer(regu_scale),
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=slim.batch_norm), \
                slim.arg_scope( [slim.conv2d],
                                       stride=1, padding='SAME',
                                       weights_regularizer=slim.l2_regularizer(regu_scale),
                                       ## For style_content testing 
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                       ## For style_content testing
                                       biases_regularizer=slim.l2_regularizer(regu_scale),
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=slim.batch_norm):

                ### FPN-styled conent Module ###
                with tf.variable_scope('fpn_downscale_encoder'):
                    
                    sc = 'content_code'
                    
                    p1, p2, p3, content_code = style_content_mv.fpn_downscale(input_tensor, scope=sc, reuse=False)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, content_code):
                        return content_code, end_points
                    
                    # (p1, p2, p3, p4) = (128, 64, 32, 16)
                    
                with tf.variable_scope('fpn_upscale_decoder'):  
                
                    sc = 'styled_guidance_map'
                    # This should be smaller, like 8x8
                    # (?, 16, 16, 128)
                    
                    d1, d2, d3, styled_guidance_map = style_content_mv.fpn_upscale(p1, p2, p3, content_code, scope=sc, reuse=False)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, styled_guidance_map):
                        return styled_guidance_map, end_points
                    
#                 with tf.variable_scope('style_content_encoder'):           
#                     sc = 'content_code'
#                     # (?, 128, 128, 3)
#                     style_aware_content_code = style_content_mv.encoder(styled_guidance_map, scope=sc, reuse=True)
#                     if add_and_check_final('style_aware_content_code', style_aware_content_code):
#                         return style_aware_content_code, end_points
                    
                with tf.variable_scope('multiscale_heat_map'):   
                    
                                      
                    sc = 'hmap128'
                    br0 = slim.conv2d(d3, 21*3, 3, stride=1)
                    br0 = slim.max_pool2d(br0, 3, stride=2)
                    # (?, 64, 64, 63)
 
                    sc = 'hmap64'
                    br1 = slim.conv2d(d2, 21*3, 3, stride=1)
                    # (?, 64, 64, 63)

                    net = br0 + br1 
               
                    sc = 'hmap32'
                    net = slim.max_pool2d(net, 3, stride=2)
                    # ( net = 32, 32 63)
                    
                    br2 = slim.conv2d(d1, 21*3, 3, stride=1)
                    # (?, 32, 32, 63)
                    
                    sc = 'mv_hmap32'
                    net = net + br2
                    net = slim.conv2d(net, 21*3, 3, stride=1)
                    # (?, 32, 32, 63)
                    
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                
                
                ### Residual Module ###
                with tf.variable_scope('stage128'):
                    sc = 'stage128_image'
                    # (?, 128, 128, 3)
                    net = slim.conv2d(input_tensor, 8, 3)
                    # (?, 128, 128, 8)
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    # (? , 64, 64, 16)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points

                    sc = 'stage64_image'
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    # (?, 32, 32, 32)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    
                    sc = 'stage32_pre'
                    net = incept_resnet.resnet_k(net, scope='stage32_res')
                    net = slim.conv2d(net, num_feature, 1, scope='stage32_out')
                    #(?, 32, 32, 64)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    
                for hg in range(hg_repeat):
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        
                        ## TODO: check if hourglass useful
                        # Add hg_net, check this out later
                        branch0 = hourglass.hg_net(net, 2, scope=sc + '_hg')            
                        branch0 = incept_resnet.resnet_k(branch0, scope='_res')
                        
                        # Multiply bt 3 here
                        # Styled Map becomes 63
                        net_maps = slim.conv2d( branch0, num_joint*3, 1, normalizer_fn=None, activation_fn=None)
                        # (? 32, 32, 63)

                        self.end_point_list.append(sc)
                        if add_and_check_final(sc, net_maps):
                            return net_maps, end_points
                        
                        branch1 = slim.conv2d(net_maps, num_feature, 1)
                        net = net + branch0 + branch1   
                    
                with tf.variable_scope('stage32'):
                    sc = 'stage32_post'
                    
                    ## Why max pool only?
                    #net = incept_resnet.conv_maxpool(net, scope=sc)
                    net = slim.max_pool2d(net, 3, scope=sc)    
                    #(?, 16, 16, 64)
                    
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points

                with tf.variable_scope('stage16'):
                    sc = 'stage16_image'
                    # (?, 8, 8, 128)
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points

                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    # (?, 63)
                    net = incept_resnet.pullout8(net, self.out_dim, is_training, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

        return net, end_points
    
    @staticmethod
    def smooth_l1(xa):
        return tf.where(
            1 < xa,
            xa - 0.5,
            0.5 * (xa ** 2)
        )

    
    def placeholder_inputs(self, batch_size=None):
        
#         if batch_size is None:
#             batch_size=self.batch_size
        
        frames_tf = tf.placeholder(
                            tf.float32,
                            shape=(batch_size, self.crop_size, self.crop_size, 3) )
        
        depth_frames_tf = tf.placeholder(
                            tf.float32,
                            shape=(batch_size, self.crop_size, self.crop_size, 3) )

        poses_tf = tf.placeholder(
                           tf.float32,
                           shape=(batch_size, self.out_dim))
        
        # No input guidance map here, currently
        
        Placeholders = namedtuple("Placeholders", "frames_tf depth_frames_tf poses_tf")
        return Placeholders(frames_tf, depth_frames_tf, poses_tf)
   

    def get_loss(self, pred, depth_frame, echt, end_points):
        """ simple sum-of-squares loss
            pred: Batch x Joints
            echt: Batch x Joints
        """
        loss_pred_l2  = tf.nn.l2_loss(pred - echt)  
        
        #loss_lc  = tf.reduce_mean(tf.abs(end_points['style_aware_content_code'] - end_points['content_code']))
        loss_lc = 0
        loss_ld  = tf.reduce_mean(tf.abs(end_points['styled_guidance_map'] - depth_frame))
        
        loss_edt = 0
        
        ## TODO: There must have some problems here
        for name, net in end_points.items():
            if name.startswith('hourglass_'):
                loss_edt += tf.reduce_mean(self.smooth_l1(tf.abs(net - end_points['mv_hmap32'])))
                
        loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        return loss_pred_l2, loss_lc, loss_ld, loss_edt, loss_reg

    def get_loss_eval(self, pred, echt):
        """ simple sum-of-squares loss
            pred: BxHxWx(J*5)
            echt: BxHxWx(J*5)
        """
        loss_l2 = tf.nn.l2_loss(pred - echt) 
        loss_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss_l2, loss_reg
    
# evaluating mv_fpn_sn_2 ...
# figures saved: mv_fpn_sn_2_error_bar.png
# 19-03-07 16:05:07 [INFO ]  maximal per-joint mean error: 10.197790059651174
# 19-03-07 16:05:07 [INFO ]  mean error: [6.95437662]

# evaluating mv_fpn_sn_2 ...
# figures saved: mv_fpn_sn_2_error_bar.png
# 19-03-08 01:09:14 [INFO ]  maximal per-joint mean error: 9.33265960364377
# 19-03-08 01:09:14 [INFO ]  mean error: [6.27086674]

# evaluating mv_fpn_sn_2 ...
# figures saved: mv_fpn_sn_2_error_bar.png
# 19-03-10 03:55:11 [INFO ]  maximal per-joint mean error: 7.6343988426485465
# 19-03-10 03:55:11 [INFO ]  mean error: [5.6026745]

