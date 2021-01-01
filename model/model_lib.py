import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet3d_xl import Net
from .lib.non_local_dot_product import NONLocalBlock1D, NONLocalBlock2D
from .STRG.rgcn_models import RGCN
import sys
sys.path.append('.')
import utils
from configs import cfg_init


args = cfg_init.model_args()
print(args)

class VideoRegionModel(nn.Module):
    """
    This model built on a set of regions.
    """

    def __init__(self, opt):
        super(VideoRegionModel, self).__init__()

        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames // 2
        self.dataset_name = opt.dataset_name

        self.img_feature_dim = opt.img_feature_dim
        self.coord_feature_dim = opt.coord_feature_dim
        self.pred = opt.pred
        self.hidden_feature_dim = args.hidden_feature_dim

        self.joint = args.joint
        self.vis_info, self.coord_info, self.category_info = args.vis_info, args.coord_info, args.category_info
        self.input_list = [self.vis_info, self.coord_info, self.category_info]
        
        self.i3d = args.i3d if (self.vis_info) else False
        if self.vis_info:
            if self.i3d:
                self.backbone = Net(self.nr_actions, extract_features=True, obtain_arc_id=2, loss_type='softmax', layer_num=50)
            else:
                self.backbone = Net(self.nr_actions, extract_features=True, obtain_arc_id=1, loss_type='softmax', layer_num=50)
            self.conv = nn.Conv3d(2048, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)

        
        region_total_dim = 0
        if self.vis_info:
            self.crop_size = [3, 3]
            self.region_vis_embed = nn.Sequential(
                nn.Linear(self.img_feature_dim * self.crop_size[0] * self.crop_size[1], 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            )
            region_total_dim += 512
        if self.coord_info:
            self.region_coord_embed = nn.Sequential(
                nn.Linear(4, self.coord_feature_dim//2, bias=False),
                nn.BatchNorm1d(self.coord_feature_dim//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
                nn.BatchNorm1d(self.coord_feature_dim),
                nn.ReLU()
            )
            region_total_dim += self.coord_feature_dim
            
        if self.category_info:
            self.category_embed_layer = nn.Embedding(3, self.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
            
        if self.coord_info and self.category_info:
            self.coord_category_fusion = nn.Sequential(
                nn.Linear(self.coord_feature_dim + self.coord_feature_dim // 2, self.hidden_feature_dim, bias=False),
                nn.BatchNorm1d(self.hidden_feature_dim),
                nn.ReLU(inplace=True),
            )

        # if sum(self.input_list)>=2:
        #     self.region_fusion = nn.Sequential(
        #             nn.Linear(region_total_dim, self.hidden_feature_dim),
        #             nn.ReLU(inplace=True),
        #             # nn.Dropout(0.3)
        #         )
        ############################################
        if self.vis_info:
            self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))#
            self.avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))#
        ############################################
        if self.i3d:
            self.temp_dim = (self.nr_frames//2)
        else:
            self.temp_dim = self.nr_frames


        num_feas = 1 if sum(self.input_list[:-1])==1 else 2
        num_feas = 1 if (self.joint and ('pool' not in args.reasoning_module)) else num_feas

        self.fc_dim = None

        if args.reasoning_module == 'pool':
            self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        elif args.reasoning_module == 'pool_T':
            self.avgpool1d = nn.AdaptiveAvgPool1d((1))
            self.navie_temp = nn.Sequential(
                                nn.Linear(self.hidden_feature_dim*self.temp_dim*num_feas, self.hidden_feature_dim*num_feas, bias=False),
                                nn.ReLU()
                            )# should be consisted with STCR_temporal_fusion
        else:
            if self.joint:
                spatial_fea_dim = self.hidden_feature_dim*2
            else:
                spatial_fea_dim = self.hidden_feature_dim

            temp_hidden_dim = self.hidden_feature_dim

            if args.reasoning_module == 'STCR' and self.joint:
                # pass
                spatial_hidden_dim = self.hidden_feature_dim*2
                temp_fea_dim = self.temp_dim * self.hidden_feature_dim * 2
            else:
                spatial_hidden_dim = self.hidden_feature_dim
                temp_fea_dim = self.temp_dim * self.hidden_feature_dim
            if args.reasoning_module == 'STRG':
                self.rgcn = RGCN(in_channel=spatial_fea_dim, out_channel=spatial_fea_dim)
                self.avgpool1d = nn.AdaptiveAvgPool1d((1))
                self.navie_temp = nn.Sequential(
                                nn.Linear(spatial_fea_dim*self.temp_dim, spatial_fea_dim, bias=False),
                                nn.ReLU()
                            )
                self.fc_dim = spatial_fea_dim
            elif args.reasoning_module == 'STNL':
                self.STNL_NLs = nn.Sequential(
                    NONLocalBlock2D(in_channels=spatial_fea_dim)
                )
                self.STNL_reduce = nn.Linear(spatial_fea_dim, self.hidden_feature_dim*num_feas)
                self.STNL_temp = nn.Linear(self.hidden_feature_dim*num_feas*self.temp_dim, self.hidden_feature_dim*num_feas)

            elif args.reasoning_module == 'STCR':
                if args.multiple_interaction:
                    self.STCR_empty_other = nn.Sequential(
                        nn.Linear(spatial_fea_dim*2, spatial_fea_dim, bias=False),
                        nn.ReLU()
                    )
                    self.STCR_hand_hand = nn.Sequential(
                        nn.Linear(spatial_fea_dim*2, spatial_fea_dim, bias=False),
                        nn.ReLU()
                    )
                    self.STCR_obj_obj = nn.Sequential(
                        nn.Linear(spatial_fea_dim*2, spatial_fea_dim, bias=False),
                        nn.ReLU()
                    )
                    self.STCR_hand_obj = nn.Sequential(
                        nn.Linear(spatial_fea_dim*2, spatial_fea_dim, bias=False),
                        nn.ReLU()
                    )
                else:
                    self.STCR_pair_fusion = nn.Sequential(
                        nn.Linear(spatial_fea_dim*2, spatial_fea_dim, bias=False),
                        nn.ReLU()
                    )
                self.STCR_fea_embed = nn.Linear(spatial_fea_dim, spatial_fea_dim, bias=False)
                self.STCR_spaital_fusion = nn.Linear(spatial_fea_dim, spatial_fea_dim, bias=False)
                self.STCR_temporal_fusion = nn.Sequential(
                    nn.Linear(self.temp_dim*spatial_fea_dim, spatial_fea_dim, bias=False),
                    nn.BatchNorm1d(spatial_fea_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_fea_dim, self.hidden_feature_dim, bias=False),
                    nn.BatchNorm1d(self.hidden_feature_dim),
                    nn.ReLU()
                )
                self.STCR_temporal_flow = nn.LSTM(spatial_fea_dim, spatial_fea_dim, num_layers=2)
            else:
                spatial_fea_dim = spatial_fea_dim*2
                self.spatial_node_fusion_list = nn.Sequential(
                    nn.Linear(spatial_fea_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True)
                    )

                self.temporal_node_fusion_list = nn.Sequential(
                    nn.Linear(temp_fea_dim, temp_hidden_dim, bias=False),
                    nn.BatchNorm1d(temp_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(temp_hidden_dim, temp_hidden_dim, bias=False),
                    nn.BatchNorm1d(temp_hidden_dim),
                    nn.ReLU(inplace=True)
                    )
                
        self.non_appearance_encoder = nn.Sequential(
            nn.Linear(self.hidden_feature_dim, self.hidden_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_feature_dim, self.hidden_feature_dim),
            nn.ReLU(inplace=True),
        )

        if self.fc_dim:
            self.fc = nn.Linear(self.fc_dim, self.nr_actions)
        else:
            self.fc = nn.Linear(self.hidden_feature_dim*num_feas, self.nr_actions)

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)

        if self.pred:
            if args.reasoning_module == 'STNL' or 'STCR':
                predictor_input_dim = spatial_fea_dim
                predictor_output_dim = predictor_input_dim // 4
            else:
                predictor_input_dim = self.hidden_feature_dim
            # self.pred_LSTM = nn.LSTM(self.LSTM_input_dim, self.hidden_feature_dim, num_layers=1)


            self.pred_LSTM = self.STCR_temporal_flow
            predictor_output_dim = spatial_fea_dim
            #self.predictor =  nn.Sequential(
            #    nn.Linear((self.temporal_dim//2)*self.hidden_feature_dim, self.hidden_feature_dim, bias=False)
            #)

            self.pos_decoder = nn.Linear(predictor_output_dim, 4)
            self.offset_decoder = nn.Linear(predictor_output_dim, 2)
        
        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)


    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        if not torch.cuda.is_available():
            weights = torch.load(restore_path, map_location=torch.device('cpu'))['state_dict']
        else:
            weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    def forward(self, global_img_input, box_categories, box_input, video_labels=None, is_inference=False):
        """
        #V: num of videos 
        #T: num of frames, --> nr_frames
        #P: num of proposals, --> nr_boxes
        global_img_input: [V x 3 x T x 224 x 224]
        box_categories: [V, T, P]
        box_input:  [V, T, P, 4]
        """
        self.video_labels = video_labels
        self.is_inference = is_inference
        self.full_box_tensor = box_input
        # dim check
        assert (self.nr_frames) == box_input.size(1)
        V, T, _, _ = box_input.size()
        # for test###################
        T = T//2 if self.i3d else T##
        #############################
        ################# build region_features for each object ##########################
        # visual information
        if self.vis_info:
            # dim check
            assert (self.nr_frames) == global_img_input.size(2)
            ## build conv_fea_maps for frames  - [V x 2048 x T x 14 x 14]
            _, org_feas = self.backbone(global_img_input)
            T = org_feas.size(2)
            ### Reduce dimension video_features - [V x 512 x T x 14 x 14], d = 512
            conv_fea_maps = self.conv(org_feas)
            ### get global feas
            global_vis_feas = self.avgpool3d(conv_fea_maps).squeeze() # [V x 512], d = 512
            ### reshape fea_maps
            conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # [V x T x d x 14 x 14]
            conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # [V * T x d x 14 x 14]
        
            ## build region_vis_feas = [V x T x P x D_vis]
            ### sample timesteps and reshape
            box_tensor = box_input[:, ::(self.nr_frames//T), :, :].contiguous() # [V, T, P, 4]
            box_tensor = box_tensor.view(-1, *box_tensor.size()[2:]) # [V*T/2, P, 4]
            ### convert tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
            boxes_list = utils.box_to_normalized(box_tensor, crop_size=[224,224])
            img_size = global_img_input.size()[-2:]
            ### get region feas via RoIAlign
            region_vis_feas = utils.build_region_feas(conv_fea_maps, 
                                                 boxes_list, self.crop_size, img_size) #[V*T*P x C], where C= 3*3*d
            region_vis_feas = region_vis_feas.view(V, T, self.nr_boxes, region_vis_feas.size(-1)) #[V x T x P x C]
            ### reduce dim of region_vis_feas
            region_vis_feas = self.region_vis_embed(region_vis_feas) #[V x T x P x D_vis]

        # coord information
        if self.coord_info or args.reasoning_module=='STRG':
            ## sample timesteps and reshape
            box_tensor = box_input[:, ::(self.nr_frames//T), :, :].contiguous() # [V, T, P, 4]
            self.box_tensor = box_tensor # [V, T, P, 4]
            box_tensor = box_tensor.view(-1, 4) #[V*T*P, 4]
            
        if self.coord_info:
            ## get coord features for each box - [V x T x P x D_coord] ####
            coord_feas = self.region_coord_embed(box_tensor) #[V*T*P, D_coord] where D_coord is the coord_feature_dim
            region_coord_feas = coord_feas.view(V, T, self.nr_boxes, -1) #[V x T x P x D_coord]
        
        if self.category_info or args.multiple_interaction:
            ## sample timesteps and reshape the box_category_tensor
            box_category_tensor = box_categories[:,::(self.nr_frames//T),:].contiguous() # [V, T, P]
            self.box_category_tensor = box_category_tensor
            #box_category_tensor = box_category_tensor.long()
            box_category_tensor = box_category_tensor.view(-1) # [V*T*P]
        
        # category information
        if self.category_info:
            ## build region_category_feas - [V, T, P, D_category]
            box_category_embeddings = self.category_embed_layer(box_category_tensor.long())  # [V x T x P x D_category], D_category=coord_feature_dim//2
            region_category_feas = box_category_embeddings.view(V, T, self.nr_boxes, -1) # [V, T, P, D_category]
        
        fea_dict = {}
        if self.vis_info:
            fea_dict['vis'] = region_vis_feas
            fea_dict['global_vis'] = global_vis_feas
        if self.coord_info:
            fea_dict['coord'] = region_coord_feas
        if self.category_info:
            fea_dict['category'] = region_category_feas

        global_feas = self.late_fusion(fea_dict)
        # prediction - [V x NUM_CALSS]
        #print('global_feas:', global_feas.size())
        cls_output = self.fc(global_feas)

        if self.pred:
            return cls_output, self.pred_loss
        return cls_output
    

    def late_fusion(self, fea_dict):
        fea_list = []
        if self.joint:
            if self.vis_info:
                vis_feas = fea_dict['vis'] # [V x T x P x D_vis]
                fea_list.append(vis_feas)

            if self.coord_info and self.category_info:
                V, T, P, _ = fea_dict['coord'].size()
                concated_feas = torch.cat([fea_dict['coord'], fea_dict['category']], dim=-1) #[V x (D_coord+D_category)]
                concated_feas = concated_feas.view(-1, concated_feas.size(-1))
                concated_feas = self.coord_category_fusion(concated_feas)  # (V*T*P, D_coord)
                concated_feas = concated_feas.view(V, T, P, -1) # (V, T, P, D_coord)
                fea_list.append(self.non_appearance_encoder(concated_feas)) # (V, T, P, D_coord)

            global_feas = torch.cat(fea_list, dim=-1) # [V x T x P x (D_vis+D_coord)]
            if self.pred:
                global_feas, self.pred_loss = self.Auxiliary_Pred(global_feas)
            else:
                if args.reasoning_module=='pool':
                    # maxpool
                    global_feas = global_feas.permute(0, 3, 1, 2) # [V, T, P, (D_vis+D_coord)] ---> [V, (D_vis+D_coord), T, P]
                    global_feas = self.avgpool2d(global_feas).view(global_feas.size(0),-1) # [V x (D_vis+D_coord)]
                elif args.reasoning_module=='pool_T':
                    global_feas = self.pool_T(global_feas)
                else:
                    _, global_feas = eval('self.'+args.reasoning_module)(global_feas, args.reasoning_mode)
        else:
            pass
            # if self.vis_info:
            #     # fea_list.append(fea_dict['global_vis']) # [V x D_vis]
            #     # 'pool' region features [V x T x P x D_vis] ---> [V x D_vis]
            #     ##################################################################################
            #     vis_feas = fea_dict['vis'].permute(0, 3, 1, 2)  # [V x D_vis x T x P]
            #     if args.reasoning_module=='STRG' and not self.coord_info and not self.category_info:
            #         vis_feas = vis_feas.permute(0,2,3,1) # [V x T x P x D_vis]
            #         ### test for STRG, use visual information as the instance-centric feature only
            #         _, vis_feas = eval('self.'+args.reasoning_module)(vis_feas, args.reasoning_mode) #[V x D_vis]
            #     elif args.reasoning_module=='pool':
            #         vis_feas = self.avgpool2d(vis_feas).view(vis_feas.size(0),-1) # [V x D_vis]
            #     elif args.reasoning_module=='pool_T':
            #         vis_feas = vis_feas.permute(0,2,3,1) # [V x T x P x D_vis]
            #         vis_feas = self.pool_T(vis_feas)
            #     ###################################################################################
            #     fea_list.append(vis_feas)
            # if self.coord_info and not self.category_info:
            #     # 'STIN' 
            #     coord_feas = fea_dict['coord'] #[V x T x P x D_coord]
            #     if self.pred:
            #         pass
            #     else:
            #         if args.reasoning_module=='pool':
            #             coord_feas = coord_feas.permute(0, 3, 1, 2) # [V, T, P, D_] ---> [V, D_, T, P]
            #             coord_feas = self.avgpool2d(coord_feas).view(coord_feas.size(0),-1) # [V x D_]
            #         elif args.reasoning_module=='pool_T':
            #             coord_feas = self.pool_T(coord_feas) # [V x D_]
            #         else:
            #             _, coord_feas = eval('self.'+args.reasoning_module)(coord_feas, args.reasoning_mode)
            #     fea_list.append(self.non_appearance_encoder(coord_feas))
            # if self.coord_info and self.category_info:
            #     # fuse and do 'STIN'
            #     V, T, P, _ = fea_dict['coord'].size()
            #     concated_feas = torch.cat([fea_dict['coord'], fea_dict['category']], dim=-1) #[V x (D_coord+D_category)]
            #     concated_feas = concated_feas.view(-1, concated_feas.size(-1))
            #     concated_feas = self.coord_category_fusion(concated_feas)  # (V*T*P, coord_feature_dim)
            #     concated_feas = concated_feas.view(V, T, P, -1) # (V, T, P, coord_feature_dim)
            #     if self.pred:
            #         pass
            #     else:
            #         if args.reasoning_module=='pool':
            #             concated_feas = concated_feas.permute(0, 3, 1, 2) # [V, T, P, D_] ---> [V, D_, T, P]
            #             concated_feas = self.avgpool2d(concated_feas).view(concated_feas.size(0),-1) # [V x D_]
            #         elif args.reasoning_module=='pool_T':
            #             concated_feas = self.pool_T(concated_feas) # [V x D_]
            #         else:
            #             _, concated_feas = eval('self.'+args.reasoning_module)(concated_feas, args.reasoning_mode)
            #     fea_list.append(self.non_appearance_encoder(concated_feas))
            # global_feas = torch.cat(fea_list, dim=-1) #[V x (D_vis*2+D_coord)]
            # # global_feas = self.dropout(global_feas)
        return global_feas
    
    def pool_T(self, inputs):
        # spatial 1D pooling and do navie temporal
        # input [V x T x P x D_vis]
        inputs = inputs.permute(0,1,3,2).contiguous() # [V x T x D_vis x P]
        V, T, _, P = inputs.size()
        inputs = inputs.view(V, -1, P) # [V x T*D_vis x P]
        inputs = self.avgpool1d(inputs).view(V, -1) # [V x T*D_vis]
        outputs = self.navie_temp(inputs) # [V x D_]
        return outputs

    def Auxiliary_Pred(self, states, offset_pred=True):
        self.pred_LSTM.flatten_parameters()
        # predict the latent object-states in the last frame via previous data
        ### use two kinds of feature to predict the location????
        V, T, P, D = states.size()
        S_feas, ST_feas = eval('self.'+args.reasoning_module)(states, args.reasoning_mode)

        pred_pos_list = []
        pred_offset_list = []
        pred_state_list = []

        # observation length
        if self.dataset_name == 'charades':
            obs_len = int(T*args.obs_len_ratio) #16/4*3=12
        else:
            obs_len = T//2

        states = S_feas # <V, T, P, D>
        for t in range(0, T):
            if t>=obs_len:
                previous_states = states[:,t-obs_len:t,...] # <V, obs_len, P, D>
                previous_states = previous_states.permute(1,0,2,3) # <obs_len, V, P, D>
                previous_states = previous_states.contiguous().view(obs_len, V*P, -1) # <obs_len, V*P, D>
                output, _ = self.pred_LSTM(previous_states) # <obs_len, V*P, D>
                pred_state = output[-1].view(V, P, -1) # <V, P, D>
                pred_pos_list.append(self.pos_decoder(pred_state).unsqueeze(1)) # <V, 1, P, 4>
                pred_offset_list.append(self.offset_decoder(pred_state).unsqueeze(1)) # <V, 1, P, 2>

        pred_pos = torch.cat(pred_pos_list, dim=1) # <V, T-obs_len, P, 4>
        gt_pos = self.box_tensor[:,obs_len:,...] # <V, T-obs_len, P, 4>
                
        p_loss = utils.Euclidean_loss(pred_pos, gt_pos)
        loss = p_loss.mean()

        if offset_pred:
            pred_offset = torch.cat(pred_offset_list, dim=1) # <V, T-obs_len, P, 2>
            gt_offset = self.box_tensor[:,obs_len:,:,:2] - self.box_tensor[:,obs_len-1:-1,:,:2] # <V, T-obs_len, P, 2>

            o_loss = utils.Euclidean_loss(pred_offset, gt_offset)
            loss += o_loss.mean()
        return ST_feas, loss#, loc_pred_out

    def STIN(self, input_feas, mode=None, layer_num=0):
        """
        perform the spatio-temporal interaction among input_feas
        # input_feas: [V, T, P, D] where V is the number of video, P and T are the number of regions and frames, respectively. D is the feature dimension.
        """
        V, T, P, D = input_feas.size()
        input_feas = input_feas.permute(0, 2, 1, 3).contiguous() # [V x P x T x D]

        if mode == 'ST':
            # spatial message passing (graph)
            spatial_message = input_feas.sum(dim=1, keepdim=True)  # (V x 1 x T x D)
            # message passed should substract itself, and normalize to it as a single feature
            spatial_message = (spatial_message - input_feas) / (P - 1)  # [V x P x T x D]
            in_spatial_feas = torch.cat([input_feas, spatial_message], dim=-1)  # (V x P x T x 2*D)
            # fuse the spatial feas into temporal ones
            # print('in_spatial_feas size:', in_spatial_feas.size())
            S_feas = self.spatial_node_fusion_list[layer_num](in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
            temporal_feas = S_feas.view(V*P, -1) # [V*P x T*D_fusion]
        elif mode == 'T':
            temporal_feas = input_feas.view(V*P, -1) # [V*P x T*D_fusion]

        # get gloabl features
        node_feas = self.temporal_node_fusion_list[layer_num](temporal_feas)  # (V*P x D_fusion)
        ST_feas = torch.mean(node_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        if mode == 'T':
            return input_feas, ST_feas
        return S_feas.view(V, P, T, -1).contiguous().permute(0,2,1,3), ST_feas

    def STNL(self, inputs, mode=None, layer_num=0):
        V, T, P, D = inputs.size() #
        NLs_out = self.STNL_NLs(inputs.permute(0,3,1,2)) # <V, D, T, P> ---> <V, D, T, P>
        NLs_out = NLs_out.permute(0,2,3,1) # <V, T, P, D>
        NLs_out = self.STNL_reduce(NLs_out) # <V, T, P, D'>
        NLs_out = NLs_out.permute(0,2,1,3).contiguous() # <V, P, T, D'>
        ST_feas = self.STNL_temp(NLs_out.view(V*P, -1)) # <V, P, D_fusion>
        ST_feas = torch.mean(ST_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        return NLs_out.permute(0,2,1,3), ST_feas # <V, D'>

    def STRG(self, rois_features, mode=None, layer_num=0):
        """
        rois: <V,T,P,4>
        rois_features: <V,T,P,D>
        """
        # self.box_tensor # <V,T,P,4>
        ### convert tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
        rois = utils.box_to_normalized(self.box_tensor, crop_size=[224,224], mode='tensor')# <V,T,P,D>
        video_feas = self.rgcn(rois_features, rois) # <V, D>## <V, T, P, D>
        video_feas = self.pool_T(video_feas)
        return None, video_feas

    def STCR(self, inputs, mode=None, layer_num=0, LSTM=False, ADD_TMP=False):
        """
        This function is designed to capture the relative spatio-temporal interaction among the instances in the scene.
        """
        V, T, P, D = inputs.size() #
        ## box_category_tensor: [V*T*P] where 0 is for none, 1 is for hand, and 2 for object.
        # here, we aim at building different interaction between objec-object pairs, hand-hold-object pairs, hand-unhold-object pairs, and hand-hand pairs?
        # We first test three simple pairs, i.e., object-object, hand-hand, hand-object.
        box_idt = self.box_category_tensor.view(V, T, P) # <V, T, P>
        pair_data_list = []
        for p in range(P):
            for pp in range(P):
                pair_data = torch.cat([inputs[...,p,:], inputs[...,pp,:]], dim=-1).unsqueeze(-2) # <V, T, 1, D>
                pair_data_list.append(pair_data)
                #pair_data = torch.cat([inputs[v,t,p,:], inputs[v,t,pp,:]], dim=-1).unsqueeze(1).unsqueeze(1).unsqueeze(1) # <1,1,1,2d>
        assert len(pair_data_list) == P*P, 'wrong num of spatial nodes!'
        pair_data = torch.cat(pair_data_list, dim=-2) # <V, T, P*P, D>

        full_mask = utils.get_fast_mask(box_idt)

        # compositional reasoning #
        # mask: 0: empty-other; 1: hand-hand; 2: hand-obj; 4: obj-obj
        empty_other_mask = utils.get_bin_mask(full_mask, type_ids=[0]) #  <V,T,P,P>
        empty_other_mask = empty_other_mask.view(V,T,-1) #  <V,T,P*P>?????
        empty_other_mask = empty_other_mask.unsqueeze(-1).expand(*empty_other_mask.size(), pair_data.size(-1))
        pair_data_empty_other = self.STCR_empty_other(pair_data*empty_other_mask) # <V,T,P*P,2d>

        hand_hand_mask = utils.get_bin_mask(full_mask, type_ids=[1]) #  <V,T,P,P>
        hand_hand_mask = hand_hand_mask.view(V,T,-1) #  <V,T,P*P>?????
        hand_hand_mask = hand_hand_mask.unsqueeze(-1).expand(*hand_hand_mask.size(), pair_data.size(-1))
        pair_data_hand_hand = self.STCR_hand_hand(pair_data*hand_hand_mask) # <V,T,P*P,2d>

        hand_obj_mask = utils.get_bin_mask(full_mask, type_ids=[2]) #  <V,T,P,P>
        hand_obj_mask = hand_obj_mask.view(V,T,-1) #  <V,T,P*P>?????
        hand_obj_mask = hand_obj_mask.unsqueeze(-1).expand(*hand_obj_mask.size(), pair_data.size(-1))
        pair_data_hand_obj = self.STCR_hand_obj(pair_data*hand_obj_mask) #  <V,T,P*P,2d>

        obj_obj_mask = utils.get_bin_mask(full_mask, type_ids=[4]) #  <V,T,P,P>
        obj_obj_mask = obj_obj_mask.view(V,T,-1) #  <V,T,P*P>?????
        obj_obj_mask = obj_obj_mask.unsqueeze(-1).expand(*obj_obj_mask.size(), pair_data.size(-1))
        pair_data_obj_obj = self.STCR_obj_obj(pair_data*obj_obj_mask) #  <V,T,P*P,2d>

        pair_data = pair_data_empty_other + pair_data_hand_hand + pair_data_obj_obj + pair_data_hand_obj #  <V,T,P*P,2d>
        pair_data.view(V, T, len(pair_data_list), -1)
        # sum spatial
        sp_message_list = []
        for p in range(P):
            sp_message_list.append(torch.sum(pair_data[...,p*P:(p+1)*P,:], dim=-2).unsqueeze(-2)) # <V, T, 1, D>



        # 
        sp_message = torch.cat(sp_message_list, dim=-2) # <V, T, P, D>
        S_feas = self.STCR_spaital_fusion(self.STCR_fea_embed(inputs) + sp_message) # <V, T, P, D>

        if args.LSTM_flow:
            self.STCR_temporal_flow.flatten_parameters()
            ## S_feas <V, T, P, D>
            #seq_len, batch, input_size
            S_feas_ = S_feas.permute(1,0,2,3).contiguous() # <T, V, P, D>
            S_feas_ = S_feas_.view(T, -1, D) # <T, V*P, D>
            temp_outputs, _ = self.STCR_temporal_flow(S_feas_) # <T, V*P, D>
            temp_outputs = temp_outputs.view(T, V, P, -1).contiguous() # <T, V, P, D>
            S_feas_ = temp_outputs.permute(1,0,2,3) # <V, T, P, D>
            S_feas_ = S_feas + S_feas_
        else:
            S_feas_ = S_feas

        # perform temporal modeling
        S_feas_ = S_feas_.permute(0,2,1,3).contiguous() # <V, P', T, D>
        S_feas_ = S_feas_.view(-1, T*D) # <V*P', T*D>
        ST_feas = self.STCR_temporal_fusion(S_feas_)  # (V*P x D_fusion)
        
        # here, build a temporal only fea <V, T, P, D>
        if ADD_TMP:
            T_feas = inputs.permute(0,2,1,3).contiguous().view(V*P, -1) # <V*P, T*D>
            T_feas = self.STCR_temporal_fusion(T_feas) # <V*P, D>
            ST_feas += T_feas

        # mean
        # ST_feas = torch.mean(ST_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        # sum
        ST_feas = torch.sum(ST_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        return S_feas, ST_feas

if __name__ == "__main__":
    opt = cfg.load_main_cfg()
    net = VideoRegionModel(opt)
    opt.batch_size = 1
    V, T, P = opt.batch_size, opt.num_frames//2, 4 # num of videos, frames, proposals
    #global_img_input: [V x 3 x T x 224 x 224]
    #box_categories: [V, T, P]
    #box_input:  [V, T, P, 4]
    global_img_input, box_categories, box_input = torch.rand(V, 3, T, 224, 224), torch.rand(V, T, P), torch.rand(V, T, P, 4)
    output = net(global_img_input, box_categories, box_input)
    print(output.size())