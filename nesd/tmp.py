# Pad zero frames after the last frame.
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        mic_pos_xyz_emb = mic_position_emb
        mic_direction_emb = mic_look_direction_emb
        ray_pos_xyz_emb = agent_position_emb
        ray_direction_emb = agent_look_direction_emb

        ray_direction = agent_look_direction
        origin_len = frames_num
        
        # input value FC
        enc1_pool, enc1 = self.input_value_cnn1(x)
        enc2_pool, enc2 = self.input_value_cnn2(enc1_pool)
        enc3_pool, enc3 = self.input_value_cnn3(enc2_pool)
        # a1 = self.input_value_cnn4(a1)
        a1 = enc3_pool

        a1 = a1.permute(0, 2, 1, 3)
        a1 = a1.flatten(2)

        a1 = F.leaky_relu_(self.input_value_fc1(a1), negative_slope=0.01)

        # input angle FC
        a2 = mic_direction_emb.permute(0, 2, 1, 3).flatten(2)
        a2 = F.leaky_relu_(self.input_angle_fc(a2))    # (batch_size, 1, T', C)

        a2_xyz = mic_pos_xyz_emb.permute(0, 2, 1, 3).flatten(2)
        a2_xyz = F.leaky_relu_(self.input_pos_xyz_fc(a2_xyz))

        a2 = a2[:, 0 :: self.time_downsample_ratio, :]
        a2_xyz = a2_xyz[:, 0 :: self.time_downsample_ratio, :]

        x = torch.cat((a1, a2, a2_xyz), dim=-1) # (batch_size, T, C * 2)

        rays_num = ray_direction.shape[1]

        x = torch.tile(x[:, None, :, :], (1, rays_num, 1, 1))
        # (batch_size, outputs_num, T, C * 2)

        # output angle FC
        a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # from IPython import embed; embed(using=False); os._exit(0)
        inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # (batch_size, outputs_num, T, C * 3)
        
        batch_size, rays_num, _T, _C = inter_emb.shape

        x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # (batch_size * outputs_num, T, C * 3)

        x = x.transpose(1, 2)[:, :, :, None]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x[:, :, :, 0].transpose(1, 2)

        x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # (batch_size, outputs_num, T, C * 2)

        output_dict = {}

        if True:
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['agent_see_source'] = cla_output